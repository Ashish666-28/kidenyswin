"""Image-level 4-class and pixel-level 3-mask inference. Research only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image,ImageOps
from kv_multitask import SwinMultiTask,DISEASES,MASK_CLASSES
from kv_data import raster_normalize,window_hu,save_json
from kv_infer import predict_slice,overlay,components_2d,infer_nifti

def load_model(ckpath,device='cuda'):
    c=torch.load(str(ckpath),map_location='cpu',weights_only=True)
    if list(c.get('classes',[]))!=list(DISEASES) or list(c.get('mask_classes',[]))!=list(MASK_CLASSES):
        raise ValueError('Unexpected class order in multitask checkpoint')
    m=SwinMultiTask(c['encoder_config']);m.load_state_dict(c['model'],strict=True)
    m.segmentation_thresholds=list(map(float,c.get('segmentation_thresholds',[.5,.5,.5])))
    return m.to(device).eval(),c

def read_2d(path,manual_spacing=None,spacing_source=None):
    path=Path(path);spacing=manual_spacing;source=spacing_source
    if path.suffix.lower()=='.dcm':
        import pydicom
        ds=pydicom.dcmread(str(path))
        if str(getattr(ds,'Modality',''))!='CT' or int(getattr(ds,'NumberOfFrames',1))!=1:
            raise ValueError('Expected single-frame CT DICOM')
        if str(getattr(ds,'PhotometricInterpretation','MONOCHROME2'))!='MONOCHROME2':raise ValueError('Unsupported photometric interpretation')
        hu=ds.pixel_array.astype(np.float32)*float(getattr(ds,'RescaleSlope',1.))+float(getattr(ds,'RescaleIntercept',0.))
        x=window_hu(hu)
        if hasattr(ds,'PixelSpacing'):
            spacing=[float(z) for z in ds.PixelSpacing];source='DICOM PixelSpacing row/col mm'
    else:
        with Image.open(path) as im:
            im=ImageOps.exif_transpose(im).convert('L');a=np.asarray(im)
        x=raster_normalize(a)
    if spacing is not None:
        if not source:raise ValueError('A manually supplied spacing requires documented provenance')
        if len(spacing)!=2 or not np.isfinite(spacing).all() or min(spacing)<=0:raise ValueError('Invalid pixel spacing')
    return x,spacing,source

@torch.inference_mode()
def predict_classification(model,x,tile=256,device='cuda'):
    a=np.uint8(np.clip(x,0,1)*255)
    with Image.fromarray(a) as im:
        arr=np.asarray(im.resize((tile,tile),Image.Resampling.BILINEAR),np.float32)/255.
    inp=torch.from_numpy(arr.copy())[None,None].to(device)
    with torch.autocast('cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
        probs=model(inp,task='cls').float().softmax(1).flatten().cpu().numpy()
    return {name:float(probs[i]) for i,name in enumerate(DISEASES)}

def infer_2d_multitask(model,path,outdir,tile=256,device='cuda',spacing=None,spacing_source=None,thresholds=None):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    x,spacing,spacing_source=read_2d(path,spacing,spacing_source)
    cls=predict_classification(model,x,tile,device)
    p=predict_slice(model,x,tile,device=device,tta=True)
    threshold=np.asarray(getattr(model,'segmentation_thresholds',[.5,.5,.5]) if thresholds is None else thresholds,dtype=np.float32)
    if threshold.shape!=(3,) or np.any((threshold<=0)|(threshold>=1)):raise ValueError('Need 3 probability thresholds')
    masks=(p>=threshold[:,None,None])
    rows=components_2d(masks,spacing=spacing,provenance=spacing_source or 'mm unavailable: missing verified pixel spacing')
    probs={name:float(value) for name,value in cls.items()}
    winner=max(cls,key=cls.get)
    overlap_pixels=int((masks.sum(0)>1).sum())
    warnings=[]
    if overlap_pixels:warnings.append('Disease masks overlap on some pixels; not a jointly exclusive multiclass segmentation.')
    if winner=='normal' and masks.any():warnings.append('Classifier predicts Normal but segmentation detects candidate lesions.')
    if winner!='normal' and not masks.any():warnings.append('Classifier suggests disease but no segmentation region passes threshold.')
    if spacing is None:warnings.append('Physical size in mm cannot be calculated from this image alone.')
    np.savez_compressed(outdir/'prediction_2d.npz',masks=masks,probabilities=p)
    pd.DataFrame(rows).to_csv(outdir/'lesions_2d.csv',index=False)
    overlay(x,masks,outdir/'overlay_2d.png',title='Four-class + three-mask research prediction')
    save_json({'input_file':str(path),'predicted_single_class':winner,'class_probabilities':probs,
               'segmentation_classes':list(MASK_CLASSES),'segmentation_thresholds':list(map(float,threshold)),
               'lesions':rows,'flags':warnings,
               'note':'Research only; four-way class forced, but coexisting conditions are possible. Probability is not calibrated confidence.'},outdir/'prediction_2d.json')
    return {'prediction':winner,'class_probabilities':probs,'lesions':rows,'warnings':warnings,'overlay':str(outdir/'overlay_2d.png')}

def infer_volume_multitask(model,path,outdir,tile=256,device='cuda',unit_override=None):
    """Produces native masks and measurements; volume-level classification is NOT validated."""
    import nibabel as nib
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    rows=infer_nifti(model,str(path),outdir,tile=tile,device=device,unit_override=unit_override,threshold=.5,tta=False)
    image=nib.as_closest_canonical(nib.load(str(path)))
    arr=np.asarray(image.dataobj,dtype=np.float32)
    # Sampled slice labels are for exploration, NOT a reliable single patient-level class.
    indices=np.linspace(0,arr.shape[2]-1,min(24,arr.shape[2])).round().astype(int)
    per_slice=[]
    for z in sorted(set(indices)):
        cls=predict_classification(model,window_hu(arr[:,:,z].T),tile,device)
        per_slice.append({'slice_index':int(z),**cls})
    pd.DataFrame(per_slice).to_csv(outdir/'sampled_slice_class_probabilities.csv',index=False)
    save_json({'segmentation_lesions':rows,'class_outputs':per_slice,
               'warning':'Sampled slice predictions are not calibrated 3D scan-level classification'},outdir/'multitask_volume_summary.json')
    return rows
