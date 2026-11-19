"""Native-grid tiled inference and geometry-aware measurements. Research outputs only."""
import json,math,itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull,distance
from kv_model import CLASSES,SwinSegmenter,UNet
from kv_data import read_gray,raster_normalize,window_hu,valid_affine,save_json

@torch.inference_mode()
def predict_slice(model,image,tile=256,overlap=.25,device='cuda',tta=False,infer_batch=8):
    """Native-grid tiled inference with batched GPU tiles; numerically equivalent reconstruction."""
    model.eval();x=np.asarray(image,dtype=np.float32)
    assert x.ndim==2 and np.isfinite(x).all() and 0<=overlap<1
    h,w=x.shape;hp=max(h,tile);wp=max(w,tile);x=np.pad(x,((0,hp-h),(0,wp-w)))
    step=max(1,int(tile*(1-overlap)))
    def starts(n):return sorted(set(list(range(0,n-tile+1,step))+[n-tile]))
    coords=[(yy,xx) for yy in starts(hp) for xx in starts(wp)]
    accum=np.zeros((3,hp,wp),np.float32);denom=np.zeros((hp,wp),np.float32)
    ramp=np.hanning(tile).astype(np.float32);weight=np.maximum(np.outer(ramp,ramp),.01)
    infer_batch=max(1,int(infer_batch))
    for base in range(0,len(coords),infer_batch):
        chunk=coords[base:base+infer_batch]
        patches=np.stack([x[yy:yy+tile,xx:xx+tile] for yy,xx in chunk])
        batch=torch.from_numpy(np.ascontiguousarray(patches))[:,None].to(device,non_blocking=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
            p=model(batch).float().sigmoid()
            if tta:p=(p+model(batch.flip(-1)).float().sigmoid().flip(-1))*.5
        probs=p.cpu().numpy()
        for j,(yy,xx) in enumerate(chunk):
            accum[:,yy:yy+tile,xx:xx+tile]+=probs[j]*weight
            denom[yy:yy+tile,xx:xx+tile]+=weight
    assert denom.min()>0
    out=accum/denom
    if not np.isfinite(out).all():raise FloatingPointError('Nonfinite segmentation probabilities')
    return out[:,:h,:w]

def feret(mask,spacing=(1.,1.)):
    """Maximum distance between convex-hull vertices of occupied pixel-cell corners."""
    mask=np.asarray(mask,bool)
    if not mask.any():return 0.
    border=mask & ~ndi.binary_erosion(mask)
    yx=np.argwhere(border).astype(float)
    corners=np.concatenate([yx+off for off in ((-.5,-.5),(-.5,.5),(.5,-.5),(.5,.5))])
    pts=np.unique(corners*np.asarray(spacing)[None,:],axis=0)
    hull=pts[ConvexHull(pts).vertices]
    return float(distance.pdist(hull).max())

def measure_2d(mask,spacing=None,provenance='mm_unavailable: no physical calibration'):
    mask=np.asarray(mask,bool);coords=np.argwhere(mask)
    if not len(coords):return None
    lo=coords.min(0);hi=coords.max(0)+1;center=coords.mean(0)
    row={'area_px2':int(mask.sum()),'bbox_x_px':int(lo[1]),'bbox_y_px':int(lo[0]),'bbox_width_px':int(hi[1]-lo[1]),'bbox_height_px':int(hi[0]-lo[0]),'centroid_x_px':float(center[1]),'centroid_y_px':float(center[0]),'maximum_in_plane_diameter_px':feret(mask),'diameter_method':'maximum Feret diameter of pixel-cell boundary convex hull','approximate':True,'spacing_source':provenance,'size_status':'mm_unavailable'}
    for k in ('bbox_width_mm','bbox_height_mm','area_mm2','equivalent_diameter_mm','maximum_in_plane_diameter_mm'):row[k]=None
    if spacing is not None:
        ry,cx=map(float,spacing)
        if not np.isfinite([ry,cx]).all() or min(ry,cx)<=0:raise ValueError('Invalid pixel spacing')
        area=float(mask.sum()*ry*cx)
        row.update(size_status='approximate_mm',bbox_width_mm=float((hi[1]-lo[1])*cx),bbox_height_mm=float((hi[0]-lo[0])*ry),area_mm2=area,equivalent_diameter_mm=2*math.sqrt(area/math.pi),maximum_in_plane_diameter_mm=feret(mask,(ry,cx)))
    return row

def components_2d(masks,spacing=None,provenance='mm_unavailable'):
    rows=[]
    for c,name in enumerate(CLASSES):
        lab,n=ndi.label(masks[c],np.ones((3,3)))
        for i,sl in enumerate(ndi.find_objects(lab),1):
            if sl is None:continue
            component=lab[sl]==i;row=measure_2d(component,spacing,provenance)
            row['bbox_x_px']+=sl[1].start;row['bbox_y_px']+=sl[0].start
            row['centroid_x_px']+=sl[1].start;row['centroid_y_px']+=sl[0].start
            row.update(lesion_class=name,component=i);rows.append(row)
    return rows

def components_3d(masks,affine_mm=None,provenance='mm_unavailable'):
    """masks [3,X,Y,Z] on the canonical scan grid. Components use 26-connectivity."""
    rows=[];a=None if affine_mm is None else np.asarray(affine_mm,float)
    spacing=None if a is None else (np.linalg.norm(a[:3,1]),np.linalg.norm(a[:3,0]))
    for c,name in enumerate(CLASSES):
        lab,n=ndi.label(masks[c],np.ones((3,3,3)))
        for i,sl in enumerate(ndi.find_objects(lab),1):
            if sl is None:continue
            comp=lab[sl]==i;coords=np.argwhere(comp);offset=np.array([s.start for s in sl]);global_coords=coords+offset
            row={'lesion_class':name,'component':i,'voxel_count':int(comp.sum()),'centroid_voxel_xyz':global_coords.mean(0).tolist(),'bbox_start_xyz':offset.tolist(),'bbox_stop_xyz':[s.stop for s in sl],'approximate':True,'spacing_source':provenance,'size_status':'mm_unavailable','volume_mm3':None,'volume_cm3':None,'extent_RAS_mm':None,'centroid_RAS_mm':None,'maximum_in_plane_diameter_mm':None,'touches_volume_boundary':any(s.start==0 or s.stop==masks.shape[j+1] for j,s in enumerate(sl))}
            if a is not None:
                physical=global_coords@a[:3,:3].T+a[:3,3]
                edge=np.abs(a[:3,:3]).sum(1)
                row.update(size_status='approximate_mm',volume_mm3=float(comp.sum()*abs(np.linalg.det(a[:3,:3]))),extent_RAS_mm=(physical.max(0)-physical.min(0)+edge).tolist(),centroid_RAS_mm=physical.mean(0).tolist(),maximum_in_plane_diameter_mm=max(feret(comp[:,:,z].T,spacing) for z in range(comp.shape[2])))
                row['volume_cm3']=row['volume_mm3']/1000.
            rows.append(row)
    return rows

def load_segmenter(path,device='cuda'):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck.get('classes')!=CLASSES:raise ValueError('Segmentation class order mismatch')
    if ck['kind']=='swin':m=SwinSegmenter(ck['encoder_config'])
    elif ck['kind']=='unet':m=UNet()
    else:raise ValueError('Unknown model kind')
    m.load_state_dict(ck['model'],strict=True);return m.to(device).eval(),ck

def overlay(image,masks,path,title='Predicted masks — research only'):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors=np.array([[1.,.3,.15],[.1,.8,.25],[.2,.5,1.]])
    fig,ax=plt.subplots(figsize=(7,7));ax.imshow(image,cmap='gray',vmin=0,vmax=1)
    for c,name in enumerate(CLASSES):
        rgba=np.zeros((*image.shape,4));rgba[...,:3]=colors[c];rgba[...,3]=masks[c]*.45;ax.imshow(rgba)
    ax.legend(handles=[Patch(color=co,label=name) for co,name in zip(colors,CLASSES)],loc='upper right');ax.set_title(title);ax.axis('off');fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def infer_2d(model,path,outdir,tile=256,device='cuda',spacing=None,spacing_provenance=None,threshold=.5,tta=True):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True);path=Path(path)
    if path.suffix.lower()=='.dcm':
        import pydicom
        ds=pydicom.dcmread(path)
        if getattr(ds,'Modality','')!='CT' or int(getattr(ds,'NumberOfFrames',1))!=1:raise ValueError('Only single-frame CT DICOM supported here')
        if getattr(ds,'PhotometricInterpretation','MONOCHROME2')!='MONOCHROME2':raise ValueError('Unsupported CT photometric interpretation')
        x=window_hu(ds.pixel_array.astype(np.float32)*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0)))
        if hasattr(ds,'PixelSpacing'):spacing=tuple(map(float,ds.PixelSpacing));spacing_provenance='DICOM PixelSpacing (0028,0030), row then column, native image'
    else:x=raster_normalize(read_gray(path))
    if spacing is not None and not spacing_provenance:raise ValueError('Manual spacing requires documented source; never use a guessed value')
    if spacing is not None and (len(spacing)!=2 or not np.isfinite(spacing).all() or min(spacing)<=0):
        raise ValueError('Pixel spacing must contain two positive finite numbers')
    p=predict_slice(model,x,tile,device=device,tta=tta);m=p>=threshold
    entropy=-(p*np.log(p.clip(1e-7,1))+(1-p)*np.log((1-p).clip(1e-7,1)))
    np.savez_compressed(outdir/'prediction_2d.npz',probabilities=p,masks=m,entropy=entropy)
    rows=components_2d(m,spacing,spacing_provenance or 'mm_unavailable: raster has no verified physical scale')
    pd.DataFrame(rows).to_csv(outdir/'lesions_2d.csv',index=False)
    save_json({'source_name':path.name,'classes':CLASSES,'threshold':threshold,'lesions':rows,'status':'research_only; absence of predicted pixels does not rule out disease','uncertainty':'sigmoid entropy is uncalibrated; not clinical confidence','physical_measurement_status':'approximate_mm' if spacing else 'mm_unavailable','cross_class_overlap_pixels':int((m.sum(0)>1).sum())},outdir/'prediction_2d.json')
    overlay(x,m,outdir/'overlay_2d.png');return rows

def infer_nifti(model,path,outdir,tile=256,device='cuda',unit_override=None,threshold=.5,tta=False,complete_volume=False):
    """Full provided volume is processed. Whole-lesion completeness still requires provenance."""
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    original=nib.load(path)
    if len(original.shape)!=3:raise ValueError('Need one 3D CT volume')
    valid_affine(original,unit_override)
    canonical=nib.as_closest_canonical(original);af,source=valid_affine(canonical,unit_override)
    directions=canonical.affine[:3,:3]/np.linalg.norm(canonical.affine[:3,:3],axis=0)
    if np.min(np.diag(directions))<.995:raise ValueError('Oblique input: resample to axial using validated imaging software')
    masks=np.zeros((3,*canonical.shape),dtype=np.uint8);x=np.asarray(canonical.dataobj,dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Nonfinite CT')
    for z in range(x.shape[2]):
        p=predict_slice(model,window_hu(x[:,:,z].T),tile,device=device,tta=tta)
        masks[:,:,:,z]=(p>=threshold).transpose(0,2,1)
        if z%25==0:print(f'Inference slice {z+1}/{x.shape[2]}',flush=True)
    back=nib.orientations.ornt_transform(nib.orientations.io_orientation(canonical.affine),nib.orientations.io_orientation(original.affine))
    for c,name in enumerate(CLASSES):
        native=nib.orientations.apply_orientation(masks[c],back)
        assert native.shape==original.shape
        header=original.header.copy();header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(native,original.affine,header),outdir/f'{name}_mask_native.nii.gz')
    rows=components_3d(masks,af,source)
    for row in rows:
        row['volume_scope']='complete_scan_asserted_by_user' if complete_volume else 'provided_volume_only_completeness_unverified'
        row['whole_lesion_size_verified']=False
    pd.DataFrame(rows).to_csv(outdir/'lesions_3d.csv',index=False)
    save_json({'classes':CLASSES,'lesions':rows,'source_name':Path(path).name,'all_provided_slices_processed':True,'complete_scan_asserted':complete_volume,'measurement_scope':'segmented voxels in supplied CT; gaps, coverage and boundary quality need review','cross_class_overlap_voxels':int((masks.sum(0)>1).sum()),'spacing_source':source},outdir/'prediction_3d.json')
    mid=x.shape[2]//2;overlay(window_hu(x[:,:,mid].T),masks[:,:,:,mid].transpose(0,2,1),outdir/'middle_slice_overlay.png')
    return rows

def dicom_series_to_nifti(folder,destination):
    """Strict single-series, regular-grid CT conversion. Reject gaps, duplicates and tilt."""
    import pydicom
    records=[]
    for p in Path(folder).rglob('*'):
        if not p.is_file():continue
        try:ds=pydicom.dcmread(p,stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:continue
        if getattr(ds,'Modality','')=='CT':records.append((p,ds))
    if len(records)<2:raise ValueError('Need >=2 CT instances')
    series={str(ds.SeriesInstanceUID) for _,ds in records}
    if len(series)!=1:raise ValueError('Multiple CT series: choose one series folder')
    first=records[0][1];iop=np.array(first.ImageOrientationPatient,float);col=iop[:3];row=iop[3:];normal=np.cross(col,row)
    if not np.allclose([np.linalg.norm(col),np.linalg.norm(row),np.dot(col,row)],[1,1,0],atol=1e-4):raise ValueError('Invalid orientation')
    spacing=np.array(first.PixelSpacing,float)
    if not np.isfinite(spacing).all() or np.min(spacing)<=0:raise ValueError('Invalid spacing')
    for _,ds in records:
        if 'LOCALIZER' in getattr(ds,'ImageType',[]):raise ValueError('Localizer/scout images are not a volumetric CT series')
        if int(getattr(ds,'NumberOfFrames',1))!=1:raise ValueError('Enhanced multiframe CT requires validated conversion software')
        if ds.Rows!=first.Rows or ds.Columns!=first.Columns or not np.allclose(ds.ImageOrientationPatient,iop,atol=1e-4) or not np.allclose(ds.PixelSpacing,spacing,atol=1e-4):raise ValueError('Inconsistent CT geometry')
    records.sort(key=lambda rec:float(np.dot(np.array(rec[1].ImagePositionPatient,float),normal)))
    positions=np.array([ds.ImagePositionPatient for _,ds in records],float);distances=positions@normal;delta=np.diff(distances);dz=float(np.median(delta))
    if dz<=0 or not np.allclose(delta,dz,rtol=.01,atol=.01):raise ValueError('Duplicate, missing, or irregularly spaced slices')
    if not np.allclose(np.diff(positions,axis=0),normal*dz,atol=.05):raise ValueError('Gantry tilt / in-plane shifts are not supported')
    vol=[]
    for p,_ in records:
        ds=pydicom.dcmread(p)
        if getattr(ds,'PhotometricInterpretation','MONOCHROME2')!='MONOCHROME2':raise ValueError('Unsupported photometric interpretation')
        vol.append(ds.pixel_array.astype(np.float32)*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0)))
    # NIfTI axes are row, column, slice here; canonicalization handles their anatomical orientation.
    a=np.eye(4);a[:3,0]=row*spacing[0];a[:3,1]=col*spacing[1];a[:3,2]=normal*dz;a[:3,3]=positions[0]
    a=np.diag([-1.,-1.,1.,1.])@a # DICOM LPS -> NIfTI RAS
    im=nib.Nifti1Image(np.stack(vol,axis=2),a);im.header.set_xyzt_units('mm');nib.save(im,destination)
    return str(destination)