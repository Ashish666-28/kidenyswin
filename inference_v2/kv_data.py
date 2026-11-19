"""Acquisition, validation, partial-label preparation, and leakage-aware splits."""
import json,hashlib,os,time,shutil,zipfile,random,re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import nibabel as nib
from PIL import Image
import torch
from torch.utils.data import Dataset

KITS_REV='c1088353084c17b8882a11db71429e7c022b7785'
KSSD='murillobouzon/kssd2025-kidney-stone-segmentation-dataset'
SAM_STONE='bemorekgg/kidney-stone-segmentation-dataset'
PIPELINE_VERSION='kvseg-2.0-allcase-stream'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save_json(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));os.replace(tmp,path)

def download(url,path,expected_sha=None,expected_bytes=None,git_blob=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);receipt=path.with_name(path.name+'.receipt.json')
    if path.exists() and receipt.exists():
        rec=json.loads(receipt.read_text())
        if rec['url']==url and sha256(path)==rec['sha256'] and (not expected_sha or rec['sha256']==expected_sha):return path
    last=None
    for attempt in range(4):
        tmp=path.with_name(path.name+'.part')
        try:
            with requests.get(url,stream=True,timeout=(20,120)) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for b in r.iter_content(1024*1024):
                        if b:f.write(b)
            if expected_bytes and tmp.stat().st_size!=expected_bytes:raise ValueError('Download size mismatch')
            checksum=sha256(tmp)
            if expected_sha and checksum!=expected_sha:raise ValueError('Download SHA256 mismatch')
            if git_blob:
                raw=tmp.read_bytes();digest=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
                if digest!=git_blob:raise ValueError('Official Git blob hash mismatch')
            os.replace(tmp,path);save_json({'url':url,'sha256':checksum,'bytes':path.stat().st_size},receipt);return path
        except (requests.RequestException,OSError,ValueError) as exc:
            last=exc;tmp.unlink(missing_ok=True)
            print(f'Download attempt {attempt+1}/4 failed: {type(exc).__name__}; retrying',flush=True)
            if attempt<3:time.sleep(2**attempt)
    raise RuntimeError(f'Could not download {path.name}. Check internet, quota or dataset access, then rerun. {last}')

def fetch_json(url,path):
    download(url,path);return json.loads(Path(path).read_text())

def acquire(cfg,root):
    root=Path(root);raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
    stone=raw/'kssd';archive=raw/'kssd_v1.zip'
    if cfg.get('KSSD_LOCAL_DIR'):
        stone=Path(cfg['KSSD_LOCAL_DIR']);assert stone.is_dir()
    else:
        # Public v1 archive. Kaggle authentication may be required if access rules change.
        download('https://www.kaggle.com/api/v1/datasets/download/'+KSSD+'?datasetVersionNumber=1',archive)
        stamp=stone/'complete.json'
        if not stamp.exists() or json.loads(stamp.read_text()).get('archive_sha256')!=sha256(archive):
            stone.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                bad=z.testzip()
                if bad:raise ValueError(f'Corrupted KSSD archive member: {bad}')
                for item in z.infolist():
                    p=(stone/item.filename).resolve()
                    if not p.is_relative_to(stone.resolve()):raise ValueError('Unsafe archive member')
                z.extractall(stone)
            save_json({'archive_sha256':sha256(archive)},stamp)
    # Query the pinned tree: no fragile assumption about contiguous case numbers.
    tree=fetch_json(f'https://api.github.com/repos/neheller/kits23/git/trees/{KITS_REV}?recursive=1',raw/'kits_tree.json')
    assert not tree.get('truncated'), 'Truncated repository tree; cannot guarantee complete case discovery.'
    masks={x['path'].split('/')[1]:x for x in tree['tree'] if re.fullmatch(r'dataset/case_\d+/segmentation.nii.gz',x['path'])}
    candidates=sorted(masks);rng=random.Random(cfg['SEED']);rng.shuffle(candidates)
    n=cfg['KITS_CASES'];selected=candidates if n==0 else candidates[:n]
    assert len(selected)>=12, 'Use at least 12 KiTS cases for three nonempty splits.'
    hf=fetch_json('https://huggingface.co/api/datasets/neheller/KiTS-Challenge-Imaging',raw/'hf_info.json')
    rev=hf['sha'];chosen=raw/'selected_cases.json'
    selection={'cases':selected,'kits_revision':KITS_REV,'hf_revision':rev,'seed':cfg['SEED']}
    if chosen.exists() and json.loads(chosen.read_text())!=selection:raise RuntimeError('Dataset settings changed. Use a new PROJECT_NAME to preserve the original split/run.')
    save_json(selection,chosen)
    # FULL-COHORT STREAMING: discover all 489 annotated KiTS training cases,
    # but defer downloads until each case is processed. This keeps raw CTs from
    # exhausting Drive and allows progress to resume across Colab runtimes.
    cases=[]
    for case in selected:
        m=masks[case]
        dest=raw/'kits'/case
        cases.append({'case_id':case,'image':str(dest/'imaging.nii.gz'),
                      'mask':str(dest/'segmentation.nii.gz'),
                      'image_url':f'https://huggingface.co/datasets/neheller/KiTS-Challenge-Imaging/resolve/{rev}/images/{case}.nii.gz',
                      'mask_url':f'https://raw.githubusercontent.com/neheller/kits23/{KITS_REV}/{m["path"]}',
                      'mask_bytes':m['size'],'mask_blob':m['sha']})
    print('Discovered',len(cases),'annotated KiTS23 cases (all public training cases if KITS_CASES=0)')
    return stone,cases,selection

def acquire_sam(cfg,root):
    """Acquire optional generated SAM masks. They are weak training labels only."""
    if not cfg.get('USE_SAM_STONE',False):return None
    root=Path(root);raw=root/'raw';raw.mkdir(parents=True,exist_ok=True)
    if cfg.get('SAM_LOCAL_DIR'):
        folder=Path(cfg['SAM_LOCAL_DIR'])
        if not folder.is_dir():raise FileNotFoundError(folder)
        return folder
    archive=raw/'sam_stone_masks.zip';folder=raw/'sam_stone'
    download('https://www.kaggle.com/api/v1/datasets/download/'+SAM_STONE,archive)
    if not zipfile.is_zipfile(archive):
        raise RuntimeError('SAM dataset response is not a ZIP archive; accept the Kaggle terms/authenticate or set SAM_LOCAL_DIR')
    stamp=folder/'complete.json';digest=sha256(archive)
    if not stamp.exists() or json.loads(stamp.read_text()).get('archive_sha256')!=digest:
        folder.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            bad=z.testzip()
            if bad:raise ValueError('Corrupted SAM archive member: '+bad)
            for item in z.infolist():
                target=(folder/item.filename).resolve()
                if not target.is_relative_to(folder.resolve()):raise ValueError('Unsafe SAM archive member')
            z.extractall(folder)
        save_json({'archive_sha256':digest},stamp)
    return folder

def discover_sam_pairs(root):
    """Pair common Kaggle layouts without assuming one filename convention."""
    root=Path(root);ext={'.jpg','.jpeg','.png','.tif','.tiff','.bmp'}
    files=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ext]
    mask_dirs={'mask','masks','label','labels','annotation','annotations','ground_truth','groundtruth'}
    def is_mask(p):
        parents={x.lower() for x in p.relative_to(root).parts[:-1]}
        stem=p.stem.lower()
        return bool(parents&mask_dirs) or bool(re.search(r'(^mask[_-]|[_-](mask|label|seg|segmentation)$)',stem))
    def key(p):
        rel=p.relative_to(root);parts=[x for x in rel.parts[:-1] if x.lower() not in mask_dirs|{'image','images','img','imgs'}]
        stem=re.sub(r'(^mask[_-]|[_-](mask|label|seg|segmentation)$)','',p.stem,flags=re.I)
        return '/'.join(parts+[stem]).lower()
    def index(paths):
        out={}
        for p in paths:out.setdefault(key(p),[]).append(p)
        return out
    images=index([p for p in files if not is_mask(p)]);masks=index([p for p in files if is_mask(p)])
    common=sorted(set(images)&set(masks));pairs=[];unpaired_images=[];unpaired_masks=[]
    for k in common:
        if len(images[k])==1 and len(masks[k])==1:pairs.append((k,images[k][0],masks[k][0]))
        else:
            unpaired_images.extend(str(p.relative_to(root)) for p in images[k])
            unpaired_masks.extend(str(p.relative_to(root)) for p in masks[k])
    for k in sorted(set(images)-set(masks)):unpaired_images.extend(str(p.relative_to(root)) for p in images[k])
    for k in sorted(set(masks)-set(images)):unpaired_masks.extend(str(p.relative_to(root)) for p in masks[k])
    return pairs,unpaired_images,unpaired_masks

def read_gray(path):
    with Image.open(path) as im:
        if getattr(im,'n_frames',1)!=1:raise ValueError(f'Multiframe raster not supported: {path}')
        a=np.array(im)
    if a.ndim==3:
        if a.shape[-1] not in (3,4) or not np.all(a[...,:3]==a[...,:1]):raise ValueError(f'Color raster requires explicit conversion: {path}')
        a=a[...,0]
    if a.ndim!=2 or not np.isfinite(a).all():raise ValueError(f'Invalid 2D raster: {path}')
    return a

def stone_mask(a):
    if a.min()<0 or a.max()>255:raise ValueError('Expected KSSD 0/255 binary-label encoding')
    if a.max()<=1:return a.astype(bool)
    # Actual KSSD TIF labels have near-black/near-white compression noise (0..5, 251..255).
    if np.any((a>32)&(a<223)):raise ValueError('Mask is not bimodal near 0/255; inspect polarity/encoding')
    out=a>127
    if out.mean()>.15:raise ValueError('Suspicious stone foreground area; do not silently invert labels')
    return out

def raster_normalize(a):
    if a.dtype==np.uint8:return a.astype(np.float32)/255.
    if np.issubdtype(a.dtype,np.integer):return a.astype(np.float32)/np.iinfo(a.dtype).max
    a=a.astype(np.float32)
    if a.min()<0 or a.max()>1:raise ValueError('Float raster needs explicit intensity normalization')
    return a

def window_hu(a):return np.clip((np.asarray(a,dtype=np.float32)+200.)/600.,0,1)

def valid_affine(img,unit_override=None):
    if int(img.header['sform_code'])==0 and int(img.header['qform_code'])==0:
        raise ValueError('No coded spatial transform; orientation/physical geometry cannot be verified')
    q,qc=img.get_qform(coded=True);s,sc=img.get_sform(coded=True)
    if qc and sc and not np.allclose(q,s,atol=1e-3):
        raise ValueError('Conflicting NIfTI qform and sform; resolve geometry before measuring')
    a=np.array(img.affine,dtype=float)
    if not np.isfinite(a).all() or abs(np.linalg.det(a[:3,:3]))<1e-10:raise ValueError('Invalid affine')
    sp=np.linalg.norm(a[:3,:3],axis=0);basis=a[:3,:3]/sp
    if not np.allclose(basis.T@basis,np.eye(3),atol=1e-4):raise ValueError('Sheared geometry: resample with validated imaging software first')
    unit=img.header.get_xyzt_units()[0];factor={'mm':1.,'meter':1000.,'micron':.001}.get(unit)
    source='NIfTI header units and affine'
    if factor is None and unit_override:
        assert unit_override['unit']=='mm' and unit_override['provenance']
        factor=1.;source=unit_override['provenance']+'; NIfTI unit field='+unit
    if factor is None:return None,'mm_unavailable: NIfTI spatial unit missing'
    a[:3,:]*=factor
    return a,source

def canonical_pair(image_path,mask_path,unit_override=None):
    im=nib.load(image_path);ma=nib.load(mask_path)
    if len(im.shape)!=3 or im.shape!=ma.shape or not np.allclose(im.affine,ma.affine,atol=1e-4):raise ValueError('Image/mask grids do not align')
    if int(im.header['sform_code'])==0 and int(im.header['qform_code'])==0:raise ValueError('No coded spatial transform')
    orig_shape=im.shape;orig_aff=im.affine.copy()
    im=nib.as_closest_canonical(im);ma=nib.as_closest_canonical(ma)
    af,source=valid_affine(im,unit_override)
    # Require near-axial slices; closest-canonical permutes/flips, it does not de-oblique.
    directions=im.affine[:3,:3]/np.linalg.norm(im.affine[:3,:3],axis=0)
    if np.min(np.diag(directions))<.995:raise ValueError('Oblique CT requires validated axial resampling')
    x=np.asarray(im.dataobj,dtype=np.float32);y=np.asanyarray(ma.dataobj)
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite volume values')
    if not np.isin(y,[0,1,2,3]).all():raise ValueError('KiTS labels must be 0=background,1=kidney,2=tumor,3=cyst')
    meta={'canonical_affine':im.affine.tolist(),'affine_mm':None if af is None else af.tolist(),'spacing_source':source,'original_affine':orig_aff.tolist(),'original_shape':list(orig_shape),'canonical_shape':list(im.shape),'original_units':im.header.get_xyzt_units()[0]}
    return x,y.astype(np.uint8),meta

def discover_stones(root):
    root=Path(root);ext={'.png','.jpg','.jpeg','.tif','.tiff','.bmp'}
    images=[p for p in root.rglob('*') if p.suffix.lower() in ext and any(a.lower() in ('image','images') for a in p.relative_to(root).parts[:-1])]
    masks=[p for p in root.rglob('*') if p.suffix.lower() in ext and any(a.lower() in ('mask','masks','label','labels') for a in p.relative_to(root).parts[:-1])]
    def index(ps):
        out={}
        for p in ps:
            if p.stem in out:raise ValueError(f'Ambiguous duplicate basename: {p.stem}')
            out[p.stem]=p
        return out
    a,b=index(images),index(masks)
    if not a or set(a)!=set(b):raise ValueError(f'Image/mask pairing failed: images={len(a)}, masks={len(b)}, unmatched={sorted(set(a)^set(b))[:10]}')
    return [(key,a[key],b[key]) for key in sorted(a)]

def split_groups(ids,seed,positive=None):
    ids=sorted(set(ids));n=len(ids);nval=max(1,round(n*.15));ntest=max(1,round(n*.15))
    if n-nval-ntest<2:raise ValueError('Too few independent groups for split')
    for offset in range(500):
        order=ids.copy();random.Random(seed+offset).shuffle(order)
        out={x:('test' if i<ntest else 'val' if i<ntest+nval else 'train') for i,x in enumerate(order)}
        if positive is None or all(any(positive[g][c] for g in ids if out[g]==split) for c in (0,1) for split in ('train','val','test')):return out
    raise ValueError('Cannot form splits with tumor and cyst positives in each. Increase KITS_CASES using a new project.')

class UnionFind:
    def __init__(self,n):self.p=list(range(n))
    def find(self,a):
        while self.p[a]!=a:self.p[a]=self.p[self.p[a]];a=self.p[a]
        return a
    def union(self,a,b):self.p[self.find(b)]=self.find(a)

def prepare(cfg,root,local,stone_root,cases,selection):
    root,local=Path(root),Path(local);proc=root/'processed';proc.mkdir(exist_ok=True)
    policy={k:cfg[k] for k in ('SEED','KITS_CASES','KITS_UNIT_OVERRIDE','KSSD_PATIENT_CSV','KITS_KEEP_RAW','CACHE_POLICY')}
    policy.update(version=PIPELINE_VERSION,selection=selection)
    policy['patient_csv_sha']=sha256(cfg['KSSD_PATIENT_CSV']) if cfg['KSSD_PATIENT_CSV'] else None
    policy_path=proc/'policy.json';manifest_path=proc/'manifest.csv'
    if policy_path.exists() and json.loads(policy_path.read_text())!=policy:raise ValueError('Preparation policy changed: use a new PROJECT_NAME')
    if manifest_path.exists():
        if not policy_path.exists():raise ValueError('Manifest lacks policy; use a new project')
        frame=pd.read_csv(manifest_path);print('Reusing completed preparation',len(frame))
    else:
        save_json(policy,policy_path);rows=[];stone_records=[];tiny=[]
        pairs=discover_stones(stone_root)
        patient_map={}
        if cfg['KSSD_PATIENT_CSV']:
            mapping=pd.read_csv(cfg['KSSD_PATIENT_CSV'],dtype=str)
            assert {'image_id','patient_id'}.issubset(mapping.columns) and not mapping['image_id'].duplicated().any()
            assert not mapping[['image_id','patient_id']].isna().any().any()
            patient_map=dict(zip(mapping.image_id,mapping.patient_id))
            assert set(k for k,_,_ in pairs)<=set(patient_map), 'Patient map must cover all images'
        for key,im,ma in pairs:
            x=read_gray(im);y=stone_mask(read_gray(ma))
            if x.shape!=y.shape:raise ValueError(f'Misaligned stone pair {key}')
            norm=raster_normalize(x);digest=hashlib.sha256(np.ascontiguousarray(x).tobytes()+str(x.shape).encode()).hexdigest()
            thumb=np.asarray(Image.fromarray((norm*255).astype('uint8')).resize((32,32)),dtype=np.float32)
            tiny.append(thumb/255.);record={'id':key,'digest':digest,'image':im,'mask':ma,'mask_digest':hashlib.sha256(y.tobytes()).hexdigest()};stone_records.append(record)
        uf=UnionFind(len(pairs));byhash={};bypatient={};duplicate_links=[]
        for i,r in enumerate(stone_records):
            if r['digest'] in byhash:
                j=byhash[r['digest']]
                if r['mask_digest']!=stone_records[j]['mask_digest']:raise ValueError('Identical images have conflicting masks: inspect annotation conflict')
                uf.union(i,j);duplicate_links.append((r['id'],stone_records[j]['id'],'exact'))
            byhash[r['digest']]=i
            if patient_map:
                pid=patient_map[r['id']]
                if pid in bypatient:uf.union(i,bypatient[pid])
                bypatient[pid]=i
        # Conservative thumbnail similarity grouping; not a substitute for patient IDs.
        ts=np.stack(tiny)
        for i in range(len(ts)):
            err=np.mean(np.abs(ts[i+1:]-ts[i]),axis=(1,2))
            for j in np.where(err<.008)[0]+i+1:
                uf.union(i,int(j));duplicate_links.append((stone_records[i]['id'],stone_records[j]['id'],'near_thumbnail'))
        gids=[f'kssd_group_{uf.find(i):05d}' for i in range(len(pairs))];sp=split_groups(gids,cfg['SEED'])
        pd.DataFrame(duplicate_links,columns=['image_a','image_b','reason']).to_csv(root/'reports'/'duplicate_links.csv',index=False)
        used=set()
        for i,r in enumerate(stone_records):
            if r['digest'] in used:continue
            used.add(r['digest']);x=raster_normalize(read_gray(r['image']));mask=stone_mask(read_gray(r['mask']))
            y=np.zeros((3,*x.shape),dtype=np.uint8);y[0]=mask
            rel=f'kssd/{r["id"]}.npz';p=proc/rel;p.parent.mkdir(exist_ok=True)
            np.savez_compressed(p,x=x.astype(np.float16),y=y)
            rows.append({'id':'kssd_'+r['id'],'source':'kssd','group':gids[i],'split':sp[gids[i]],'relpath':rel,'slice':0,'has_stone':int(y[0].any()),'has_tumor':0,'has_cyst':0,'known_stone':1,'known_tumor':0,'known_cyst':0,'patient_verified':bool(patient_map),'image_hash':r['digest']})
        # Process whole KiTS cases BEFORE slice indexing. All slices remain available for evaluation.
        case_frames=[];positive={}
        for num,c in enumerate(cases,1):
            case=c['case_id'];out=proc/'kits'/case
            completed=out/'case_complete.json'
            if completed.exists():
                item=json.loads(completed.read_text())
                cr=item['records'];positive[case]=item['positive']
                if not (out/'geometry.json').exists() or not all((proc/r['relpath']).exists() for r in cr):
                    raise RuntimeError(f'Incomplete processed KiTS case {case}: delete its processed case folder to retry')
                case_frames.extend(cr)
                print(f'KiTS {num}/{len(cases)} reused: {case}',flush=True)
                continue
            # A previous interrupted case cannot be trusted. Rebuild it atomically by stamp.
            if out.exists():shutil.rmtree(out)
            out.mkdir(parents=True,exist_ok=True)
            image=Path(c['image']);mask=Path(c['mask'])
            # For Drive-quota control, only ONE case's raw CT+mask need exist at a time.
            if shutil.disk_usage(root).free < 2*1024**3:
                raise RuntimeError('Less than 2 GiB Drive space remains. Expand Drive storage before proceeding.')
            download(c['mask_url'],mask,expected_bytes=c['mask_bytes'],git_blob=c['mask_blob'])
            download(c['image_url'],image)
            x,y,meta=canonical_pair(image,mask,cfg['KITS_UNIT_OVERRIDE'])
            positive[case]=[bool((y==2).any()),bool((y==3).any())]
            meta.update(image_path=str(image),mask_path=str(mask),case_id=case)
            save_json(meta,out/'geometry.json')
            cr=[]
            for z in range(x.shape[2]):
                xx=window_hu(x[:,:,z].T)
                yy=np.zeros((3,*xx.shape),dtype=np.uint8)
                yy[1]=(y[:,:,z].T==2);yy[2]=(y[:,:,z].T==3)
                rel=f'kits/{case}/{z:04d}.npz'
                temp=(proc/rel).with_suffix('.npz.part')
                with open(temp,'wb') as f:np.savez_compressed(f,x=xx.astype(np.float16),y=yy)
                os.replace(temp,proc/rel)
                cr.append({'id':f'{case}_{z:04d}','source':'kits','group':case,
                           'relpath':rel,'slice':z,'has_stone':0,
                           'has_tumor':int(yy[1].any()),'has_cyst':int(yy[2].any()),
                           'known_stone':0,'known_tumor':1,'known_cyst':1,
                           'patient_verified':True,'image_hash':''})
            save_json({'positive':positive[case],'records':cr},completed)
            case_frames.extend(cr)
            print(f'KiTS {num}/{len(cases)} prepared: {case}; {len(cr)} CT slices',flush=True)
            del x,y
            if not cfg.get('KITS_KEEP_RAW',False):
                for f in (image,mask):
                    f.unlink(missing_ok=True)
                    f.with_name(f.name+'.receipt.json').unlink(missing_ok=True)
        ks=split_groups(positive.keys(),cfg['SEED'],positive)
        for r in case_frames:r['split']=ks[r['group']]
        rows.extend(case_frames);frame=pd.DataFrame(rows)
        assert frame.groupby('group')['split'].nunique().max()==1
        for s in ('train','val','test'):
            for cls in ('stone','tumor','cyst'):
                if not frame.loc[frame.split==s,'has_'+cls].any():raise ValueError(f'No {cls} positive in {s}')
        frame.to_csv(manifest_path,index=False)
    # With 489 volumes, a second copy of every slice on /content is usually impossible.
    # Use persistent prepared Drive files in FULL mode; optional SSD copy for large runtimes.
    cache=local/'processed';cache.mkdir(parents=True,exist_ok=True)
    use_ssd=cfg.get('CACHE_POLICY')=='ssd'
    if use_ssd:
        bytes_needed=sum((proc/r.relpath).stat().st_size for r in frame.itertuples())
        free=shutil.disk_usage(local).free
        if bytes_needed>free*.75:
            raise RuntimeError(f'SSD cache needs {bytes_needed/2**30:.1f} GiB; only {free/2**30:.1f} GiB free. Set CACHE_POLICY=drive.')
        for r in frame.itertuples():
            src=proc/r.relpath;dst=cache/r.relpath;dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists() or src.stat().st_size!=dst.stat().st_size:shutil.copy2(src,dst)
        for p in proc.glob('kits/*/geometry.json'):
            dst=cache/p.relative_to(proc);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    else:
        cache=proc
    frame['path']=[str(cache/p) for p in frame.relpath]
    frame.to_csv(root/'reports'/'manifest.csv',index=False)
    save_json({'manifest_sha256':sha256(manifest_path),'groups_by_split':frame.groupby(['source','split']).group.nunique().to_dict().__str__(),'stone_patient_ids_verified':bool(cfg['KSSD_PATIENT_CSV']),'pretraining_overlap':'unknown; KSSD derives from CT Kidney classification dataset','joint_three_class_validation':False},root/'reports'/'data_audit.json')
    return frame,cache

def prepare_sam(cfg,root,sam_root,base_frame):
    """Add all safe SAM pairs as down-weighted TRAIN-ONLY stone supervision.

    Exact/near matches to KSSD are excluded because KSSD supplies held-out
    reference masks. Generated SAM masks never define validation/test scores.
    """
    if sam_root is None:return base_frame
    root=Path(root);proc=root/'processed';folder=proc/'sam_weak';folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'manifest_sam.csv';audit_path=root/'reports'/'sam_weak_audit.json'
    if manifest.exists():
        sam=pd.read_csv(manifest)
        missing=[str(proc/r.relpath) for r in sam.itertuples() if not (proc/r.relpath).exists()]
        if missing:raise RuntimeError('SAM manifest has missing prepared files; restore them or remove only processed/sam_weak to rebuild')
        print('Reusing prepared SAM weak-label images:',len(sam),flush=True)
    else:
        pairs,unpaired_images,unpaired_masks=discover_sam_pairs(sam_root)
        kssd=base_frame[base_frame.source=='kssd'];reference=[]
        for r in kssd.itertuples():
            with np.load(r.path,allow_pickle=False) as d:x=d['x'].astype(np.float32)
            reference.append(np.asarray(Image.fromarray(np.uint8(np.clip(x,0,1)*255)).resize((32,32)),np.float32)/255.)
        reference=np.stack(reference) if reference else np.empty((0,32,32),np.float32)
        rows=[];seen={};quarantine=[];threshold=float(cfg.get('SAM_NEAR_DUPLICATE_MAE',.008))
        for number,(key,image_path,mask_path) in enumerate(pairs,1):
            try:
                x=raster_normalize(read_gray(image_path));mask=stone_mask(read_gray(mask_path))
                if x.shape!=mask.shape:raise ValueError('image/mask shape mismatch')
                pixels=np.uint8(np.clip(x,0,1)*255);digest=hashlib.sha256(pixels.tobytes()+str(x.shape).encode()).hexdigest()
                mask_digest=hashlib.sha256(mask.tobytes()).hexdigest()
                if digest in seen:
                    if seen[digest]!=mask_digest:quarantine.append({'id':key,'reason':'duplicate image with conflicting weak mask'})
                    else:quarantine.append({'id':key,'reason':'duplicate image'})
                    continue
                thumb=np.asarray(Image.fromarray(pixels).resize((32,32)),np.float32)/255.
                if len(reference) and float(np.min(np.mean(np.abs(reference-thumb),axis=(1,2))))<threshold:
                    quarantine.append({'id':key,'reason':'near/exact KSSD overlap'});continue
                seen[digest]=mask_digest;y=np.zeros((3,*x.shape),np.uint8);y[0]=mask
                rel=f'sam_weak/{digest[:20]}.npz';np.savez_compressed(proc/rel,x=x.astype(np.float16),y=y)
                rows.append({'id':'sam_'+key,'source':'sam_weak','group':'sam_'+digest[:20],'split':'train','relpath':rel,
                             'slice':0,'has_stone':int(mask.any()),'has_tumor':0,'has_cyst':0,
                             'known_stone':1,'known_tumor':0,'known_cyst':0,'patient_verified':False,'image_hash':digest})
            except (ValueError,OSError) as exc:quarantine.append({'id':key,'reason':str(exc)})
            if number%200==0:print('SAM quality/leakage audit',number,'/',len(pairs),flush=True)
        if not rows:raise ValueError('No safe paired SAM samples remained after quality/leakage checks')
        sam=pd.DataFrame(rows);sam.to_csv(manifest,index=False)
        pd.DataFrame(quarantine,columns=['id','reason']).to_csv(root/'reports'/'sam_weak_quarantine.csv',index=False)
        save_json({'dataset':'bemorekgg/kidney-stone-segmentation-dataset','mask_type':'SAM-generated weak labels',
                   'paired_candidates':len(pairs),'retained_train_only':len(sam),'quarantined':len(quarantine),
                   'unpaired_images':len(unpaired_images),'unpaired_masks':len(unpaired_masks),
                   'evaluation_use':False,'near_kssd_threshold_mae_32x32':threshold},audit_path)
    sam=sam.copy();sam['path']=[str(proc/p) for p in sam.relpath]
    combined=pd.concat([base_frame,sam],ignore_index=True)
    if (combined[combined.source=='sam_weak'].split!='train').any():raise AssertionError('Weak SAM masks must be train-only')
    return combined

class TileDataset(Dataset):
    def __init__(self,frame,tile,seed,training=True):
        self.rows=frame.to_dict('records');self.tile=tile;self.seed=seed;self.training=training
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]
        with np.load(r['path'],allow_pickle=False) as a:x=a['x'].astype(np.float32);y=a['y'].astype(np.float32)
        if not np.isfinite(x).all():raise ValueError('Corrupt processed image')
        t=self.tile;h,w=x.shape;pad=(max(0,t-h),max(0,t-w));x=np.pad(x,((0,pad[0]),(0,pad[1])))
        y=np.pad(y,((0,0),(0,pad[0]),(0,pad[1])),constant_values=-1)
        h,w=x.shape;known=np.array([r['known_'+c] for c in ('stone','tumor','cyst')],np.float32)
        if self.training:
            fg=np.any((y>0)&known[:,None,None].astype(bool),axis=0);pts=np.argwhere(fg)
            if len(pts) and np.random.rand()<.9:
                cy,cx=pts[np.random.randint(len(pts))];top=int(np.clip(cy-np.random.randint(t//4,3*t//4),0,h-t));left=int(np.clip(cx-np.random.randint(t//4,3*t//4),0,w-t))
            else:top=np.random.randint(h-t+1);left=np.random.randint(w-t+1)
        else:
            # Deterministic image-only validation crop; full-volume testing is separate.
            rng=np.random.default_rng(self.seed+index);top=int(rng.integers(h-t+1));left=int(rng.integers(w-t+1))
        x=x[top:top+t,left:left+t];y=y[:,top:top+t,left:left+t]
        if self.training:
            if np.random.rand()<.5:x=x[:,::-1];y=y[:,:,::-1]
            if np.random.rand()<.4:x=np.clip(x*np.random.uniform(.9,1.1)+np.random.uniform(-.03,.03),0,1)
        return torch.from_numpy(x.copy()).unsqueeze(0),torch.from_numpy(y.copy()),torch.from_numpy(known)

def training_weights(frame):
    weights=np.zeros(len(frame),float)
    for source in frame.source.unique():
        ids=np.where(frame.source.to_numpy()==source)[0];sub=frame.iloc[ids]
        # Half of draws per dataset. Within KiTS, boost rare positive slices but retain background.
        raw=np.ones(len(sub))
        for cls in ('stone','tumor','cyst'):
            pos=sub['has_'+cls].to_numpy().astype(bool)
            if pos.any():raw+=pos*(len(sub)/max(1,pos.sum()))
        source_mass=.15 if source=='sam_weak' else 1.0
        weights[ids]=source_mass*raw/raw.sum()/frame.source.nunique()
    return weights