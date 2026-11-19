"""Validation-guarded recovery from the user's trained v2 Swin specialists.

Same model tensors/inference architecture; revised TRAINING recipe and separate v3 state.
No claim of a measured accuracy improvement until this is trained on the real cohort.
"""
import copy
import gc
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from kv_independent_lesions import (load_composite, validation_partitions, predict_probability,
                                   summarize, set_phase, FORMAT)
from kv_protected_finetune import PositiveCache, atomic_save, file_hash, tensor_hash, mask_rule

RECIPE = 'context_precision_ema_v3'


def positive_path(row, cache):
    return cache.get(row) if int(row.has_tumor) or int(row.has_cyst) else Path(row.path)


def read_pair(row, name, cache):
    if int(row['known_'+name]) != 1:
        raise ValueError('Unknown target annotations cannot become background labels.')
    with np.load(positive_path(row,cache),allow_pickle=False) as data:
        x,y=data['x'].astype(np.float32),data['y']
    if (x.ndim!=2 or y.shape!=(3,*x.shape) or not np.isfinite(x).all()
            or x.min()<0 or x.max()>1 or not np.isin(y,[0,1]).all()):
        raise ValueError('Invalid prepared slice: '+str(row.path))
    target=y[1 if name=='tumor' else 2].astype(np.float32)
    if bool(target.any())!=bool(int(row['has_'+name])):
        raise ValueError('Manifest/mask disagreement: '+str(row.path))
    return x,target


def repair_rows(context,rows,cfg):
    if rows.empty:return
    from kv_repair_selected import repair_selected
    repair_selected(Path(cfg['ROOT']),rows.drop_duplicates('id'),
                    allow_download=cfg['ALLOW_TEMPORARY_RAW_DOWNLOAD'])


def expanded_validation(frame,seed,positive_slices=2,negative_slices=2):
    """Preserve the existing validation PATIENT partition; expand within each patient."""
    cal,screen=validation_partitions(frame,seed)
    val=frame[(frame.source=='kits')&(frame.split=='val')].copy()
    output=[]
    for old in (cal,screen):
        part=val[val.group.astype(str).isin(set(old.group.astype(str)))]
        selected=[]
        for _,patient in part.groupby('group',sort=True):
            patient=patient.sort_values(['slice','id'])
            for name in ('tumor','cyst'):
                positive=patient[patient['has_'+name]==1]
                if len(positive):
                    positions=np.unique(np.rint(np.linspace(0,len(positive)-1,
                        min(positive_slices,len(positive))+2)[1:-1]).astype(int))
                    selected.append(positive.iloc[positions])
            negative=patient[(patient.has_tumor==0)&(patient.has_cyst==0)]
            if len(negative):
                positions=np.unique(np.rint(np.linspace(0,len(negative)-1,
                    min(negative_slices,len(negative))+2)[1:-1]).astype(int))
                selected.append(negative.iloc[positions])
        output.append(pd.concat(selected).drop_duplicates('id').reset_index(drop=True))
    if set(output[0].group)&set(output[1].group):raise AssertionError('Patient partition leakage')
    if set(output[0].group)|set(output[1].group)!=set(val.group):
        raise AssertionError('A validation patient was omitted')
    return output


def initialize_recovery(source_path,frame,output_root,cfg):
    source_path=Path(source_path)
    if not source_path.is_file():raise FileNotFoundError('Your trained v2 export is required: '+str(source_path))
    source,metadata=load_composite(source_path,'cpu')
    metadata={k:v for k,v in metadata.items() if k!='model'}
    train=frame[(frame.source=='kits')&(frame.split=='train')].copy().reset_index(drop=True)
    if train.empty or frame[frame.source=='kits'].id.duplicated().any():
        raise ValueError('Invalid or duplicate KiTS manifest rows.')
    if (frame.groupby(['source','group']).split.nunique()>1).any():
        raise ValueError('A patient occurs in more than one split.')
    for name in ('tumor','cyst'):
        if not train['has_'+name].any() or not (train['known_'+name]==1).all():
            raise ValueError('Require known KiTS targets and positive patients for '+name)
    cal,screen=expanded_validation(frame,cfg['SEED'],cfg['VAL_POSITIVE_SLICES'],cfg['VAL_NEGATIVE_SLICES'])
    if set(train.group)&(set(cal.group)|set(screen.group)):raise AssertionError('Train/validation leakage')
    keys=('SEED','BATCH','ENCODER_LR','DECODER_LR','MINIMUM_POSITIVE','CONTEXT_VIEWS',
          'RANDOM_VIEWS','MINED_NEGATIVES','EMA_DECAY','VALIDATE_EVERY','THRESHOLD_GRID',
          'VAL_POSITIVE_SLICES','VAL_NEGATIVE_SLICES','HEAD_WARMUP_ROUNDS','UNFREEZE_ALL_ROUND')
    payload={'recipe':RECIPE,'code_sha256':file_hash(__file__),'source_sha256':file_hash(source_path),
             'train':train[['id','group','slice','has_tumor','has_cyst']].astype(str).values.tolist(),
             'cal':cal.id.astype(str).tolist(),'screen':screen.id.astype(str).tolist(),
             'settings':{k:cfg[k] for k in keys}}
    signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    run=Path(output_root)/signature[:12];run.mkdir(parents=True,exist_ok=True)
    receipt=run/'experiment.json'
    if receipt.is_file() and json.loads(receipt.read_text())['signature']!=signature:
        raise ValueError('Existing recovery configuration differs.')
    if not receipt.is_file():receipt.write_text(json.dumps(dict(payload,signature=signature,
        source_path=str(source_path),prior_selection=metadata.get('specialist_selection',{})),indent=2))
    print('Starting from trained specialists:',metadata.get('specialist_selection',{}),flush=True)
    print('Original epoch:',metadata.get('epoch'),'| recovery:',run,flush=True)
    print('Training patients:',train.group.nunique(),'| calibration/screen patients:',
          cal.group.nunique(),screen.group.nunique(),'| sampled validation slices:',len(cal)+len(screen),flush=True)
    print('New recipe: fresh optimizer; existing trained model weights retained. Own v3 batches resume exactly.',flush=True)
    return {'source':source,'metadata':metadata,'source_path':source_path,'source_sha':payload['source_sha256'],
            'original_sha':tensor_hash(source.original.state_dict()),'signature':signature,
            'train':train,'cal':cal,'screen':screen,'run':run}


def make_plan(frame,name,round_number,cfg,mined):
    """Every training patient + cyclic positive slices + unconditioned contextual views."""
    rng=np.random.default_rng(cfg['SEED']+1009*round_number+(name=='cyst'))
    plan=[]
    key='has_'+name
    for _,patient in frame.groupby('group',sort=True):
        positive=patient[patient[key]==1]
        pool=(positive if len(positive) else patient).sort_values(['slice','id'])
        order=np.random.default_rng(cfg['SEED']+int(hashlib.sha256(str(patient.group.iloc[0]).encode()).hexdigest()[:8],16)).permutation(len(pool))
        position=int(order[(round_number-1)%len(pool)])
        plan.append({'index':int(pool.index[position]),'view':'focus' if len(positive) else 'global'})
    positives=[p.sort_values(['slice','id']) for _,p in frame[frame[key]==1].groupby('group',sort=True)]
    current=sum(p['view']=='focus' for p in plan)
    for repeat in range(max(0,int(cfg['MINIMUM_POSITIVE'])-current)):
        patient=positives[repeat%len(positives)]
        offset=int(hashlib.sha256(str(patient.group.iloc[0]).encode()).hexdigest()[:8],16)
        order=np.random.default_rng(cfg['SEED']+offset).permutation(len(patient))
        position=int(order[(round_number-1+repeat//len(positives))%len(patient)])
        plan.append({'index':int(patient.index[position]),'view':'focus'})
    patients=[p for _,p in frame.groupby('group',sort=True)]
    for view,key in (('global','CONTEXT_VIEWS'),('random','RANDOM_VIEWS')):
        order=rng.permutation(len(patients))
        for j in range(int(cfg[key])):
            patient=patients[int(order[j%len(patients)])]
            # Unconditioned selection: no requirement that a random/context view contains a lesion.
            index=int(rng.choice(patient.index))
            plan.append({'index':index,'view':view})
    plan.extend(mined)
    rng.shuffle(plan)
    covered=set(frame.loc[[p['index'] for p in plan]].group)
    if covered!=set(frame.group):raise AssertionError('Training patient coverage incomplete')
    return plan


class ContextDataset(Dataset):
    def __init__(self,frame,plan,name,tile,cache,seed):
        self.frame,self.plan,self.name,self.tile,self.cache,self.seed=frame,plan,name,int(tile),cache,int(seed)
    def __len__(self):return len(self.plan)
    def __getitem__(self,index):
        item=self.plan[index];row=self.frame.loc[item['index']]
        if str(row.split)!='train':raise ValueError('Training dataset refuses validation/test rows.')
        x,y=read_pair(row,self.name,self.cache)
        rng=np.random.default_rng(self.seed+index)
        if item['view']!='global':
            # Mix native crops and moderate scale changes; no change to model input dimensions.
            side=int(rng.choice([self.tile*3//4,self.tile,self.tile*5//4]))
            h,w=x.shape
            x=np.pad(x,((0,max(0,side-h)),(0,max(0,side-w))))
            y=np.pad(y,((0,max(0,side-h)),(0,max(0,side-w))),constant_values=-1)
            h,w=x.shape
            if item['view']=='focus':
                points=np.argwhere(y>0)
                if not len(points):raise ValueError('Positive target is missing.')
                cy,cx=points[int(rng.integers(len(points)))]
            elif item['view']=='hard':
                cy,cx=float(item['cy'])*h,float(item['cx'])*w
            else:
                cy,cx=int(rng.integers(h)),int(rng.integers(w))
            top=int(np.clip(cy-side//2,0,h-side));left=int(np.clip(cx-side//2,0,w-side))
            x,y=x[top:top+side,left:left+side],y[top:top+side,left:left+side]
        xt=torch.from_numpy(np.ascontiguousarray(x))[None,None]
        yt=torch.from_numpy(np.ascontiguousarray(y))[None,None]
        xt=F.interpolate(xt,size=(self.tile,self.tile),mode='bilinear',align_corners=False)[0]
        yt=F.interpolate(yt,size=(self.tile,self.tile),mode='nearest')[0]
        if item['view']=='focus' and not (yt>0).any():
            # A one-pixel lesion must survive spatial augmentation; preserve its nearest target point.
            location=np.argwhere(y>0)[0]
            yy=min(self.tile-1,int(location[0]*self.tile/y.shape[0]))
            xx=min(self.tile-1,int(location[1]*self.tile/y.shape[1]))
            yt[0,yy,xx]=1.
        # Joint spatial transforms and small CT-window intensity jitter, training only.
        if rng.random()<.5:xt,yt=xt.flip(-1),yt.flip(-1)
        turns=int(rng.integers(4));xt,yt=torch.rot90(xt,turns,(-2,-1)),torch.rot90(yt,turns,(-2,-1))
        xt=(xt*float(rng.uniform(.95,1.05))+float(rng.uniform(-.025,.025))).clamp(0,1)
        return xt.contiguous(),yt.contiguous()


def precision_loss(logits,target):
    """Pixel-normalized BCE + Dice + FP-weighted Tversky; modest true-boundary weighting."""
    z=logits.float();valid=(target>=0).float();y=target.clamp(0,1).float()
    pos=(y*valid).sum((2,3));area=valid.sum((2,3)).clamp_min(1)
    # Bound the positive multiplier instead of equalizing a tiny lesion with the whole background.
    pos_weight=((area-pos+1)/(pos+1)).sqrt().clamp(1,4)[:,:,None,None]
    boundary=(F.max_pool2d(y,3,1,1)+F.max_pool2d(-y,3,1,1)).clamp(0,1)*valid
    weight=valid*(1+boundary)
    bce=F.binary_cross_entropy_with_logits(z,y,reduction='none',pos_weight=pos_weight)
    bce=(bce*weight).sum((2,3))/weight.sum((2,3)).clamp_min(1)
    p=z.sigmoid()*valid
    tp=(p*y).sum((2,3));fp=(p*(1-y)).sum((2,3));fn=((1-p)*y*valid).sum((2,3))
    dice=1-(2*tp+1)/(2*tp+fp+fn+1)
    tversky=1-(tp+1)/(tp+.65*fp+.35*fn+1)
    # Explicit empty-target supervision suppresses spurious regions on background examples.
    empty=p.sum((2,3))/area
    overlap=torch.where(pos>0,.65*dice+.35*tversky,empty)
    return (.5*bce+.5*overlap).mean()


def training_loss(output,target):
    logits,auxiliary=output
    loss=precision_loss(logits,target)
    for logits in auxiliary:
        small=F.interpolate(target,size=logits.shape[-2:],mode='nearest')
        loss=loss+.05*precision_loss(logits,small)
    return loss


@torch.no_grad()
def update_ema(ema,net,decay):
    for old,new in zip(ema.parameters(),net.parameters()):old.mul_(decay).add_(new,alpha=1-decay)
    for old,new in zip(ema.buffers(),net.buffers()):old.copy_(new)


@torch.inference_mode()
def mine_negatives(net,frame,name,cfg,cache,round_number):
    """Choose false-positive centers only from target-negative TRAINING slices."""
    negatives=frame[frame['has_'+name]==0]
    rng=np.random.default_rng(cfg['SEED']+round_number+71)
    groups=[p for _,p in negatives.groupby('group',sort=True)]
    if not groups or cfg['MINED_NEGATIVES']<=0:return []
    chosen=rng.choice(len(groups),size=min(int(cfg['MINED_NEGATIVES']),len(groups)),replace=False)
    result=[];net.eval();tile=int(cfg['TILE_SIZE'])
    for j in chosen:
        patient=groups[int(j)];row=patient.loc[int(rng.choice(patient.index))]
        image,truth=read_pair(row,name,cache)
        if truth.any() or str(row.split)!='train':raise AssertionError('Invalid negative-mining input')
        x=torch.from_numpy(image)[None,None].to(cfg['DEVICE'])
        x=F.interpolate(x,size=(tile,tile),mode='bilinear',align_corners=False)
        with torch.autocast('cuda',dtype=torch.float16,enabled=str(cfg['DEVICE']).startswith('cuda')):
            score=net(x)[0].float().sigmoid()
        score=F.avg_pool2d(score,17,stride=1,padding=8)[0,0]
        flat=int(score.argmax());cy,cx=divmod(flat,tile)
        result.append({'index':int(row.name),'view':'hard','cy':(cy+.5)/tile,'cx':(cx+.5)/tile})
    print(name,': mined',len(result),'training-only hard-negative crop centers',flush=True)
    return result


@torch.inference_mode()
def collect_cached(net,rows,name,context,cfg):
    cache=PositiveCache(cfg['CACHE_DIR'],cfg['CACHE_BYTES'],allowed_splits=('val',))
    records=[]
    for index,(_,row) in enumerate(rows.iterrows(),1):
        image,actual=read_pair(row,name,cache)
        probability=predict_probability(net,image,cfg['TILE_SIZE'],cfg['DEVICE'],cfg['INFER_BATCH'])
        records.append((str(row.group),probability.astype(np.float16),actual>0))
        if index%25==0 or index==len(rows):
            print(f'{name}: validation {index}/{len(rows)}; positive-cache hits={cache.hits}',flush=True)
    return records


def assess(net,context,name,cfg):
    cal=collect_cached(net,context['cal'],name,context,cfg)
    results=[(float(t),summarize(cal,t)) for t in cfg['THRESHOLD_GRID']]
    threshold,_=max(results,key=lambda pair:(pair[1]['selection_score'],pair[1]['precision']))
    del cal
    metrics=summarize(collect_cached(net,context['screen'],name,context,cfg),threshold)
    return threshold,metrics


def allowed_candidate(candidate,best_score,baseline,min_gain):
    return bool(candidate['selection_score']>best_score+min_gain and
                candidate['dice']>=baseline['dice'] and
                candidate['sampled_positive_patient_dice']>=baseline['sampled_positive_patient_dice'])


def paths(context,name):
    folder=context['run']/name;folder.mkdir(parents=True,exist_ok=True)
    return folder/'last.pth',folder/'best.pth'


def status(context,name):
    path=context['run']/name/'status.json'
    if not path.is_file():return {'completed_rounds':0,'bad_rounds':0,'next_batch':0}
    value=json.loads(path.read_text())
    if value['signature']!=context['signature']:raise ValueError('Resume signature mismatch')
    return value


def run_round(context,name,cfg,deadline,initialize_only=False):
    last,best=paths(context,name)
    net=copy.deepcopy(getattr(context['source'],name)).to(cfg['DEVICE'])
    ema=copy.deepcopy(net).requires_grad_(False).eval()
    optimizer=torch.optim.AdamW([
        {'params':list(net.encoder.parameters()),'lr':cfg['ENCODER_LR']},
        {'params':[p for n,p in net.named_parameters() if not n.startswith('encoder.')],'lr':cfg['DECODER_LR']}],
        weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=2,min_lr=1e-7)
    scaler=torch.amp.GradScaler('cuda',enabled=str(cfg['DEVICE']).startswith('cuda'))
    completed=cursor=bad=0;history=[];plan=[];updates=0
    channel=1 if name=='tumor' else 2
    threshold=float(context['metadata']['segmentation_thresholds'][channel])
    if last.is_file():
        saved=torch.load(last,map_location='cpu',weights_only=True)
        if saved['signature']!=context['signature']:raise ValueError('Resume signature mismatch')
        net.load_state_dict(saved['model'],strict=True);ema.load_state_dict(saved['ema'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        completed,cursor,bad=saved['completed_rounds'],saved['next_batch'],saved['bad_rounds']
        history,plan,updates=saved['history'],saved['plan'],saved['updates']
        baseline,best_score,threshold=saved['baseline'],saved['best_score'],saved['threshold']
        torch.set_rng_state(saved['torch_rng'])
        if saved['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(saved['cuda_rng'])
        del saved
    else:
        print(name,': re-evaluating the trained v2 baseline on the expanded validation screen',flush=True)
        minimum=int(context['metadata'].get('mask_postprocess',{}).get(name,{}).get('min_pixels',0))
        baseline=summarize(collect_cached(net,context['screen'],name,context,cfg),threshold,minimum)
        best_score=baseline['selection_score']
        atomic_save({'model':net.state_dict(),'round':0,'threshold':threshold,'metrics':baseline,
            'baseline':baseline,'min_pixels':minimum,'signature':context['signature'],
            'provenance':'existing v2 selected weights; no v3 update','weights':'v2 source'},best)
        torch.manual_seed(cfg['SEED']+(name=='cyst'))
        if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['SEED']+(name=='cyst'))
    if not best.is_file():raise FileNotFoundError('Selected checkpoint missing: '+str(best))
    def save():
        atomic_save({'model':net.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'completed_rounds':completed,
            'next_batch':cursor,'bad_rounds':bad,'history':history,'plan':plan,'updates':updates,
            'baseline':baseline,'best_score':best_score,'threshold':threshold,'signature':context['signature'],
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},last)
        info={'signature':context['signature'],'completed_rounds':completed,'next_batch':cursor,
              'bad_rounds':bad,'best_score':best_score}
        temp=last.parent/'status.tmp';temp.write_text(json.dumps(info));os.replace(temp,last.parent/'status.json')
    if initialize_only:
        save();return True
    if time.monotonic()>=deadline:return False
    round_number=completed+1
    cache=PositiveCache(cfg['CACHE_DIR'],cfg['CACHE_BYTES'])
    if not plan:
        negative=context['train'][context['train']['has_'+name]==0]
        # Repair only a bounded deterministic mining pool before selecting centers.
        mining_rows=negative.groupby('group',sort=True).head(1)
        if len(mining_rows)>cfg['MINED_NEGATIVES']:
            mining_rows=mining_rows.sample(n=cfg['MINED_NEGATIVES'],random_state=cfg['SEED']+round_number)
        if cfg['MINED_NEGATIVES']>0:repair_rows(context,mining_rows,cfg)
        mined=mine_negatives(net,mining_rows,name,cfg,cache,round_number)
        plan=make_plan(context['train'],name,round_number,cfg,mined)
        repair_rows(context,context['train'].loc[sorted({p['index'] for p in plan})],cfg)
        save()  # Persist the exact mining decisions and crop plan before the first update.
    phase=set_phase(net,round_number,cfg)
    ds=ContextDataset(context['train'],plan,name,cfg['TILE_SIZE'],cache,cfg['SEED']+100000*round_number)
    loader=DataLoader(Subset(ds,range(min(cursor*cfg['BATCH'],len(ds)),len(ds))),batch_size=cfg['BATCH'],
        shuffle=False,num_workers=0,pin_memory=str(cfg['DEVICE']).startswith('cuda'),
        generator=torch.Generator().manual_seed(cfg['SEED']+round_number))
    total=math.ceil(len(ds)/cfg['BATCH'])
    print(f'{name}: v3 round {round_number}, next batch {cursor+1}/{total}, {phase}; '
          f'{len(plan)} views; every training patient represented',flush=True)
    in_batch=False
    try:
        for x,y in loader:
            if time.monotonic()>=deadline:
                save();print('Saved partial recovery round:',name,'next batch',cursor+1,flush=True);return False
            in_batch=True
            x,y=x.to(cfg['DEVICE'],non_blocking=True),y.to(cfg['DEVICE'],non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=str(cfg['DEVICE']).startswith('cuda')):
                loss=training_loss(net(x,auxiliary=True),y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; model was not exported.')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=previous_scale:
                updates+=1;update_ema(ema,net,min(cfg['EMA_DECAY'],1-1/(updates+1)))
            cursor+=1
            in_batch=False
            if cursor==1 or cursor%25==0 or cursor==total:
                print(f'{name} round {round_number}: {cursor}/{total}, loss={loss.item():.4f}, '
                      f'SSD hits={cache.hits}, copies={cache.copies}',flush=True)
            if cursor%cfg['SAVE_EVERY']==0:save()
        save()
        do_validation=(round_number%cfg['VALIDATE_EVERY']==0 or round_number==cfg['MAX_ROUNDS_PER_CLASS'])
        if do_validation:
            # EMA weights are the candidate; baseline is retained if either validation floor regresses.
            threshold,metrics=assess(ema,context,name,cfg)
            accepted=allowed_candidate(metrics,best_score,baseline,cfg['MIN_GAIN'])
            if accepted:
                best_score,bad=metrics['selection_score'],0
                atomic_save({'model':ema.state_dict(),'round':round_number,'threshold':threshold,
                    'metrics':metrics,'baseline':baseline,'min_pixels':0,'signature':context['signature'],
                    'weights':'EMA','provenance':'v3 context/precision recovery'},best)
            else:bad+=1
            scheduler.step(metrics['selection_score'])
            history.append({'round':round_number,'phase':phase,'metrics':metrics,'threshold':threshold,
                            'accepted':accepted,'weights':'EMA'})
            print(name,'VALIDATION:',metrics,'| accepted:',accepted,flush=True)
        else:print('Validation scheduled after the next round; no new accuracy claim.',flush=True)
        completed,cursor,plan=round_number,0,[]
        save();return True
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True)
        if not in_batch:
            save();print('Interrupted between updates; completed batch cursor saved.',flush=True)
        else:
            print('Interrupted during an update; retaining the last successfully saved batch checkpoint.',flush=True)
        raise


def train_session(context,cfg):
    if cfg['CACHE_BYTES']>5*1024**3:raise ValueError('Positive-slice cache must stay within 5 GiB.')
    if str(cfg['DEVICE']).startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Select a T4 GPU.')
    repair_rows(context,pd.concat([context['cal'],context['screen']]),cfg)
    deadline=time.monotonic()+float(cfg['SESSION_MINUTES'])*60
    for name in ('tumor','cyst'):
        last,_=paths(context,name)
        if not last.is_file():run_round(context,name,cfg,deadline,initialize_only=True)
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    while time.monotonic()<deadline:
        choices=[]
        for name in ('cyst','tumor'):
            value=status(context,name)
            if value['completed_rounds']<cfg['MAX_ROUNDS_PER_CLASS'] and value['bad_rounds']<cfg['PATIENCE']:
                choices.append((value['completed_rounds'],name))
        if not choices:
            print('Round or patience limit reached; inspect development history before a new experiment.',flush=True);break
        _,name=min(choices,key=lambda p:(p[0],0 if p[1]=='cyst' else 1))
        if not run_round(context,name,cfg,deadline):break
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if tensor_hash(context['source'].original.state_dict())!=context['original_sha']:
        raise RuntimeError('Protected original weights changed.')
    print('SESSION SAVED:',context['run'],flush=True)
    return context['run']


def report(context):
    output=[]
    for name in ('tumor','cyst'):
        last,best=paths(context,name)
        info=status(context,name)
        if best.is_file():
            print('Reading selected recovery checkpoint:',name,flush=True)
            saved=torch.load(best,map_location='cpu',weights_only=True)
            output.append({'lesion':name,'completed_rounds':info['completed_rounds'],
                'saved_batches_in_next_round':info['next_batch'],'selected_v3_round':saved['round'],
                'weights':saved['weights'],'threshold':saved['threshold'],**saved['metrics']})
            del saved
    return pd.DataFrame(output)


def export_model(context):
    print('Export 1/4: restore frozen source and validation-selected specialists',flush=True)
    if file_hash(context['source_path'])!=context['source_sha']:raise ValueError('Source export changed during recovery.')
    model=copy.deepcopy(context['source']).cpu().eval()
    metadata=dict(context['metadata']);thresholds=list(metadata['segmentation_thresholds'])
    post=copy.deepcopy(metadata.get('mask_postprocess',{}));selection={}
    for channel,name in ((1,'tumor'),(2,'cyst')):
        _,best=paths(context,name)
        saved=torch.load(best,map_location='cpu',weights_only=True)
        if saved['signature']!=context['signature']:raise ValueError('Checkpoint signature mismatch.')
        getattr(model,name).load_state_dict(saved['model'],strict=True)
        thresholds[channel]=float(saved['threshold']);post[name]={'min_pixels':saved['min_pixels']}
        selection[name]={'round':saved['round'],'development_metrics':saved['metrics'],
                         'baseline_metrics':saved['baseline'],'weights':saved['weights']}
        del saved
    if tensor_hash(model.original.state_dict())!=context['original_sha']:raise RuntimeError('Original branch changed.')
    metadata['prior_specialist_selection']=copy.deepcopy(metadata.get('specialist_selection',{}))
    metadata.update(format=FORMAT,model=model.state_dict(),segmentation_thresholds=thresholds,
        mask_postprocess=post,specialist_selection=selection,original_sha256=context['original_sha'],
        recovery_recipe=RECIPE,recovery_signature=context['signature'],
        prior_v2_source_sha256=context['source_sha'],goal_80_verified=False)
    out=context['run']/'export';out.mkdir(parents=True,exist_ok=True)
    print('Export 2/4: save ONE model.pth',flush=True);atomic_save(metadata,out/'model.pth')
    print('Export 3/4: matching architecture/inference code',flush=True)
    for name in ('kv_independent_lesions','kv_protected_finetune','kv_multitask','kv_model','kv_infer','kv_joint_infer','kv_data'):
        module=__import__(name);shutil.copyfile(module.__file__,out/(name+'.py'))
    (out/'requirements.txt').write_text('torch>=2.3\nnumpy\npandas\nscipy\nPillow\nmatplotlib\nnibabel\npydicom\nrequests\n')
    (out/'README.txt').write_text('ONE model.pth; matching loader: kv_independent_lesions.load_composite.\n'
        'Classification: normal, stone, tumor, cyst. Independent masks: stone, tumor, cyst.\n'
        'Frozen original branch retained; three Swin paths inside one module.\n'
        'New recovery weights were selected only on validation. Development scores are sampled slices.\n'
        'No 80% or external-performance guarantee. Physical millimetres require verified spacing.\n')
    print('Export 4/4: strict reload and stone/classification equality check',flush=True)
    check,_=load_composite(out/'model.pth','cpu')
    tile=int(metadata['tile_size']);probe=torch.rand((1,1,tile,tile),generator=torch.Generator().manual_seed(61))
    with torch.inference_mode():
        old=context['source'].original(probe,task='both');new=check.original(probe,task='both')
    for key in ('classification_logits','segmentation_logits'):
        torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
    (out/'protection_check.json').write_text(json.dumps({'original_checksum_preserved':True,
        'classification_and_stone_logits_identical_on_cpu_probe':True,
        'scope':'source weights and same preprocessing/policy; not an accuracy measurement'},indent=2))
    del model,check
    print('EXPORT COMPLETE:',out/'model.pth','| selection:',selection,flush=True)
    return out


@torch.inference_mode()
def preview_masks(context,export_dir,cfg):
    """Actual selected masks on validation slices, including a background-only example."""
    import matplotlib.pyplot as plt
    from IPython.display import display
    model,meta=load_composite(Path(export_dir)/'model.pth',cfg['DEVICE'])
    cache=PositiveCache(cfg['CACHE_DIR'],cfg['CACHE_BYTES'],allowed_splits=('val',))
    for name,channel in (('tumor',1),('cyst',2)):
        positives=context['screen'][context['screen']['has_'+name]==1].head(2)
        negatives=context['screen'][(context['screen'].has_tumor==0)&(context['screen'].has_cyst==0)].head(1)
        examples=pd.concat([positives,negatives]).drop_duplicates('id')
        for _,row in examples.iterrows():
            image,truth=read_pair(row,name,cache)
            probability=predict_probability(getattr(model,name),image,cfg['TILE_SIZE'],cfg['DEVICE'],cfg['INFER_BATCH'])
            mask=mask_rule(probability,meta['segmentation_thresholds'][channel],
                           meta.get('mask_postprocess',{}).get(name,{}).get('min_pixels',0))
            fig,axes=plt.subplots(1,3,figsize=(12,4))
            for ax,title in zip(axes,('CT','Annotated mask','Predicted mask')):
                ax.imshow(image,cmap='gray',vmin=0,vmax=1);ax.set_title(title);ax.axis('off')
            axes[1].imshow(np.ma.masked_where(truth==0,truth),cmap='autumn',alpha=.6,vmin=0,vmax=1)
            axes[2].imshow(np.ma.masked_where(~mask,mask),cmap='autumn',alpha=.6,vmin=0,vmax=1)
            fig.suptitle(f'{name}: {row.group}, slice {row.slice}; annotated={int(truth.sum())}, predicted={int(mask.sum())} pixels')
            display(fig);plt.close(fig)
    del model
