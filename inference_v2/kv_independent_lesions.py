"""Independent Swin specialists; validation goals are measured, never guaranteed."""
import copy
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from kv_protected_finetune import (load_parent, tensor_hash, file_hash, atomic_save,
                                   PositiveCache, mask_rule)

FORMAT = 'kidneyvision_independent_swin_v2'


class LesionSwin(nn.Module):
    """One lesion's complete encoder/decoder, initialized from existing trained tensors."""
    def __init__(self, protected_parent, channel):
        super().__init__()
        self.encoder = copy.deepcopy(protected_parent.base.encoder)
        decoder = protected_parent.expert
        for name in ('lateral', 'fuse', 'stem', 'final'):
            setattr(self, name, copy.deepcopy(getattr(decoder, name)))
        self.head = nn.Conv2d(decoder.head.in_channels, 1, 1)
        with torch.no_grad():
            self.head.weight.copy_(decoder.head.weight[channel-1:channel])
            self.head.bias.copy_(decoder.head.bias[channel-1:channel])
        self.aux = nn.ModuleList([nn.Conv2d(64, 1, 1) for _ in range(2)])
        for head in self.aux:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -3.)
        self.requires_grad_(True)

    def forward(self, x, auxiliary=False):
        feats = self.encoder(x)
        z = self.lateral[3](feats[3])
        deep = []
        for j, i in enumerate((2, 1, 0)):
            z = self.fuse[j](torch.cat((F.interpolate(z, size=feats[i].shape[-2:],
                mode='bilinear', align_corners=False), self.lateral[i](feats[i])), 1))
            if auxiliary and j in (1, 2):
                deep.append(self.aux[j-1](z))
        z = F.interpolate(z, size=x.shape[-2:], mode='bilinear', align_corners=False)
        logits = self.head(self.final(torch.cat((z, self.stem(x)), 1)))
        return logits, deep if auxiliary else []


class CompositeSwin(nn.Module):
    def __init__(self, parent, protected_parent=None):
        super().__init__()
        original=parent if protected_parent is None else protected_parent
        self.original = copy.deepcopy(original.base).requires_grad_(False).eval()
        self.tumor = LesionSwin(parent, 1)
        self.cyst = LesionSwin(parent, 2)

    def train(self, mode=True):
        super().train(mode)
        self.original.eval()
        return self

    def forward(self, x, task='seg'):
        if task not in ('seg', 'cls', 'both'):
            raise ValueError('task must be seg, cls, or both')
        old = self.original(x, task=task)
        if task == 'cls':
            return old
        stone = old['segmentation_logits'][:, :1] if task == 'both' else old[:, :1]
        tumor = self.tumor(x)[0]
        cyst = self.cyst(x)[0]
        seg = torch.cat((stone, tumor, cyst), 1)
        return {'segmentation_logits': seg, 'classification_logits': old['classification_logits']} if task == 'both' else seg


def set_phase(branch, round_number, cfg):
    """Optimizer groups exist from the start; progressive unfreezing preserves their state."""
    warm = round_number <= cfg['HEAD_WARMUP_ROUNDS']
    full = round_number >= cfg['UNFREEZE_ALL_ROUND']
    for name, parameter in branch.named_parameters():
        encoder = name.startswith('encoder.')
        late = name.startswith(('encoder.stages.2.', 'encoder.stages.3.', 'encoder.mergers.2.', 'encoder.norm.'))
        parameter.requires_grad_(not encoder or (not warm and (full or late)))
    branch.train()
    branch.encoder.gradient_checkpointing = True
    if warm:
        branch.encoder.eval()
    return 'decoder warm-up' if warm else ('full lesion encoder' if full else 'last two Swin stages')


def binary_loss(logits, target):
    z = logits.float()
    valid = (target >= 0).float()
    y = target.clamp(0, 1).float()
    positive, negative = y * valid, (1-y) * valid
    area, background = positive.sum((2,3)), negative.sum((2,3))
    pos = (F.softplus(-z)*positive).sum((2,3))/area.clamp_min(1)
    neg_values = F.softplus(z)*negative
    neg = neg_values.sum((2,3))/background.clamp_min(1)
    # A modest hard-negative term targets false positives without dominating tiny lesions.
    k = max(1, z.shape[-1]*z.shape[-2]//100)
    hard = neg_values.flatten(2).topk(k, dim=2).values.mean(2)
    neg = .85*neg+.15*hard
    bce = torch.where(area > 0, .5*pos+.5*neg, neg)
    p = z.sigmoid()*valid
    dice = 1-(2*(p*y).sum((2,3))+1)/(p.sum((2,3))+area+1)
    return (.5*bce+.5*dice*(area > 0)).mean()


def total_loss(output, target):
    logits, deep = output
    loss = binary_loss(logits, target)
    for auxiliary in deep:
        down = F.interpolate(target, size=auxiliary.shape[-2:], mode='nearest')
        loss = loss+.1*binary_loss(auxiliary, down)
    return loss


def plan_round(frame, name, seed, minimum_positive=256):
    rng = np.random.default_rng(seed)
    plan = []
    key = 'has_'+name
    # Every existing TRAINING patient is represented; validation/test patients never enter this plan.
    for _, group in frame.groupby('group', sort=True):
        positive = group[group[key] == 1]
        if len(positive):
            pool = positive
        else:
            tc = group[(group.has_tumor == 1) | (group.has_cyst == 1)]
            pool = tc if len(tc) else group
        plan.append((int(rng.choice(pool.index)), bool(len(positive))))
    groups = [g for _, g in frame[frame[key] == 1].groupby('group', sort=True)]
    if not groups:
        raise ValueError('No positive patients for '+name)
    current = sum(positive for _, positive in plan)
    for _ in range(max(0, minimum_positive-current)):
        group = groups[int(rng.integers(len(groups)))]
        plan.append((int(rng.choice(group.index)), True))
    rng.shuffle(plan)
    return plan


class LesionDataset(Dataset):
    def __init__(self, frame, plan, name, tile, cache, seed, augment=True):
        self.frame, self.plan, self.name = frame, plan, name
        self.tile, self.cache, self.seed, self.augment = tile, cache, seed, augment
    def __len__(self):
        return len(self.plan)
    def __getitem__(self, index):
        row_index, positive = self.plan[index]
        row = self.frame.loc[row_index]
        if int(row['known_'+self.name]) != 1:
            raise ValueError('Unannotated target cannot be treated as negative')
        path = self.cache.get(row) if int(row.has_tumor) or int(row.has_cyst) else Path(row.path)
        with np.load(path, allow_pickle=False) as d:
            x, all_y = d['x'].astype(np.float32), d['y']
        channel = 1 if self.name == 'tumor' else 2
        if x.ndim != 2 or all_y.shape != (3, *x.shape) or not np.isfinite(x).all() or not np.isin(all_y,[0,1]).all():
            raise ValueError('Invalid prepared slice: '+str(path))
        y = all_y[channel].astype(np.float32)
        if bool(y.any()) != bool(int(row['has_'+self.name])):
            raise ValueError('Manifest and actual mask disagree: '+str(path))
        rng = np.random.default_rng(self.seed+index)
        ph, pw = max(0,self.tile-x.shape[0]), max(0,self.tile-x.shape[1])
        x = np.pad(x,((0,ph),(0,pw)))
        y = np.pad(y,((0,ph),(0,pw)),constant_values=-1)
        h,w = x.shape
        # Two thirds of positive requests explicitly retain a lesion; other crops supply context/background.
        focus = positive and (not self.augment or index % 3 != 2)
        if focus:
            points = np.argwhere(y > 0)
            cy,cx = points[int(rng.integers(len(points)))]
            top = int(np.clip(cy-int(rng.integers(self.tile//4,3*self.tile//4)),0,h-self.tile))
            left = int(np.clip(cx-int(rng.integers(self.tile//4,3*self.tile//4)),0,w-self.tile))
        else:
            top,left = int(rng.integers(h-self.tile+1)),int(rng.integers(w-self.tile+1))
        x,y = x[top:top+self.tile,left:left+self.tile],y[top:top+self.tile,left:left+self.tile]
        if focus and not (y > 0).any():
            raise AssertionError('Target disappeared during crop')
        if self.augment and index % 2:
            x,y = x[:,::-1],y[:,::-1]
        return torch.from_numpy(x.copy())[None],torch.from_numpy(y.copy())[None]


def validation_partitions(frame, seed):
    val = frame[(frame.source == 'kits') & (frame.split == 'val')].copy()
    patients = np.asarray(sorted(val.group.astype(str).unique()))
    for offset in range(100):
        order = np.random.default_rng(seed+offset).permutation(patients)
        first = set(order[:len(order)//2])
        parts = (val[val.group.astype(str).isin(first)],val[~val.group.astype(str).isin(first)])
        if len(patients) >= 8 and all(p.has_tumor.any() and p.has_cyst.any() for p in parts):
            break
    else:
        raise ValueError('Need independent calibration/screening patients with both lesions')
    selected = []
    for part in parts:
        picks = []
        for _, group in part.groupby('group',sort=True):
            for name in ('tumor','cyst'):
                positive = group[group['has_'+name] == 1].sort_values('slice')
                if len(positive):
                    picks.append(positive.iloc[[len(positive)//2]])
            negative = group[(group.has_tumor == 0) & (group.has_cyst == 0)].sort_values('slice')
            if len(negative):
                picks.append(negative.iloc[[len(negative)//2]])
            elif not (group.has_tumor.any() or group.has_cyst.any()):
                picks.append(group.iloc[[len(group)//2]])
        selected.append(pd.concat(picks).drop_duplicates('id').reset_index(drop=True))
    if set(selected[0].group)&set(selected[1].group):
        raise AssertionError('Calibration and screening patients overlap')
    return selected


@torch.inference_mode()
def predict_probability(branch, image, tile, device='cuda', infer_batch=4):
    branch.eval()
    image = np.asarray(image,np.float32)
    h,w = image.shape
    image = np.pad(image,((0,max(0,tile-h)),(0,max(0,tile-w))))
    hp,wp = image.shape
    step = max(1,int(tile*.75))
    def starts(size):
        return sorted(set(list(range(0,size-tile+1,step))+[size-tile]))
    coords = [(y,x) for y in starts(hp) for x in starts(wp)]
    result,denom = np.zeros((hp,wp),np.float32),np.zeros((hp,wp),np.float32)
    ramp = np.hanning(tile).astype(np.float32)
    weight = np.maximum(np.outer(ramp,ramp),.01)
    for begin in range(0,len(coords),infer_batch):
        chunk = coords[begin:begin+infer_batch]
        batch = torch.from_numpy(np.stack([image[y:y+tile,x:x+tile] for y,x in chunk]))[:,None].to(device)
        with torch.autocast('cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
            probabilities = branch(batch)[0].float().sigmoid()[:,0].cpu().numpy()
        for probability,(y,x) in zip(probabilities,chunk):
            result[y:y+tile,x:x+tile] += probability*weight
            denom[y:y+tile,x:x+tile] += weight
    return (result/denom)[:h,:w]


def summarize(records, threshold, min_pixels=0):
    total = np.zeros(3,np.float64)
    patients = {}
    for group, probability, actual in records:
        mask = mask_rule(probability, threshold, min_pixels)
        count = np.asarray([(mask&actual).sum(),mask.sum(),actual.sum()],np.float64)
        total += count
        patients.setdefault(group,np.zeros(3,np.float64))[:] += count
    tp,pred,actual = total
    means = [2*a/max(b+c,1) for a,b,c in patients.values() if c > 0]
    metrics = {'dice':float(2*tp/max(pred+actual,1)),
        'sampled_positive_patient_dice':float(np.mean(means)) if means else 0.,
        'precision':float(tp/max(pred,1)), 'recall':float(tp/max(actual,1))}
    metrics['selection_score'] = min(metrics['dice'],metrics['sampled_positive_patient_dice'])
    return metrics


@torch.inference_mode()
def collect(branch, rows, name, meta, cfg):
    records = []
    channel = 1 if name == 'tumor' else 2
    for index,row in enumerate(rows.itertuples(),1):
        with np.load(row.path,allow_pickle=False) as d:
            image,actual = d['x'].astype(np.float32),d['y'][channel] > 0
        probability = predict_probability(branch,image,meta['tile_size'],cfg['DEVICE'],cfg['INFER_BATCH'])
        records.append((str(row.group),probability.astype(np.float16),actual))
        if index % 20 == 0 or index == len(rows):
            print(f'{name}: validation {index}/{len(rows)} native slices, streamed without SSD caching',flush=True)
    return records


def assess(branch, cal, screen, name, meta, cfg):
    calibration = collect(branch,cal,name,meta,cfg)
    grid = cfg['THRESHOLD_GRID']
    threshold = max(grid,key=lambda t:(summarize(calibration,t)['selection_score'],
                                       summarize(calibration,t)['precision']))
    del calibration
    metrics = summarize(collect(branch,screen,name,meta,cfg),threshold)
    return float(threshold),metrics


def _data_signature(train,cal,screen,cfg,parent_sha):
    columns = ['id','group','slice','has_tumor','has_cyst']
    keys = ('SEED','BATCH','DECODER_LR','ENCODER_LR','MINIMUM_POSITIVE','HEAD_WARMUP_ROUNDS',
            'UNFREEZE_ALL_ROUND','THRESHOLD_GRID')
    payload = {'source':parent_sha,'train':train[columns].astype(str).values.tolist(),
        'cal':cal[columns].astype(str).values.tolist(),'screen':screen[columns].astype(str).values.tolist(),
        'settings':{key:cfg[key] for key in keys}}
    prior=Path(cfg.get('PRIOR_RUN_PATH',''))/'last_finetune.pth'
    payload['prior_saved_decoder_sha256']=file_hash(prior) if cfg.get('PRIOR_RUN_PATH') and prior.is_file() else None
    protected=cfg.get('PROTECTED_MODEL_PATH')
    payload['protected_checkpoint_sha256']=file_hash(protected) if protected else parent_sha
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def initialize(source_path,frame,output_root,cfg):
    parent_sha = file_hash(source_path)
    parent,metadata = load_parent(source_path,'cpu')
    metadata.setdefault('tile_size',256)
    metadata.setdefault('segmentation_thresholds',[.5,.5,.5])
    protected_path=Path(cfg.get('PROTECTED_MODEL_PATH') or source_path)
    if protected_path==Path(source_path):
        protected_parent,protected_meta=parent,metadata
    else:
        protected_parent,protected_meta=load_parent(protected_path,'cpu')
        if dict(protected_meta['encoder_config'])!=dict(metadata['encoder_config']) or int(protected_meta.get('tile_size',256))!=int(metadata['tile_size']):
            raise ValueError('Protected and lesion source architectures/tile sizes differ; supply matching existing Swin runs.')
    lesion_source_sha=tensor_hash(parent.base.state_dict())
    thresholds=list(metadata['segmentation_thresholds'])
    thresholds[0]=float(protected_meta.get('segmentation_thresholds',[.5]*3)[0])
    post=dict(metadata.get('mask_postprocess',{}))
    post['stone']=protected_meta.get('mask_postprocess',{}).get('stone',{})
    metadata=dict(metadata,segmentation_thresholds=thresholds,mask_postprocess=post)
    train = frame[(frame.source == 'kits') & (frame.split == 'train')].copy().reset_index(drop=True)
    cal,screen = validation_partitions(frame,cfg['SEED'])
    if set(train.group)&(set(cal.group)|set(screen.group)):
        raise ValueError('Train and validation patients overlap')
    signature = _data_signature(train,cal,screen,cfg,parent_sha)
    run = Path(output_root)/signature[:12]
    run.mkdir(parents=True,exist_ok=True)
    metadata = {k:metadata[k] for k in ('encoder_config','classes','mask_classes','tile_size',
                 'segmentation_thresholds','mask_postprocess','epoch') if k in metadata}
    original_sha = tensor_hash(protected_parent.base.state_dict())
    receipt = {'signature':signature,'parent_sha256':parent_sha,'source_path':str(source_path),
        'original_sha256':original_sha,'calibration_ids':cal.id.astype(str).to_list(),
        'screen_ids':screen.id.astype(str).to_list(),'original_epoch':metadata.get('epoch'),
        'version':FORMAT,'settings':{k:v for k,v in cfg.items() if isinstance(v,(str,int,float,bool,list))}}
    receipt['protected_source_path']=str(protected_path)
    record = run/'experiment.json'
    if record.is_file() and json.loads(record.read_text())['signature'] != signature:
        raise ValueError('Existing experiment signature changed')
    if not record.is_file():
        record.write_text(json.dumps(receipt,indent=2))
    print('Original epoch:',metadata.get('epoch'),'| independent experiment:',run,flush=True)
    print('Training patients:',train.group.nunique(),'| calibration patients:',cal.group.nunique(),
          '| screening patients:',screen.group.nunique(),flush=True)
    return {'parent':parent,'protected_parent':protected_parent,'protected_source_path':str(protected_path),
            'lesion_source_base_sha':lesion_source_sha,'metadata':metadata,'train':train,'cal':cal,'screen':screen,
            'run':run,'signature':signature,'original_sha':original_sha,'source_path':str(source_path)}


def learnability_audit(context,cfg):
    """Temporary fixed-crop training test; never seeds the real experiment or its optimizer."""
    from IPython.display import display
    import matplotlib.pyplot as plt
    cache = PositiveCache(cfg['CACHE_DIR'],cfg['CACHE_BYTES'])
    train = context['train']
    report = {}
    # Restore global generators so enabling the diagnostic cannot alter the real run's initialization.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        for channel,name in ((1,'tumor'),(2,'cyst')):
            positives = train[train['has_'+name] == 1].drop_duplicates('group').head(cfg['AUDIT_PATIENTS'])
            plan = [(int(index),True) for index in positives.index]
            ds = LesionDataset(train,plan,name,context['metadata']['tile_size'],cache,cfg['SEED'],augment=False)
            patches = [ds[i] for i in range(len(ds))]
            fig,axes = plt.subplots(1,len(patches),figsize=(4*len(patches),4),squeeze=False)
            for ax,(x,y) in zip(axes[0],patches):
                ax.imshow(x[0],cmap='gray',vmin=0,vmax=1)
                ax.contour(y[0].numpy()>0,levels=[.5],colors=['lime'])
                ax.set_title(name+' training mask');ax.axis('off')
            display(fig);plt.close(fig)
            net = LesionSwin(context['parent'],channel).to(cfg['DEVICE'])
            audit_cfg = dict(cfg,HEAD_WARMUP_ROUNDS=0,UNFREEZE_ALL_ROUND=1)
            set_phase(net,1,audit_cfg)
            optimizer = torch.optim.AdamW(net.parameters(),lr=cfg['AUDIT_LR'])
            scaler = torch.amp.GradScaler('cuda',enabled=str(cfg['DEVICE']).startswith('cuda'))
            losses = []
            for step in range(cfg['AUDIT_STEPS']):
                x,y = patches[step%len(patches)]
                x,y = x[None].to(cfg['DEVICE']),y[None].to(cfg['DEVICE'])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16,enabled=str(cfg['DEVICE']).startswith('cuda')):
                    loss = total_loss(net(x,auxiliary=True),y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite small-subset training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if net.head.weight.grad is None or not torch.isfinite(net.head.weight.grad).all():
                    raise RuntimeError('Missing/nonfinite lesion-head gradient')
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                scaler.step(optimizer);scaler.update()
                losses.append(float(loss.item()))
                if (step+1)%20 == 0:
                    print(f'{name} diagnostic fixed-patch step {step+1}/{cfg["AUDIT_STEPS"]}, loss={loss.item():.4f}',flush=True)
            net.eval();dice = []
            with torch.inference_mode():
                for x,y in patches:
                    z = net(x[None].to(cfg['DEVICE']))[0].float().sigmoid()[0,0].cpu().numpy()
                    gt = y[0].numpy()>0;pred = z >= .5
                    dice.append(float(2*(pred & gt).sum()/max(pred.sum()+gt.sum(),1)))
            report[name] = {'fixed_training_patch_dice':float(np.mean(dice)),
                'first_10_loss':float(np.mean(losses[:10])),'last_10_loss':float(np.mean(losses[-10:])),
                'note':'Diagnostic training patches only; not validation or test accuracy.'}
            print(name,'learnability:',report[name],flush=True)
            del net,optimizer,scaler,patches
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng:torch.cuda.set_rng_state_all(cuda_rng)
    (context['run']/'learnability_audit.json').write_text(json.dumps(report,indent=2))
    return report


def _paths(context,name):
    folder = context['run']/name
    return folder/'last.pth',folder/'best.pth'


def class_status(context,name):
    last,best = _paths(context,name)
    status=last.parent/'status.json'
    if status.is_file():
        state=json.loads(status.read_text())
        if state['signature']!=context['signature']:raise ValueError('Class status signature mismatch')
        return int(state['completed_rounds']),int(state['bad_rounds'])
    if last.is_file():
        state = torch.load(last,map_location='cpu',weights_only=True)
        return int(state['completed_rounds']),int(state['bad_rounds'])
    return 0,0


def run_class_round(context,name,cfg,deadline):
    channel = 1 if name == 'tumor' else 2
    last,best_path = _paths(context,name)
    net = LesionSwin(context['parent'],channel).to(cfg['DEVICE'])
    groups = [{'params':list(net.encoder.parameters()),'lr':cfg['ENCODER_LR']},
              {'params':[p for n,p in net.named_parameters() if not n.startswith('encoder.')],
               'lr':cfg['DECODER_LR']}]
    optimizer = torch.optim.AdamW(groups,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=3,min_lr=1e-7)
    scaler = torch.amp.GradScaler('cuda',enabled=str(cfg['DEVICE']).startswith('cuda'))
    completed=next_batch=bad=0
    history=[]
    best_score=-1.
    threshold=float(context['metadata']['segmentation_thresholds'][channel])
    if last.is_file():
        state=torch.load(last,map_location='cpu',weights_only=True)
        if state['signature'] != context['signature']:
            raise ValueError('Resume signature mismatch')
        net.load_state_dict(state['model'],strict=True)
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        completed,next_batch,bad=int(state['completed_rounds']),int(state['next_batch']),int(state['bad_rounds'])
        history,best_score,threshold=state['history'],float(state['best_score']),float(state['threshold'])
        torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        minimum=int(context['metadata'].get('mask_postprocess',{}).get(name,{}).get('min_pixels',0))
        baseline=summarize(collect(net,context['screen'],name,context['metadata'],cfg),threshold,minimum)
        best_score=baseline['selection_score']
        provenance='existing source model'
        prior_path=Path(cfg.get('PRIOR_RUN_PATH',''))/'last_finetune.pth'
        if cfg.get('PRIOR_RUN_PATH') and prior_path.is_file():
            prior=torch.load(prior_path,map_location='cpu',weights_only=True)
            source_ok=prior.get('source_sha256')==file_hash(context['source_path'])
            source_ok=source_ok and prior.get('frozen_base_sha256')==context['lesion_source_base_sha']
            if source_ok:
                old_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                candidate=net.state_dict()
                for key,value in prior['expert'].items():
                    if key in candidate:
                        candidate[key]=value[channel-1:channel] if key.startswith('head.') else value
                net.load_state_dict(candidate,strict=True)
                prior_metrics=summarize(collect(net,context['screen'],name,context['metadata'],cfg),threshold,minimum)
                if prior_metrics['selection_score']>best_score:
                    baseline=prior_metrics;best_score=baseline['selection_score']
                    provenance='prior saved v1 round '+str(prior.get('completed_rounds',0))+' decoder'
                else:
                    net.load_state_dict(old_state,strict=True)
                del old_state,prior
            else:
                print('Prior v1 source differs; its tensors are not mixed into this experiment.',flush=True)
        atomic_save({'model':net.state_dict(),'round':0,'threshold':threshold,'metrics':baseline,
                     'signature':context['signature'],'min_pixels':minimum,
                     'selection':'saved source threshold and region rule, no v2 training',
                     'initialization':provenance},best_path)
        print(name,'BASELINE:',baseline,'| initialization:',provenance,flush=True)
    if not best_path.is_file():raise FileNotFoundError('Best checkpoint unavailable: '+str(best_path))
    def save():
        atomic_save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'completed_rounds':completed,'next_batch':next_batch,'bad_rounds':bad,
            'best_score':best_score,'threshold':threshold,'history':history,
            'signature':context['signature'],'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},last)
        status={'signature':context['signature'],'completed_rounds':completed,'next_batch':next_batch,
                'bad_rounds':bad,'best_score':best_score}
        temp=last.parent/'status.tmp'
        temp.write_text(json.dumps(status));os.replace(temp,last.parent/'status.json')
    save()
    if time.monotonic() >= deadline:
        return False
    round_number=completed+1
    phase=set_phase(net,round_number,cfg)
    print(f'{name}: specialist round {round_number}, next batch {next_batch+1}, phase={phase}',flush=True)
    plan=plan_round(context['train'],name,cfg['SEED']+round_number, cfg['MINIMUM_POSITIVE'])
    selected=context['train'].loc[sorted({index for index,_ in plan})]
    if cfg.get('REPAIR_SELECTED_FILES',True):
        from kv_repair_selected import repair_selected
        repair_selected(Path(cfg['ROOT']),selected,
                        allow_download=cfg['ALLOW_TEMPORARY_RAW_DOWNLOAD'])
    cache=PositiveCache(cfg['CACHE_DIR'],cfg['CACHE_BYTES'])
    ds=LesionDataset(context['train'],plan,name,context['metadata']['tile_size'],cache,
                     cfg['SEED']+100000*round_number,augment=True)
    total=math.ceil(len(ds)/cfg['BATCH'])
    loader=DataLoader(Subset(ds,range(min(next_batch*cfg['BATCH'],len(ds)),len(ds))),
        batch_size=cfg['BATCH'],shuffle=False,num_workers=0,pin_memory=str(cfg['DEVICE']).startswith('cuda'),
        generator=torch.Generator().manual_seed(cfg['SEED']+round_number))
    try:
        for x,y in loader:
            if time.monotonic() >= deadline:
                save();print('Session budget reached; partial '+name+' round saved.',flush=True)
                return False
            x,y=x.to(cfg['DEVICE'],non_blocking=True),y.to(cfg['DEVICE'],non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,enabled=str(cfg['DEVICE']).startswith('cuda')):
                loss=total_loss(net(x,auxiliary=True),y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            scaler.step(optimizer);scaler.update();next_batch+=1
            if next_batch%20 == 0 or next_batch == 1 or next_batch == total:
                print(f'{name} round {round_number}: {next_batch}/{total}, loss={loss.item():.4f}',flush=True)
            if next_batch%cfg['SAVE_EVERY'] == 0:save()
        save()
        threshold,metrics=assess(net,context['cal'],context['screen'],name,context['metadata'],cfg)
        accepted=metrics['selection_score'] > best_score+cfg['MIN_GAIN']
        if accepted:
            best_score,bad=metrics['selection_score'],0
            atomic_save({'model':net.state_dict(),'round':round_number,'threshold':threshold,
                'metrics':metrics,'signature':context['signature'],'min_pixels':0,
                'selection':'independent per-class minimum Dice score'},best_path)
        else:bad+=1
        scheduler.step(metrics['selection_score'])
        completed,next_batch=round_number,0
        history.append({'round':round_number,'phase':phase,'metrics':metrics,'threshold':threshold,'accepted':accepted})
        save()
        print(name,'RESULT:',metrics,'| accepted:',accepted,flush=True)
        return True
    except KeyboardInterrupt:
        optimizer.zero_grad(set_to_none=True);save()
        print('Interrupted; a recovery checkpoint was written.',flush=True)
        raise


def train_session(context,cfg):
    if str(cfg['DEVICE']).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a T4 GPU for full training')
    deadline=time.monotonic()+60*cfg['SESSION_MINUTES']
    for name in ('tumor','cyst'):
        last,best=_paths(context,name)
        if not best.is_file():
            # Persist both baseline branches, even when the session only trains one class.
            run_class_round(context,name,cfg,time.monotonic()-1)
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    while time.monotonic()<deadline:
        options=[]
        for name in ('tumor','cyst'):
            rounds,bad=class_status(context,name)
            if rounds<cfg['MAX_ROUNDS_PER_CLASS'] and bad<cfg['PATIENCE']:
                options.append((rounds,name))
        if not options:
            print('Both specialists reached their round/patience limits; review validation history before extending.',flush=True)
            break
        _,name=min(options,key=lambda x:(x[0],0 if x[1]=='cyst' else 1))
        if not run_class_round(context,name,cfg,deadline):break
        gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if tensor_hash(context['protected_parent'].base.state_dict())!=context['original_sha']:
        raise RuntimeError('Original frozen branch changed')
    print('SESSION SAVED:',context['run'],flush=True)
    return context['run']


def export_model(context):
    print('Export 1/4: compose original and independently selected specialist weights',flush=True)
    model=CompositeSwin(context['parent'],context['protected_parent']).cpu().eval()
    thresholds=list(context['metadata']['segmentation_thresholds'])
    selection={}
    post={'stone':context['metadata'].get('mask_postprocess',{}).get('stone',{})}
    for channel,name in ((1,'tumor'),(2,'cyst')):
        _,path=_paths(context,name)
        if not path.is_file():
            raise FileNotFoundError('Train/initialize both specialists before export: '+str(path))
        best=torch.load(path,map_location='cpu',weights_only=True)
        if best['signature']!=context['signature']:raise ValueError('Checkpoint signature mismatch')
        getattr(model,name).load_state_dict(best['model'],strict=True)
        thresholds[channel]=float(best['threshold'])
        post[name]={'min_pixels':int(best.get('min_pixels',0))}
        selection[name]={'round':int(best['round']),'development_metrics':best['metrics']}
    if tensor_hash(model.original.state_dict())!=context['original_sha']:
        raise RuntimeError('Original branch checksum mismatch')
    metadata=dict(context['metadata'])
    metadata.update(format=FORMAT,model=model.state_dict(),segmentation_thresholds=thresholds,
        original_sha256=context['original_sha'],parent_sha256=file_hash(context['source_path']),
        protected_source_sha256=file_hash(context['protected_source_path']),
        specialist_selection=selection,research_only=True,goal_80_verified=False)
    # New specialists use their validation-calibrated thresholds without inherited component removal.
    metadata['mask_postprocess']=post
    out=context['run']/'export';out.mkdir(parents=True,exist_ok=True)
    print('Export 2/4: write combined model.pth',flush=True)
    atomic_save(metadata,out/'model.pth')
    print('Export 3/4: copy matching source modules',flush=True)
    import shutil
    for name in ('kv_independent_lesions','kv_protected_finetune','kv_multitask','kv_model','kv_infer','kv_joint_infer','kv_data'):
        module=__import__(name)
        shutil.copyfile(module.__file__,out/(name+'.py'))
    (out/'requirements.txt').write_text('torch>=2.3\nnumpy\npandas\nscipy\nPillow\nmatplotlib\nnibabel\npydicom\nrequests\n')
    (out/'README.txt').write_text(
        'Use kv_independent_lesions.load_composite("model.pth", device) and predict_image.\n'
        'Three complete Swin paths: original stone/classification, tumor specialist, cyst specialist.\n'
        'Training resumes from each class last.pth with the original shared source and experiment settings.\n'
        'Classes: normal, stone, tumor, cyst; independent masks: stone, tumor, cyst. Background is derived.\n'
        'Development metrics are sampled slices, not full-volume or external accuracy.\n'
        'An 80% target is not a guarantee. Run a fixed final held-out assessment before making a performance claim.\n')
    print('Export 4/4: strict reload and frozen branch verification',flush=True)
    check,_=load_composite(out/'model.pth','cpu')
    protection=audit_protection(check,context['protected_parent'],context['metadata']['tile_size'])
    (out/'protection_check.json').write_text(json.dumps(protection,indent=2))
    del check,model
    print('EXPORT COMPLETE:',out/'model.pth','| selected:',selection,flush=True)
    return out


@torch.inference_mode()
def audit_protection(composite,parent,tile):
    composite.eval();parent.eval()
    if tensor_hash(composite.original.state_dict())!=tensor_hash(parent.base.state_dict()):
        raise RuntimeError('Original weights changed')
    x=torch.rand((1,1,tile,tile),generator=torch.Generator().manual_seed(123))
    old=parent.base(x,task='both')
    new=composite(x,task='both')
    torch.testing.assert_close(old['classification_logits'],new['classification_logits'],rtol=0,atol=0)
    torch.testing.assert_close(old['segmentation_logits'][:,0],new['segmentation_logits'][:,0],rtol=0,atol=0)
    return {'frozen_weight_checksum_matches':True,'classification_logits_identical_on_cpu_probe':True,
            'stone_logits_identical_on_cpu_probe':True,
            'scope':'same saved source, preprocessing, and inference policy; not an accuracy score'}


def load_composite(path,device='cpu'):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    if obj.get('format')!=FORMAT:raise ValueError('Use the matching independent-Swin export')
    from kv_multitask import SwinMultiTask
    from kv_protected_finetune import ProtectedSwin
    parent=ProtectedSwin(SwinMultiTask(obj['encoder_config']))
    net=CompositeSwin(parent)
    net.load_state_dict(obj['model'],strict=True)
    net.segmentation_thresholds=obj['segmentation_thresholds']
    if tensor_hash(net.original.state_dict())!=obj['original_sha256']:
        raise ValueError('Frozen original branch was modified')
    return net.to(device).eval(),obj


@torch.inference_mode()
def predict_image(path,image_path,outdir,device='cpu'):
    from kv_joint_infer import read_2d,predict_classification
    from kv_infer import predict_slice,overlay,components_2d
    from PIL import Image
    net,meta=load_composite(path,device)
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    x,spacing,provenance=read_2d(image_path)
    probabilities=predict_slice(net,x,tile=meta['tile_size'],overlap=.25,device=device,infer_batch=2,tta=False)
    masks=np.stack([mask_rule(probabilities[i],meta['segmentation_thresholds'][i],
        int(meta.get('mask_postprocess',{}).get(name,{}).get('min_pixels',0))) for i,name in enumerate(('stone','tumor','cyst'))])
    classification=predict_classification(net,x,tile=meta['tile_size'],device=device)
    overlay(x,masks,outdir/'highlighted_ct.png')
    labels=np.zeros(x.shape,np.uint8)
    labels[masks[2]]=3;labels[masks[1]]=2;labels[masks[0]]=1
    Image.fromarray(labels).save(outdir/'segmentation_labels.png')
    np.savez_compressed(outdir/'independent_masks.npz',masks=masks,probabilities=probabilities)
    lesions=components_2d(masks,spacing=spacing,provenance=provenance or 'No verified spacing; pixels only')
    pd.DataFrame(lesions).to_csv(outdir/'lesions.csv',index=False)
    result={'classification':classification,'lesions':lesions,'overlay':str(outdir/'highlighted_ct.png'),
        'note':'Predicted regions; no-lesion background is derived. Millimetres require verified spacing.'}
    (outdir/'prediction.json').write_text(json.dumps(result,indent=2))
    return result


@torch.inference_mode()
def heldout_chunk(export_path,frame,output_dir,rows_per_call=100,split='test',device='cuda',cache=None):
    """Save after each native slice. Partial patient averages are explicitly provisional."""
    net,meta=load_composite(export_path,device)
    rows=frame[(frame.source=='kits')&(frame.split==split)].sort_values(['group','slice','id']).reset_index(drop=True)
    if rows.empty:raise ValueError('Requested split has no KiTS slices')
    signature=hashlib.sha256((file_hash(export_path)+split+'|'.join(rows.id.astype(str))).encode()).hexdigest()
    output=Path(output_dir)/signature[:12];output.mkdir(parents=True,exist_ok=True)
    state_path=output/'progress.json'
    state=json.loads(state_path.read_text()) if state_path.is_file() else {'signature':signature,'next_row':0,
        'totals':[[0.,0.,0.],[0.,0.,0.]],'patients':{}}
    if state['signature']!=signature:raise ValueError('Evaluation inputs changed')
    start=int(state['next_row']);stop=min(len(rows),start+int(rows_per_call))
    for index in range(start,stop):
        row=rows.iloc[index]
        # Cache only lesion-positive slices; every negative row is still evaluated from Drive.
        path=cache.get(row) if cache is not None and (int(row.has_tumor) or int(row.has_cyst)) else Path(row.path)
        with np.load(path,allow_pickle=False) as d:
            image,truth=d['x'].astype(np.float32),d['y'][1:3]>0
        patient=state['patients'].setdefault(str(row.group),[[0.,0.,0.],[0.,0.,0.]])
        for i,name in enumerate(('tumor','cyst')):
            probability=predict_probability(getattr(net,name),image,meta['tile_size'],device,2)
            mask=mask_rule(probability,meta['segmentation_thresholds'][i+1],
                           int(meta.get('mask_postprocess',{}).get(name,{}).get('min_pixels',0)))
            counts=[int((mask&truth[i]).sum()),int(mask.sum()),int(truth[i].sum())]
            state['totals'][i]=[a+b for a,b in zip(state['totals'][i],counts)]
            patient[i]=[a+b for a,b in zip(patient[i],counts)]
        state['next_row']=index+1
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state));os.replace(temp,state_path)
        if (index+1)%10==0:print(f'{split}: {index+1}/{len(rows)} slices evaluated',flush=True)
    metrics={}
    for i,name in enumerate(('tumor','cyst')):
        tp,pred,actual=state['totals'][i]
        per_patient=[2*data[i][0]/max(data[i][1]+data[i][2],1) for data in state['patients'].values() if data[i][2]>0]
        metrics[name]={'dice':2*tp/max(pred+actual,1),'positive_patient_mean_dice':float(np.mean(per_patient)) if per_patient else 0.,
            'precision':tp/max(pred,1),'recall':tp/max(actual,1),
            'actual_pixels':int(actual),'predicted_pixels':int(pred)}
    complete=int(state['next_row'])==len(rows)
    result={'complete':complete,'completed_slices':state['next_row'],'total_slices':len(rows),
        'metrics':metrics,'split':split,'meets_numeric_80_target':bool(complete and all(
            m['dice']>=.8 and m['positive_patient_mean_dice']>=.8 for m in metrics.values())),
        'independent_performance_verified':False,
        'note':'The threshold goal describes this evaluated split only; prior test-driven tuning limits independence.'}
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    return result
