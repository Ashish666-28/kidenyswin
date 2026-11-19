"""Tensor-compatible Swin reconstruction; original forward-code equivalence is unverified."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

CLASSES = ['stone', 'tumor', 'cyst']

class DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__(); self.p=p
    def forward(self,x):
        if not self.training or self.p == 0: return x
        keep=1-self.p
        return x * torch.empty((x.shape[0],)+(1,)*(x.ndim-1),device=x.device,dtype=x.dtype).bernoulli_(keep)/keep

def partition(x,w):
    b,h,k,c=x.shape
    return x.reshape(b,h//w,w,k//w,w,c).permute(0,1,3,2,4,5).reshape(-1,w*w,c)

def unpartition(x,w,h,k,b):
    return x.reshape(b,h//w,k//w,w,w,-1).permute(0,1,3,2,4,5).reshape(b,h,k,-1)

class WindowAttention(nn.Module):
    def __init__(self,dim,heads,window,drop):
        super().__init__(); self.heads=heads; self.window=window; self.scale=(dim//heads)**-.5
        self.relative_bias_table=nn.Parameter(torch.zeros((2*window-1)**2,heads))
        self.qkv=nn.Linear(dim,dim*3); self.proj=nn.Linear(dim,dim)
        self.attn_drop=nn.Dropout(drop); self.proj_drop=nn.Dropout(drop)
        coords=torch.stack(torch.meshgrid(torch.arange(window),torch.arange(window),indexing='ij')).flatten(1)
        rel=coords[:,:,None]-coords[:,None,:]; rel=rel.permute(1,2,0).contiguous()+window-1
        idx=rel[:,:,0]*(2*window-1)+rel[:,:,1]
        self.register_buffer('relative_index',idx,persistent=False)
        nn.init.trunc_normal_(self.relative_bias_table,std=.02)
    def forward(self,x,mask=None):
        b,n,c=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.heads,c//self.heads).permute(2,0,3,1,4).unbind(0)
        # FP32 attention logits/softmax avoids FP16 overflow on a T4.
        attn=(q.float()*self.scale) @ k.float().transpose(-2,-1)
        attn=attn+self.relative_bias_table[self.relative_index.flatten()].reshape(n,n,-1).permute(2,0,1).float().unsqueeze(0)
        if mask is not None:
            nw=mask.shape[0]
            attn=attn.reshape(b//nw,nw,self.heads,n,n)+mask[None,:,None].float()
            attn=attn.reshape(b,self.heads,n,n)
        out=(self.attn_drop(attn.softmax(-1)).to(v.dtype) @ v).transpose(1,2).reshape(b,n,c)
        return self.proj_drop(self.proj(out))

class SwinBlock(nn.Module):
    def __init__(self,dim,heads,window,shift,drop,path):
        super().__init__(); self.window=window; self.shift=shift
        self.norm1=nn.LayerNorm(dim); self.norm2=nn.LayerNorm(dim)
        self.attention=WindowAttention(dim,heads,window,drop)
        self.mlp=nn.Sequential(nn.Linear(dim,4*dim),nn.GELU(),nn.Dropout(drop),nn.Linear(4*dim,dim),nn.Dropout(drop))
        self.drop_path=DropPath(path)
        self._mask_cache={}
    def forward(self,x):
        b,h,k,c=x.shape; w=self.window; s=self.shift if min(h,k)>w else 0
        hp=math.ceil(h/w)*w; kp=math.ceil(k/w)*w
        shortcut=x; y=F.pad(self.norm1(x),(0,0,0,kp-k,0,hp-h))
        key=(h,k,str(x.device),s)
        mask=self._mask_cache.get(key)
        if mask is None:
            region=torch.zeros((1,hp,kp,1),device=x.device)
            if s:
                n=0
                for a in (slice(0,-w),slice(-w,-s),slice(-s,None)):
                    for z in (slice(0,-w),slice(-w,-s),slice(-s,None)):
                        region[:,a,z,:]=n; n+=1
            win=partition(region,w).squeeze(-1)
            mask=(win[:,:,None]!=win[:,None,:]).float()*-100.
            valid=torch.zeros((1,hp,kp,1),device=x.device); valid[:,:h,:k]=1
            if s: valid=torch.roll(valid,(-s,-s),(1,2))
            valid=partition(valid,w).squeeze(-1)
            mask=mask+(1-valid[:,None,:])*-100.
            if len(self._mask_cache)>3: self._mask_cache.clear()
            self._mask_cache[key]=mask
        if s: y=torch.roll(y,(-s,-s),(1,2))
        y=unpartition(self.attention(partition(y,w),mask),w,hp,kp,b)
        if s: y=torch.roll(y,(s,s),(1,2))
        x=shortcut+self.drop_path(y[:,:h,:k])
        return x+self.drop_path(self.mlp(self.norm2(x)))

class PatchMerge(nn.Module):
    def __init__(self,dim):
        super().__init__();self.norm=nn.LayerNorm(4*dim);self.reduction=nn.Linear(4*dim,2*dim,bias=False)
    def forward(self,x):
        _,h,w,_=x.shape; x=F.pad(x,(0,0,0,w%2,0,h%2))
        x=torch.cat((x[:,0::2,0::2],x[:,1::2,0::2],x[:,0::2,1::2],x[:,1::2,1::2]),-1)
        return self.reduction(self.norm(x))

class SwinEncoder(nn.Module):
    def __init__(self,config):
        super().__init__(); c=config; self.config=dict(c); self.gradient_checkpointing=True
        d=c['embed_dim']; self.patch_embed=nn.Conv2d(c['in_chans'],d,c['patch_size'],c['patch_size'])
        rates=torch.linspace(0,c.get('drop_path_rate',0.),sum(c['depths'])).tolist(); i=0
        self.stages=nn.ModuleList(); self.mergers=nn.ModuleList()
        for stage,depth in enumerate(c['depths']):
            blocks=[]
            for b in range(depth):
                blocks.append(SwinBlock(d*2**stage,c['num_heads'][stage],c['window_size'],0 if b%2==0 else c['window_size']//2,c.get('drop_rate',0.),rates[i]));i+=1
            self.stages.append(nn.ModuleList(blocks))
            if stage<3:self.mergers.append(PatchMerge(d*2**stage))
        self.norm=nn.LayerNorm(d*8)
    def forward(self,x):
        x=self.patch_embed(x).permute(0,2,3,1); out=[]
        for i,stage in enumerate(self.stages):
            for block in stage:
                x=checkpoint(block,x,use_reentrant=False) if self.training and self.gradient_checkpointing and torch.is_grad_enabled() else block(x)
            out.append((self.norm(x) if i==3 else x).permute(0,3,1,2).contiguous())
            if i<3:x=self.mergers[i](x)
        return out

def convblock(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(min(8,b),b),nn.GELU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(min(8,b),b),nn.GELU())

class SwinSegmenter(nn.Module):
    def __init__(self,config):
        super().__init__();self.encoder=SwinEncoder(config);d=config['embed_dim']
        self.lateral=nn.ModuleList([nn.Conv2d(d*2**i,64,1) for i in range(4)])
        self.fuse=nn.ModuleList([convblock(128,64) for _ in range(3)])
        self.stem=convblock(1,16);self.final=convblock(80,32);self.head=nn.Conv2d(32,3,1)
    def forward(self,x):
        feats=self.encoder(x);z=self.lateral[3](feats[3])
        for j,i in enumerate((2,1,0)):
            z=self.fuse[j](torch.cat((F.interpolate(z,size=feats[i].shape[-2:],mode='bilinear',align_corners=False),self.lateral[i](feats[i])),1))
        z=F.interpolate(z,size=x.shape[-2:],mode='bilinear',align_corners=False)
        return self.head(self.final(torch.cat((z,self.stem(x)),1)))

class UNet(nn.Module):
    def __init__(self):
        super().__init__();self.down=nn.ModuleList([convblock(1,16),convblock(16,32),convblock(32,64),convblock(64,128)])
        self.up=nn.ModuleList([convblock(192,64),convblock(96,32),convblock(48,16)]);self.head=nn.Conv2d(16,3,1)
    def forward(self,x):
        f=[]
        for i,m in enumerate(self.down):
            x=m(x);f.append(x)
            if i<3:x=F.max_pool2d(x,2)
        for m,z in zip(self.up,reversed(f[:-1])):
            x=m(torch.cat((F.interpolate(x,size=z.shape[-2:],mode='bilinear',align_corners=False),z),1))
        return self.head(x)

def load_classifier_encoder(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    assert isinstance(ck,dict) and 'model' in ck and 'config' in ck, 'Expected the supplied classifier checkpoint.'
    c=ck['config']; assert c['in_chans']==1 and len(c['depths'])==4
    model=SwinSegmenter(c);expected=model.encoder.state_dict();state=ck['model']
    missing=[k for k in expected if k not in state or state[k].shape!=expected[k].shape]
    extra=[k for k in state if k not in expected and not k.startswith('classifier.')]
    if missing or extra:raise ValueError(f'Encoder incompatible: missing/mismatched={missing}; unexpected={extra}')
    model.encoder.load_state_dict({k:state[k] for k in expected},strict=True)
    report={'loaded_tensors':len(expected),'loaded_parameters':sum(v.numel() for v in expected.values()),'discarded':['classifier.weight','classifier.bias'],'forward_equivalence':'unverified without original source','old_best_f1_not_segmentation':ck.get('best_f1'),'config':dict(c)}
    return model,report

def masked_loss(logits,target,known):
    # known [B,3]; padding pixels marked -1 in all channels.
    pix=(target>=0).float(); y=target.clamp_min(0); p=logits.float().sigmoid()
    bce=(F.binary_cross_entropy_with_logits(logits.float(),y,reduction='none')*pix).sum((2,3))/pix.sum((2,3)).clamp_min(1)
    inter=(p*y*pix).sum((2,3));den=((p+y)*pix).sum((2,3))
    dice=1-(2*inter+1)/(den+1)
    valid=known.float()*(pix.sum((2,3))>0)
    return ((bce+dice)*valid).sum()/valid.sum().clamp_min(1)

def small_lesion_masked_loss(logits,target,known,alpha=.30,beta=.70,gamma=1.33,max_pos_weight=8.,class_weights=None):
    """Positive-aware BCE + Dice + focal Tversky, respecting unknown classes and padded pixels."""
    z=logits.float();pix=(target>=0).float();y=target.clamp_min(0).float();p=z.sigmoid()
    spatial=(2,3);pos=(y*pix).sum(spatial);neg=((1-y)*pix).sum(spatial)
    pos_weight=torch.sqrt((neg+1)/(pos+1)).clamp(1,float(max_pos_weight))
    bce=-(pos_weight[:,:,None,None]*y*F.logsigmoid(z)+(1-y)*F.logsigmoid(-z))*pix
    bce=bce.sum(spatial)/pix.sum(spatial).clamp_min(1)
    tp=(p*y*pix).sum(spatial);fp=(p*(1-y)*pix).sum(spatial);fn=((1-p)*y*pix).sum(spatial)
    dice=1-(2*tp+1)/(2*tp+fp+fn+1)
    tversky=(tp+1)/(tp+float(alpha)*fp+float(beta)*fn+1)
    focal_tversky=(1-tversky).clamp_min(0).pow(float(gamma))
    valid=known.float()*(pix.sum(spatial)>0)
    per_class=.35*bce+.25*dice+.40*focal_tversky
    if class_weights is not None:
        weight=torch.as_tensor(class_weights,device=z.device,dtype=z.dtype).view(1,-1)
        if weight.shape[1]!=per_class.shape[1]:raise ValueError('SEG_CLASS_WEIGHTS must match mask classes')
        valid=valid*weight
    return (per_class*valid).sum()/valid.sum().clamp_min(1)
