"""Swin multi-task: one shared encoder; image classification + partially supervised lesion masks."""
import torch
from torch import nn
import torch.nn.functional as F
from kv_model import SwinSegmenter, masked_loss

DISEASES=('normal','stone','tumor','cyst')
MASK_CLASSES=('stone','tumor','cyst')

class SwinMultiTask(SwinSegmenter):
    """ONE trainable nn.Module: ONE Swin encoder shared by TWO output heads.

    The `both` mode computes the Swin encoder ONCE and yields both predictions.
    Legacy task modes remain compatible with segmentation tiling and joint training.
    """
    def __init__(self,config):
        super().__init__(config)
        d=int(config['embed_dim'])*8
        self.cls_head=nn.Sequential(nn.LayerNorm(d),nn.Dropout(0.25),nn.Linear(d,4))

    def _seg_from_features(self,x,feats):
        z=self.lateral[3](feats[3])
        for j,i in enumerate((2,1,0)):
            z=self.fuse[j](torch.cat((F.interpolate(z,size=feats[i].shape[-2:],mode='bilinear',align_corners=False),self.lateral[i](feats[i])),1))
        z=F.interpolate(z,size=x.shape[-2:],mode='bilinear',align_corners=False)
        return self.head(self.final(torch.cat((z,self.stem(x)),1)))

    def forward(self,x,task='seg'):
        if task not in ('seg','cls','both'):
            raise ValueError('task must be seg, cls or both')
        feats=self.encoder(x)
        cls=self.cls_head(feats[-1].mean((2,3))) if task in ('cls','both') else None
        seg=self._seg_from_features(x,feats) if task in ('seg','both') else None
        if task=='cls':return cls
        if task=='seg':return seg
        return {'classification_logits':cls,'segmentation_logits':seg}


    def forward_mixed(self,seg_x,cls_x):
        """Encode paired task batches together, then run only the required head per subset."""
        if seg_x.ndim!=4 or cls_x.ndim!=4 or seg_x.shape[1:]!=cls_x.shape[1:]:
            raise ValueError('Mixed task tensors must have matching [C,H,W] shapes')
        nseg=seg_x.shape[0]
        joined=torch.cat((seg_x,cls_x),dim=0)
        feats=self.encoder(joined)
        seg_feats=[f[:nseg] for f in feats]
        cls_feat=feats[-1][nseg:]
        return {'segmentation_logits':self._seg_from_features(seg_x,seg_feats),
                'classification_logits':self.cls_head(cls_feat.mean((2,3)))}

def load_initial_weights(model,path):
    """Recognizes the previous segmentation bundle/checkpoint or original classifier checkpoint."""
    obj=torch.load(str(path),map_location='cpu',weights_only=True)
    if 'model' not in obj: raise ValueError('Expected a checkpoint with a model state dict')
    sd=obj['model']; own=model.state_dict()
    if 'encoder_config' in obj and 'head.weight' in sd:
        if dict(obj['encoder_config'])!=dict(model.encoder.config):
            raise ValueError('Existing segmentation model architecture mismatch')
        missing,unexpected=model.load_state_dict(sd,strict=False)
        if set(missing)!=set(k for k in own if k.startswith('cls_head.')) or unexpected:
            raise ValueError(f'Segmentation transfer unexpectedly incomplete: {missing}, {unexpected}')
        return {'initialization':'old_segmentation_checkpoint','loaded':len(sd),'new_head':'classification','old_best_f1':None}
    if 'config' in obj and 'classifier.weight' in sd:
        cfg=dict(obj['config'])
        if cfg!=dict(model.encoder.config): raise ValueError('Original classifier model config incompatible')
        enc=model.encoder.state_dict()
        mis=[k for k in enc if k not in sd or enc[k].shape!=sd[k].shape]
        if mis: raise ValueError(f'Original classifier encoder missing tensors: {mis[:6]}')
        model.encoder.load_state_dict({k:sd[k] for k in enc},strict=True)
        # Only transfer original classifier head when the checkpoint is exactly four classes.
        w=sd['classifier.weight'];b=sd.get('classifier.bias')
        transferred=False
        if w.shape==model.cls_head[-1].weight.shape and b is not None and b.shape==model.cls_head[-1].bias.shape:
            # Class order in old checkpoint is not verified. DO NOT transfer it.
            transferred=False
        return {'initialization':'old_classifier_encoder','loaded':len(enc),'new_head':'segmentation_and_classification','old_best_f1':obj.get('best_f1'),'old_classifier_head_transfer':transferred}
    raise ValueError('Unsupported checkpoint. Supply previous segmentation bundle or original classifier checkpoint.')

def segmentation_loss(logits,y,known):
    """Unannotated classes get zero gradient; BCE + soft Dice, no assumed negatives."""
    return masked_loss(logits,y,known)

def classification_loss(logits,labels,weights=None):
    return F.cross_entropy(logits.float(),labels,weight=weights,label_smoothing=.04)


def classification_focal_loss(logits,labels,gamma=1.5):
    """Hard-example classification loss; class balance remains controlled by the sampler."""
    z=logits.float();ce=F.cross_entropy(z,labels,reduction='none',label_smoothing=.02)
    pt=z.softmax(1).gather(1,labels[:,None]).squeeze(1).clamp(1e-5,1-1e-5)
    return (((1-pt)**float(gamma))*ce).mean()
