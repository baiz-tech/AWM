"""Official VideoMAEv2 loader and direct verb/noun/action heads."""
from __future__ import annotations
from pathlib import Path
import sys, torch
from torch import nn

def build_encoder(cfg, device):
    repo=Path(cfg['official_repo']).resolve(); sys.path.insert(0,str(repo)); from models.modeling_finetune import vit_base_patch16_224,vit_small_patch16_224,vit_giant_patch14_224
    factory={'vit_b':vit_base_patch16_224,'vit_base':vit_base_patch16_224,'vit_s':vit_small_patch16_224,'vit_small':vit_small_patch16_224,'vit_g':vit_giant_patch14_224,'vit_giant':vit_giant_patch14_224}[str(cfg.get('architecture','vit_b')).lower()]
    m=factory(num_classes=1,all_frames=int(cfg.get('all_frames',16)),tubelet_size=int(cfg.get('tubelet_size',2)),use_mean_pooling=True); m.head=nn.Identity(); p=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False); state=p.get('module',p.get('model',p)); missing,unexpected=m.load_state_dict(state,strict=False); allowed={'norm.weight','norm.bias','head.weight','head.bias','fc_norm.weight','fc_norm.bias'}; bad=(set(missing)|set(unexpected))-allowed
    if bad: raise RuntimeError(f'checkpoint mismatch: missing={missing}, unexpected={unexpected}')
    if bool(cfg.get('freeze_encoder', True)):
        m.eval()
        for q in m.parameters(): q.requires_grad_(False)
    return m.to(device)

class DirectHeads(nn.Module):
    def __init__(self,dim,verbs,nouns,actions): super().__init__(); self.verb=nn.Linear(dim,verbs); self.noun=nn.Linear(dim,nouns); self.action=nn.Linear(dim,actions)
    def forward(self,x): return self.verb(x),self.noun(x),self.action(x)

class VideoMAEClassifier(nn.Module):
    """VideoMAEv2 encoder followed by direct verb/noun/action classifiers."""
    def __init__(self, encoder, verbs, nouns, actions):
        super().__init__(); self.encoder=encoder; self.verb=nn.Linear(encoder.embed_dim,verbs); self.noun=nn.Linear(encoder.embed_dim,nouns); self.action=nn.Linear(encoder.embed_dim,actions)
    def forward(self, video):
        z=self.encoder.patch_embed(video); z=z+self.encoder.pos_embed.to(device=z.device,dtype=z.dtype); z=self.encoder.pos_drop(z)
        for block in self.encoder.blocks: z=block(z)
        feat=self.encoder.fc_norm(z.mean(1)); return self.verb(feat),self.noun(feat),self.action(feat)
