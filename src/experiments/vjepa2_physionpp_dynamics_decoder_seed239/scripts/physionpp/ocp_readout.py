"""Independent OCP readout: visual tokens fused with frozen structured-probe outputs."""
from __future__ import annotations
import torch
import torch.nn as nn

class OCPReadout(nn.Module):
    def __init__(self, visual_dim=1280, probe_dim=256, hidden_dim=256, num_heads=8, depth=2, dropout=.1):
        super().__init__()
        self.visual_proj=nn.Linear(visual_dim,hidden_dim); self.probe_proj=nn.Linear(probe_dim,hidden_dim)
        layer=nn.TransformerEncoderLayer(hidden_dim,num_heads,4*hidden_dim,dropout,batch_first=True,norm_first=True)
        self.fusion=nn.TransformerEncoder(layer,depth); self.norm=nn.LayerNorm(hidden_dim); self.head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,1))
    def forward(self, visual_tokens, probe_outputs):
        # Latent cache is stored as float16; readout weights are trained in float32.
        visual_tokens = visual_tokens.float()
        # visual_tokens is the raw concatenation of context and predicted-future
        # latents: [B,16,256,1280]. No decoder/pooling is applied here.
        if visual_tokens.ndim==4: visual_tokens=visual_tokens.reshape(visual_tokens.size(0),-1,visual_tokens.size(-1))
        required=('object_tokens','pair_tokens','trajectory_2d','contact_logits','first_contact_logits')
        missing=[k for k in required if k not in probe_outputs]
        if missing: raise KeyError(f'missing probe outputs: {missing}')
        b=visual_tokens.size(0); chunks=[probe_outputs['object_tokens'],probe_outputs['pair_tokens'],probe_outputs['trajectory_2d'].reshape(b,-1,probe_outputs['trajectory_2d'].size(-1)),probe_outputs['contact_logits'].unsqueeze(-1),probe_outputs['first_contact_logits']]
        normalized=[]
        for x in chunks:
            x=x if x.ndim==3 else x.reshape(b,-1,x.shape[-1])
            if x.size(-1)<self.probe_proj.in_features: x=torch.nn.functional.pad(x,(0,self.probe_proj.in_features-x.size(-1)))
            elif x.size(-1)>self.probe_proj.in_features: x=x[...,:self.probe_proj.in_features]
            normalized.append(x)
        probe=torch.cat(normalized,1).float()
        tokens=torch.cat([self.visual_proj(visual_tokens),self.probe_proj(probe)],1); pooled=self.norm(self.fusion(tokens)).mean(1); return self.head(pooled).squeeze(-1)

class VisualOnlyOCPReadout(nn.Module):
    """OCP baseline using only raw context + predicted-future visual latents."""
    def __init__(self, visual_dim=1280, hidden_dim=256, num_heads=8, depth=2, dropout=.1):
        super().__init__(); self.visual_proj=nn.Linear(visual_dim,hidden_dim); layer=nn.TransformerEncoderLayer(hidden_dim,num_heads,4*hidden_dim,dropout,batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,depth); self.norm=nn.LayerNorm(hidden_dim); self.head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,1))
    def forward(self, visual_tokens):
        if visual_tokens.ndim==4: visual_tokens=visual_tokens.reshape(visual_tokens.size(0),-1,visual_tokens.size(-1))
        if visual_tokens.ndim!=3 or visual_tokens.size(-1)!=self.visual_proj.in_features: raise ValueError(f'visual_tokens must be [B,N,{self.visual_proj.in_features}]')
        x=self.encoder(self.visual_proj(visual_tokens.float())); return self.head(self.norm(x).mean(1)).squeeze(-1)
