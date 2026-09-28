from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import UniversalDynamicsDecoder,Cross
class TargetOnlyDecoder(nn.Module):
    def __init__(self,input_dim=1280,hidden_dim=256,num_heads=8,ffn_dim=1024,dropout=.1):
        super().__init__();self.decoder=UniversalDynamicsDecoder(input_dim,hidden_dim,num_heads,ffn_dim,num_slots=1,dropout=dropout);self.read=Cross(hidden_dim,num_heads,ffn_dim,dropout);self.tq=nn.Parameter(torch.randn(1,16,hidden_dim)*.02);self.time=Cross(hidden_dim,num_heads,ffn_dim,dropout);self.norm=nn.LayerNorm(hidden_dim);self.state=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,7));self.presence=nn.Linear(hidden_dim,1);self.ocp=nn.Linear(hidden_dim,1)
    def forward(self,context,future):
        z=self.decoder(context,future)[:,:,0,:];obj=self.norm(self.read(torch.zeros(z.size(0),1,z.size(-1),device=z.device,dtype=z.dtype),z));h=self.norm(self.time((obj+self.tq).expand(-1,-1,-1),z));return {'trajectory_2d':self.state(h),'presence_logits':self.presence(obj).squeeze(-1).squeeze(-1),'ocp_logits':self.ocp(obj).squeeze(-1).squeeze(-1)}
def target_loss(o,b):
    valid=b['state_valid'].bool();p=o['trajectory_2d'];t=b['state_2d'].to(p);zero=p.sum()*0.;geom=F.smooth_l1_loss(p[valid],t[valid]) if valid.any() else zero;presence=F.binary_cross_entropy_with_logits(o['presence_logits'],b['object_present'].float());ov=b['ocp_valid'].bool();ocp=F.binary_cross_entropy_with_logits(o['ocp_logits'][ov],b['ocp_label'].to(o['ocp_logits'])[ov]) if ov.any() else zero;return presence+2*geom+ocp,{'total':presence+2*geom+ocp,'presence':presence,'geometry':geom,'ocp':ocp}
