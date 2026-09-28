from __future__ import annotations
from typing import NamedTuple
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

class SlotState(NamedTuple):
    slots: torch.Tensor          # [B,T,S,H]
    ownership: torch.Tensor      # [B,T,N,S]
    objectness: torch.Tensor     # [B,T,N]
    presence_logits: torch.Tensor# [B,T,S]
    state: torch.Tensor          # [B,T,S,9]
    reconstruction: torch.Tensor # [B,T,N,D]

class GroundedObjectSlots(nn.Module):
    """Context-only object slots with explicit state prediction at each frame."""
    def __init__(self, input_dim=1280, hidden_dim=256, slots=8, iterations=3):
        super().__init__(); self.h,self.s,self.it=hidden_dim,slots,iterations
        self.in_proj=nn.Linear(input_dim,hidden_dim);self.pos=nn.Parameter(torch.randn(1,1,256,hidden_dim)*.02)
        self.slot_init=nn.Parameter(torch.randn(1,slots,hidden_dim)*.02);self.xnorm=nn.LayerNorm(hidden_dim);self.snorm=nn.LayerNorm(hidden_dim)
        self.key=nn.Linear(hidden_dim,hidden_dim,bias=False);self.value=nn.Linear(hidden_dim,hidden_dim,bias=False);self.query=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.gru=nn.GRUCell(hidden_dim,hidden_dim);self.ffn=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,4*hidden_dim),nn.GELU(),nn.Linear(4*hidden_dim,hidden_dim))
        self.objectness=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,1));self.presence=nn.Linear(hidden_dim,1)
        self.state_head=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,9))
        self.decoder=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,input_dim))
    def forward(self,x):
        b,t,n,d=x.shape;z=self.in_proj(x.float());z=z+(self.pos[:,:,:n] if n<=256 else 0);z=z.reshape(b*t,n,self.h);zn=self.xnorm(z);k,v=self.key(zn),self.value(zn);gate=self.objectness(zn).squeeze(-1).sigmoid();slots=self.slot_init.expand(b*t,-1,-1).contiguous()
        for _ in range(self.it):
            a=torch.einsum('bsh,bnh->bns',self.query(self.snorm(slots)),k)/self.h**.5;a=a.softmax(-1);w=a*gate.unsqueeze(-1);u=torch.einsum('bns,bnh->bsh',w,v)/w.sum(1).unsqueeze(-1).clamp_min(1e-6);slots=self.gru(u.flatten(0,1),slots.flatten(0,1)).reshape(b*t,self.s,self.h);slots=slots+self.ffn(slots)
        recon=torch.einsum('bns,bsd->bnd',a,self.decoder(slots));shape=lambda q:q.reshape(b,t,*q.shape[1:])
        return SlotState(shape(slots),shape(a),shape(gate),shape(self.presence(slots).squeeze(-1)),shape(self.state_head(slots)),shape(recon))

def hungarian(pred,target,valid,scale):
    ids=valid.nonzero(as_tuple=False).flatten()
    if not len(ids):return [],torch.empty(0,dtype=torch.long,device=pred.device)
    cost=((pred.detach()[:,None]-target[ids][None]).abs()/scale).mean(-1).cpu().numpy();r,c=linear_sum_assignment(cost)
    return list(zip(r.tolist(),ids[c].tolist())),ids

def object_state_loss(out,target,valid,scale):
    """Hungarian matched loss; target describes the final observed context state."""
    pred,logit=out.state[:,-1],out.presence_logits[:,-1];state=pred.sum()*0;presence=logit.sum()*0;count=0
    for b in range(pred.size(0)):
        pairs,_=hungarian(pred[b],target[b].to(pred),valid[b],scale)
        if not pairs:continue
        pi=torch.tensor([i for i,_ in pairs],device=pred.device);ti=torch.tensor([j for _,j in pairs],device=pred.device)
        # Normalized factor-wise state regression prevents large dimensions
        # from dominating position/velocity/extent learning.
        state=state+F.smooth_l1_loss(pred[b,pi]/scale,target[b,ti].to(pred)/scale)
        labels=torch.zeros_like(logit[b]);labels[pi]=1;presence=presence+F.binary_cross_entropy_with_logits(logit[b],labels);count+=1
    if not count:return {'state':state,'presence':presence,'total':state+presence}
    return {'state':state/count,'presence':presence/count,'total':state/count+presence/count}

def self_supervised_loss(out,x):
    recon=F.smooth_l1_loss(out.reconstruction,x.float())*.25
    p=out.ownership.clamp_min(1e-8);entropy=-(p*p.log()).sum(-1).mean()*1e-3
    usage=out.ownership.mean(2);balance=F.mse_loss(usage,usage.new_full(usage.shape,1/usage.size(-1)))*.01
    tracking=(1-F.cosine_similarity(out.slots[:,1:],out.slots[:,:-1],dim=-1).mean())*.05 if out.slots.size(1)>1 else recon*0
    return recon+entropy+balance+tracking
