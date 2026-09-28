"""Universal dynamics decoder with a Physion++ 2D structured probe."""
from __future__ import annotations

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_OBJECTS, FUTURE_STEPS, STATE_DIM = 8, 16, 7

class FFN(nn.Module):
    def __init__(self, dim, ffn, dropout):
        super().__init__(); self.norm=nn.LayerNorm(dim); self.net=nn.Sequential(nn.Linear(dim,ffn),nn.GELU(),nn.Dropout(dropout),nn.Linear(ffn,dim),nn.Dropout(dropout))
    def forward(self,x): return x+self.net(self.norm(x))

class Cross(nn.Module):
    def __init__(self, dim, heads, ffn, dropout):
        super().__init__(); self.q=nn.LayerNorm(dim); self.m=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True); self.ffn=FFN(dim,ffn,dropout)
    def forward(self,q,m):
        a,_=self.attn(self.q(q),self.m(m),self.m(m),need_weights=False); return self.ffn(q+a)

class Self(nn.Module):
    def __init__(self, dim, heads, ffn, dropout):
        super().__init__(); self.n=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True); self.ffn=FFN(dim,ffn,dropout)
    def forward(self,x):
        a,_=self.attn(self.n(x),self.n(x),self.n(x),need_weights=False); return self.ffn(x+a)

class UniversalDynamicsDecoder(nn.Module):
    def __init__(self,input_dim=1280,hidden_dim=256,num_heads=8,ffn_dim=1024,num_slots=8,slot_depth=1,transition_depth=1,interaction_depth=1,dropout=.1):
        super().__init__(); self.input_dim=int(input_dim); self.hidden_dim=int(hidden_dim); self.num_slots=int(num_slots)
        self.proj=nn.Linear(input_dim,hidden_dim); self.spatial=nn.Parameter(torch.empty(1,1,256,hidden_dim)); self.temporal=nn.Parameter(torch.empty(1,16,1,hidden_dim)); self.source=nn.Parameter(torch.empty(1,2,1,1,hidden_dim)); self.slots=nn.Parameter(torch.empty(1,1,num_slots,hidden_dim)); self.times=nn.Parameter(torch.empty(1,8,1,hidden_dim))
        self.slot_blocks=nn.ModuleList(Cross(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(slot_depth)); self.temporal_blocks=nn.ModuleList(Self(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(transition_depth)); self.temporal_reread=nn.ModuleList(Cross(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(transition_depth)); self.interaction=nn.ModuleList(Self(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(interaction_depth)); self.interaction_reread=nn.ModuleList(Cross(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(interaction_depth));
        for p in (self.spatial,self.temporal,self.source,self.slots,self.times): nn.init.trunc_normal_(p,std=.02)
    def forward(self,context,future):
        if context.ndim!=4 or tuple(context.shape[1:])!=(8,256,self.input_dim) or tuple(future.shape)!=tuple(context.shape): raise ValueError(f"expected context/future [B,8,256,{self.input_dim}]")
        context=context.to(self.proj.weight.dtype); future=future.to(self.proj.weight.dtype); b=context.size(0); v=self.proj(torch.cat((context,future),1)); src=torch.cat((self.source[:,:1].expand(-1,8,-1,-1,-1),self.source[:,1:].expand(-1,8,-1,-1,-1)),1).reshape(1,16,1,self.hidden_dim); mem=(v+self.spatial+self.temporal+src).reshape(b,4096,self.hidden_dim); x=(self.slots+self.times).expand(b,-1,-1,-1).reshape(b,8*self.num_slots,self.hidden_dim)
        for block in self.slot_blocks: x=block(x,mem)
        x=x.reshape(b,8,self.num_slots,self.hidden_dim)
        for block,reread in zip(self.temporal_blocks,self.temporal_reread):
            y=block(x.permute(0,2,1,3).reshape(b*self.num_slots,8,self.hidden_dim)).reshape(b,self.num_slots,8,self.hidden_dim).permute(0,2,1,3); x=reread(y.reshape(b,8*self.num_slots,self.hidden_dim),mem).reshape(b,8,self.num_slots,self.hidden_dim)
        for block,reread in zip(self.interaction,self.interaction_reread): x=reread(block(x.reshape(b*8,self.num_slots,self.hidden_dim)).reshape(b,8*self.num_slots,self.hidden_dim),mem).reshape(b,8,self.num_slots,self.hidden_dim)
        return x

class PhysionProbe(nn.Module):
    def __init__(self,hidden_dim=256,num_heads=8,ffn_dim=1024,num_slots=8,dropout=.1):
        super().__init__(); self.hidden_dim=hidden_dim; self.num_slots=num_slots; self.object_q=nn.Parameter(torch.randn(1,num_slots,hidden_dim)*.02); self.object_readout=Cross(hidden_dim,num_heads,ffn_dim,dropout); self.time_q=nn.Parameter(torch.randn(1,FUTURE_STEPS,hidden_dim)*.02); self.time_readout=Cross(hidden_dim,num_heads,ffn_dim,dropout); self.norm=nn.LayerNorm(hidden_dim); self.presence=nn.Linear(hidden_dim,1); self.state=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,STATE_DIM)); pairs=torch.tensor(list(itertools.combinations(range(num_slots),2))); self.register_buffer('pairs',pairs); self.pair_time=nn.Sequential(nn.Linear(2*hidden_dim+2,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim)); self.pair=nn.Sequential(nn.Linear(3*hidden_dim,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim)); self.contact=nn.Linear(hidden_dim,1); self.first=nn.Linear(hidden_dim,FUTURE_STEPS+1); self.ocp=nn.Sequential(nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))
    def forward(self,z):
        b=z.size(0); mem=z.reshape(b,64,self.hidden_dim); obj=self.norm(self.object_readout(self.object_q.expand(b,-1,-1),mem)); tq=(obj[:,:,None,:]+self.time_q[:,None,:,:]).reshape(b,self.num_slots*FUTURE_STEPS,self.hidden_dim); tf=self.norm(self.time_readout(tq,mem)).reshape(b,self.num_slots,FUTURE_STEPS,self.hidden_dim); state=self.state(tf); i,j=self.pairs[:,0],self.pairs[:,1]; rel=state[:,i,:,:2]-state[:,j,:,:2]; pt=self.pair_time(torch.cat((tf[:,i]+tf[:,j],(tf[:,i]-tf[:,j]).abs(),rel),-1)); pair=self.pair(torch.cat((obj[:,i]+obj[:,j],(obj[:,i]-obj[:,j]).abs(),pt.mean(2)),-1)); return {'z_dyn':z,'object_tokens':obj,'pair_tokens':pair,'presence_logits':self.presence(obj).squeeze(-1),'trajectory_2d':state,'pair_distance_2d':torch.linalg.vector_norm(rel.float(),dim=-1),'contact_logits':self.contact(pt).squeeze(-1),'first_contact_logits':self.first(pair),'ocp_logits':self.ocp(obj.mean(1)).squeeze(-1)}

class PhysionDecoder(nn.Module):
    def __init__(self,**cfg): super().__init__(); self.decoder=UniversalDynamicsDecoder(**cfg); self.probe=PhysionProbe(hidden_dim=self.decoder.hidden_dim,num_heads=cfg.get('num_heads',8),ffn_dim=cfg.get('ffn_dim',1024),num_slots=cfg.get('num_slots',8),dropout=cfg.get('dropout',.1))
    def forward(self,context,future): return self.probe(self.decoder(context,future))

@torch.no_grad()
def match_objects(outputs,batch):
    result=[]; pairs=list(itertools.permutations(range(MAX_OBJECTS),int(batch['object_present'][0].sum()))) if False else None
    for b in range(outputs['presence_logits'].size(0)):
        present=batch['object_present'][b].bool(); n=int(present.sum()); assignment=torch.full((MAX_OBJECTS,),-1,dtype=torch.long,device=outputs['presence_logits'].device)
        if n:
            cand=torch.tensor(list(itertools.permutations(range(MAX_OBJECTS),n)),device=assignment.device); target=torch.arange(n,device=assignment.device); pred=outputs['trajectory_2d'][b,:,:,:5]; true=batch['state_2d'][b,present,:,:5].to(pred); valid=batch['state_valid'][b,present].to(pred.dtype); diff=(pred[:,None]-true[None]).abs(); weights=pred.new_tensor([2.,2.,1.,1.,.5]); geom=(diff*weights*valid[None,:,:,None]).sum((2,3))/(valid.sum(1)[None]*weights.sum()).clamp_min(1); cost=geom-.25*outputs['presence_logits'][b].sigmoid()[:,None]; selected=cand[cost[cand,target].sum(1).argmin()]; assignment[selected]=target
        result.append(assignment)
    return torch.stack(result)

def loss(outputs,batch):
    if not all(torch.isfinite(v).all() for v in outputs.values() if torch.is_tensor(v)): raise FloatingPointError('non-finite probe output')
    assignment=match_objects(outputs,batch); matched=assignment>=0; losses={'presence':F.binary_cross_entropy_with_logits(outputs['presence_logits'],matched.float())}; rows=matched.nonzero();
    if not len(rows): raise ValueError('batch has no objects')
    b,p=rows.T; t=assignment[b,p]; pred=outputs['trajectory_2d'][b,p]; true=batch['state_2d'][b,t].to(pred); valid=batch['state_valid'][b,t].bool()
    # Empty valid masks otherwise make mean-reduced losses NaN.
    zero=pred.sum()*0.0
    losses['center']=F.smooth_l1_loss(pred[...,:2][valid],true[...,:2][valid]) if valid.any() else zero
    losses['geometry']=F.smooth_l1_loss(pred[...,2:5][valid],true[...,2:5][valid]) if valid.any() else zero
    losses['velocity']=F.smooth_l1_loss(pred[...,5:7][valid],true[...,5:7][valid]) if valid.any() else zero
    left,right=outputs['pair_tokens'].new_tensor(list(itertools.combinations(range(MAX_OBJECTS),2)),dtype=torch.long).T; pairs=(matched[:,left]&matched[:,right]).nonzero();
    if len(pairs):
        pb,pi=pairs.T; tf,ts=assignment[pb,left[pi]],assignment[pb,right[pi]]; pv=batch['pair_valid'][pb,tf,ts].bool(); losses['distance']=F.smooth_l1_loss(outputs['pair_distance_2d'][pb,pi][pv],batch['pair_distance_2d'][pb,tf,ts].to(outputs['pair_distance_2d'])[pv]) if pv.any() else losses['center']*0; cv=batch['contact_valid'][pb,tf,ts].bool(); ct=batch['contact'][pb,tf,ts].to(outputs['contact_logits']); losses['contact']=F.binary_cross_entropy_with_logits(outputs['contact_logits'][pb,pi][cv],ct[cv]) if cv.any() else losses['center']*0; first=batch['first_contact_class'][pb,tf,ts].long(); losses['first_contact']=F.cross_entropy(outputs['first_contact_logits'][pb,pi],first)
    else: losses.update(distance=losses['center']*0,contact=losses['center']*0,first_contact=losses['center']*0)
    ov=batch['ocp_valid'].bool(); losses['ocp']=F.binary_cross_entropy_with_logits(outputs['ocp_logits'][ov],batch['ocp_label'].to(outputs['ocp_logits'])[ov]) if ov.any() else losses['center']*0; losses['total']=losses['presence']+2*losses['center']+losses['geometry']+.5*losses['velocity']+losses['distance']+2*losses['contact']+losses['first_contact']+losses['ocp']; return losses['total'],losses
