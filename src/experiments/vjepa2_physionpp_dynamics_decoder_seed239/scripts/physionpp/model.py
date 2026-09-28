"""Physion++ structured decoder/probe (CLEVRER-style, without an internal OCP head)."""
from __future__ import annotations
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_OBJECTS, FUTURE_STEPS, STATE_DIM = 8, 16, 7

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_dim, dropout):
        super().__init__(); self.norm=nn.LayerNorm(dim); self.net=nn.Sequential(nn.Linear(dim,ffn_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(ffn_dim,dim),nn.Dropout(dropout))
    def forward(self,x): return x+self.net(self.norm(x))

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__(); self.query_norm=nn.LayerNorm(dim); self.memory_norm=nn.LayerNorm(dim); self.cross_attention=nn.MultiheadAttention(dim,num_heads,dropout=dropout,batch_first=True); self.ffn=FeedForward(dim,ffn_dim,dropout)
    def forward(self,q,m):
        m=self.memory_norm(m); a,_=self.cross_attention(self.query_norm(q),m,m,need_weights=False); return self.ffn(q+a)

# Compatibility alias: ``model_target_only`` / ``train_target_only`` /
# ``visualize_target_only`` / ``evaluate_ocp_accuracy`` were written against the
# physionpp2 model, where this block is called ``Cross``. The definition is
# identical (same constructor signature and ``forward(q, m)``), so the alias
# restores those imports without changing any computation.
Cross = CrossAttentionBlock


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim, dropout):
        super().__init__(); self.norm=nn.LayerNorm(dim); self.attention=nn.MultiheadAttention(dim,num_heads,dropout=dropout,batch_first=True); self.ffn=FeedForward(dim,ffn_dim,dropout)
    def forward(self,x):
        n=self.norm(x); a,_=self.attention(n,n,n,need_weights=False); return self.ffn(x+a)

class UniversalDynamicsDecoder(nn.Module):
    def __init__(self,input_dim=1280,hidden_dim=256,num_heads=8,ffn_dim=1024,num_slots=8,slot_depth=1,transition_depth=1,interaction_depth=1,dropout=.1):
        super().__init__(); self.input_dim=int(input_dim); self.hidden_dim=int(hidden_dim); self.num_slots=int(num_slots)
        self.projection=nn.Linear(input_dim,hidden_dim); self.spatial_embedding=nn.Parameter(torch.empty(1,1,256,hidden_dim)); self.temporal_embedding=nn.Parameter(torch.empty(1,16,1,hidden_dim)); self.source_embedding=nn.Parameter(torch.empty(1,2,1,1,hidden_dim)); self.slot_queries=nn.Parameter(torch.empty(1,1,num_slots,hidden_dim)); self.future_time_queries=nn.Parameter(torch.empty(1,8,1,hidden_dim));
        self.slot_blocks=nn.ModuleList(CrossAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(slot_depth)); self.temporal_blocks=nn.ModuleList(SelfAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(transition_depth)); self.temporal_memory_blocks=nn.ModuleList(CrossAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(transition_depth)); self.interaction_blocks=nn.ModuleList(SelfAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(interaction_depth)); self.interaction_memory_blocks=nn.ModuleList(CrossAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(interaction_depth)); self.output_norm=nn.LayerNorm(hidden_dim)
        for p in (self.spatial_embedding,self.temporal_embedding,self.source_embedding,self.slot_queries,self.future_time_queries): nn.init.trunc_normal_(p,std=.02)
    def forward(self,context,future):
        expected=(8,256,self.input_dim)
        if context.ndim!=4 or tuple(context.shape[1:])!=expected or tuple(future.shape)!=tuple(context.shape): raise ValueError(f"context/future must be [B,8,256,{self.input_dim}]")
        b=context.size(0); values=self.projection(torch.cat((context.to(self.projection.weight.dtype),future.to(self.projection.weight.dtype)),1)); source=torch.cat((self.source_embedding[:,:1].expand(-1,8,-1,-1,-1),self.source_embedding[:,1:].expand(-1,8,-1,-1,-1)),1).reshape(1,16,1,self.hidden_dim); memory=(values+self.spatial_embedding+self.temporal_embedding+source).reshape(b,4096,self.hidden_dim); slots=(self.slot_queries+self.future_time_queries).expand(b,-1,-1,-1).reshape(b,8*self.num_slots,self.hidden_dim)
        for block in self.slot_blocks: slots=block(slots,memory)
        slots=slots.reshape(b,8,self.num_slots,self.hidden_dim)
        for block,reread in zip(self.temporal_blocks,self.temporal_memory_blocks):
            y=block(slots.permute(0,2,1,3).reshape(b*self.num_slots,8,self.hidden_dim)).reshape(b,self.num_slots,8,self.hidden_dim).permute(0,2,1,3); slots=reread(y.reshape(b,8*self.num_slots,self.hidden_dim),memory).reshape(b,8,self.num_slots,self.hidden_dim)
        for block,reread in zip(self.interaction_blocks,self.interaction_memory_blocks): slots=reread(block(slots.reshape(b*8,self.num_slots,self.hidden_dim)).reshape(b,8*self.num_slots,self.hidden_dim),memory).reshape(b,8,self.num_slots,self.hidden_dim)
        return self.output_norm(slots)

class PhysionShallowProbe(nn.Module):
    def __init__(self,hidden_dim=256,num_heads=8,ffn_dim=1024,num_slots=8,num_object_types=32,dropout=.1):
        super().__init__(); self.hidden_dim=hidden_dim; self.num_slots=num_slots; self.num_object_types=int(num_object_types); self.object_queries=nn.Parameter(torch.randn(1,num_slots,hidden_dim)*.02); self.object_readout=CrossAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout); self.time_queries=nn.Parameter(torch.randn(1,FUTURE_STEPS,hidden_dim)*.02); self.time_readout=CrossAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout); self.object_norm=nn.LayerNorm(hidden_dim); self.time_norm=nn.LayerNorm(hidden_dim); self.presence_head=nn.Linear(hidden_dim,1); self.target_head=nn.Linear(hidden_dim,1); self.object_type_head=nn.Linear(hidden_dim,self.num_object_types); self.color_head=nn.Linear(hidden_dim,3); self.state_head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,STATE_DIM)); pairs=torch.tensor(list(itertools.combinations(range(num_slots),2))); self.register_buffer('pair_indices',pairs); self.pair_time_encoder=nn.Sequential(nn.Linear(2*hidden_dim+2,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim)); self.pair_encoder=nn.Sequential(nn.Linear(3*hidden_dim,hidden_dim),nn.GELU(),nn.LayerNorm(hidden_dim)); self.contact_head=nn.Linear(hidden_dim,1); self.first_contact_head=nn.Linear(hidden_dim,FUTURE_STEPS+1)
    def forward(self,z_dyn):
        if z_dyn.ndim!=4 or z_dyn.shape[1:]!=(8,self.num_slots,self.hidden_dim): raise ValueError(f"z_dyn must be [B,8,{self.num_slots},{self.hidden_dim}]")
        b=z_dyn.size(0); memory=z_dyn.reshape(b,8*self.num_slots,self.hidden_dim); obj=self.object_norm(self.object_readout(self.object_queries.expand(b,-1,-1),memory)); tq=(obj[:,:,None,:]+self.time_queries[:,None,:,:]).reshape(b,self.num_slots*FUTURE_STEPS,self.hidden_dim); tf=self.time_norm(self.time_readout(tq,memory)).reshape(b,self.num_slots,FUTURE_STEPS,self.hidden_dim); state=self.state_head(tf); i,j=self.pair_indices[:,0],self.pair_indices[:,1]; rel=state[:,i,:,:2]-state[:,j,:,:2]; pt=self.pair_time_encoder(torch.cat((tf[:,i]+tf[:,j],(tf[:,i]-tf[:,j]).abs(),rel),-1)); pair=self.pair_encoder(torch.cat((obj[:,i]+obj[:,j],(obj[:,i]-obj[:,j]).abs(),pt.mean(2)),-1)); return {'z_dyn':z_dyn,'object_tokens':obj,'pair_tokens':pair,'presence_logits':self.presence_head(obj).squeeze(-1),'target_logits':self.target_head(obj).squeeze(-1),'object_type_logits':self.object_type_head(obj),'color_pred':self.color_head(obj),'trajectory_2d':state,'pair_distance_2d':torch.linalg.vector_norm(rel.float(),dim=-1),'contact_logits':self.contact_head(pt).squeeze(-1),'first_contact_logits':self.first_contact_head(pair)}

class PhysionDecoder(nn.Module):
    def __init__(self,**config):
        super().__init__(); decoder_keys={'input_dim','hidden_dim','num_heads','ffn_dim','num_slots','slot_depth','transition_depth','interaction_depth','dropout'}; self.decoder=UniversalDynamicsDecoder(**{k:v for k,v in config.items() if k in decoder_keys}); self.probe=PhysionShallowProbe(hidden_dim=self.decoder.hidden_dim,num_heads=config.get('num_heads',8),ffn_dim=config.get('ffn_dim',1024),num_slots=config.get('num_slots',8),num_object_types=config.get('num_object_types',32),dropout=config.get('dropout',.1))
    def forward(self,context,future): return self.probe(self.decoder(context,future))

# Compatibility alias for callers using the old class name.
PhysionProbe = PhysionShallowProbe

@torch.no_grad()
def match_objects(outputs,batch):
    assignments=[]
    for b in range(outputs['presence_logits'].size(0)):
        present=batch['object_present'][b].bool(); n=int(present.sum()); assignment=torch.full((MAX_OBJECTS,),-1,dtype=torch.long,device=outputs['presence_logits'].device)
        if n:
            cand=torch.tensor(list(itertools.permutations(range(MAX_OBJECTS),n)),device=assignment.device); target=torch.arange(n,device=assignment.device); pred=outputs['trajectory_2d'][b,:,:,:5]; true=batch['state_2d'][b,present,:,:5].to(pred); valid=batch['state_valid'][b,present].to(pred.dtype); diff=(pred[:,None]-true[None]).abs(); geom=(diff*pred.new_tensor([2.,2.,1.,1.,.5])*valid[None,:,:,None]).sum((2,3))/(valid.sum(1)[None]*6.5).clamp_min(1); cost=geom-.25*outputs['presence_logits'][b].sigmoid()[:,None]; assignment[cand[cost[cand,target].sum(1).argmin()]]=target
        assignments.append(assignment)
    return torch.stack(assignments)

def loss(outputs,batch):
    assignment=match_objects(outputs,batch); matched=assignment.ge(0); zero=outputs['trajectory_2d'].sum()*0.; losses={'presence':F.binary_cross_entropy_with_logits(outputs['presence_logits'],matched.float())}; rows=matched.nonzero()
    if not len(rows): raise ValueError('batch has no objects')
    b,p=rows.T; t=assignment[b,p]; pred=outputs['trajectory_2d'][b,p]; true=batch['state_2d'][b,t].to(pred); valid=batch['state_valid'][b,t].bool();
    losses['center']=F.smooth_l1_loss(pred[...,:2][valid],true[...,:2][valid]) if valid.any() else zero; losses['geometry']=F.smooth_l1_loss(pred[...,2:5][valid],true[...,2:5][valid]) if valid.any() else zero; losses['velocity']=F.smooth_l1_loss(pred[...,5:7][valid],true[...,5:7][valid]) if valid.any() else zero
    losses['target']=F.binary_cross_entropy_with_logits(outputs['target_logits'][b,p],batch['is_target'][b,t].float()) if 'is_target' in batch else zero; losses['object_type']=F.cross_entropy(outputs['object_type_logits'][b,p],batch['object_type'][b,t].long()) if 'object_type' in batch else zero; losses['color']=F.smooth_l1_loss(outputs['color_pred'][b,p],batch['color_rgb'][b,t].to(pred)) if 'color_rgb' in batch else zero
    ij=torch.tensor(list(itertools.combinations(range(MAX_OBJECTS),2)),device=matched.device); i,j=ij[:,0],ij[:,1]; pairs=(matched[:,i]&matched[:,j]).nonzero(); losses.update(distance=zero,contact=zero,first_contact=zero)
    if len(pairs):
        pb,pi=pairs.T; ti,tj=assignment[pb,i[pi]],assignment[pb,j[pi]]; pv=batch['pair_valid'][pb,ti,tj].bool(); losses['distance']=F.smooth_l1_loss(outputs['pair_distance_2d'][pb,pi][pv],batch['pair_distance_2d'][pb,ti,tj].to(pred)[pv]) if pv.any() else zero; cv=batch['contact_valid'][pb,ti,tj].bool(); losses['contact']=F.binary_cross_entropy_with_logits(outputs['contact_logits'][pb,pi][cv],batch['contact'][pb,ti,tj].to(pred)[cv]) if cv.any() else zero; losses['first_contact']=F.cross_entropy(outputs['first_contact_logits'][pb,pi],batch['first_contact_class'][pb,ti,tj].long())
    losses['total']=losses['presence']+losses['target']+losses['object_type']+losses['color']+2*losses['center']+losses['geometry']+.5*losses['velocity']+losses['distance']+2*losses['contact']+losses['first_contact']; return losses['total'],losses
