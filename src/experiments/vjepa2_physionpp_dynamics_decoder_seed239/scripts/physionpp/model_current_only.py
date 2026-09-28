"""Current-only Physion++ predictor: context latent only, no future latent input."""
from __future__ import annotations
import torch
import torch.nn as nn
from .model import (FeedForward, CrossAttentionBlock, SelfAttentionBlock,
                    PhysionShallowProbe, MAX_OBJECTS)

class CurrentOnlyDynamicsDecoder(nn.Module):
    def __init__(self,input_dim=1280,hidden_dim=256,num_heads=8,ffn_dim=1024,num_slots=8,slot_depth=1,transition_depth=1,interaction_depth=1,dropout=.1):
        super().__init__(); self.input_dim=int(input_dim); self.hidden_dim=int(hidden_dim); self.num_slots=int(num_slots)
        self.projection=nn.Linear(input_dim,hidden_dim); self.spatial_embedding=nn.Parameter(torch.empty(1,1,256,hidden_dim)); self.temporal_embedding=nn.Parameter(torch.empty(1,8,1,hidden_dim)); self.slot_queries=nn.Parameter(torch.empty(1,1,num_slots,hidden_dim)); self.time_queries=nn.Parameter(torch.empty(1,8,1,hidden_dim)); self.slot_blocks=nn.ModuleList(CrossAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(slot_depth)); self.temporal_blocks=nn.ModuleList(SelfAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(transition_depth)); self.temporal_memory_blocks=nn.ModuleList(CrossAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(transition_depth)); self.interaction_blocks=nn.ModuleList(SelfAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(interaction_depth)); self.interaction_memory_blocks=nn.ModuleList(CrossAttentionBlock(hidden_dim,num_heads,ffn_dim,dropout) for _ in range(interaction_depth)); self.output_norm=nn.LayerNorm(hidden_dim)
        for p in (self.spatial_embedding,self.temporal_embedding,self.slot_queries,self.time_queries): nn.init.trunc_normal_(p,std=.02)
    def forward(self,context):
        if context.ndim!=4 or tuple(context.shape[1:])!=(8,256,self.input_dim): raise ValueError(f'context must be [B,8,256,{self.input_dim}]')
        context=context.to(self.projection.weight.dtype); b=context.size(0); memory=(self.projection(context)+self.spatial_embedding+self.temporal_embedding).reshape(b,2048,self.hidden_dim); slots=(self.slot_queries+self.time_queries).expand(b,-1,-1,-1).reshape(b,8*self.num_slots,self.hidden_dim)
        for block in self.slot_blocks: slots=block(slots,memory)
        slots=slots.reshape(b,8,self.num_slots,self.hidden_dim)
        for block,reread in zip(self.temporal_blocks,self.temporal_memory_blocks):
            y=block(slots.permute(0,2,1,3).reshape(b*self.num_slots,8,self.hidden_dim)).reshape(b,self.num_slots,8,self.hidden_dim).permute(0,2,1,3); slots=reread(y.reshape(b,8*self.num_slots,self.hidden_dim),memory).reshape(b,8,self.num_slots,self.hidden_dim)
        for block,reread in zip(self.interaction_blocks,self.interaction_memory_blocks): slots=reread(block(slots.reshape(b*8,self.num_slots,self.hidden_dim)).reshape(b,8*self.num_slots,self.hidden_dim),memory).reshape(b,8,self.num_slots,self.hidden_dim)
        return self.output_norm(slots)

class CurrentOnlyPhysionDecoder(nn.Module):
    def __init__(self,**config):
        super().__init__(); keys={'input_dim','hidden_dim','num_heads','ffn_dim','num_slots','slot_depth','transition_depth','interaction_depth','dropout'}; self.decoder=CurrentOnlyDynamicsDecoder(**{k:v for k,v in config.items() if k in keys}); self.probe=PhysionShallowProbe(hidden_dim=self.decoder.hidden_dim,num_heads=config.get('num_heads',8),ffn_dim=config.get('ffn_dim',1024),num_slots=config.get('num_slots',8),num_object_types=config.get('num_object_types',32),dropout=config.get('dropout',.1))
    def forward(self,context): return self.probe(self.decoder(context))
