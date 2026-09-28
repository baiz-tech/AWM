"""V20 EK100 head: local multi-scale evidence for nouns, dynamics for verbs."""
from __future__ import annotations
import math,torch
from torch import nn
from .world import UnifiedWorldModel

class EK100MultiScaleReadout(nn.Module):
 def __init__(self,verb_classes=97,noun_classes=289,pairs=(),token_dim=1280,state_dim=512,slots=8,local_queries=4):
  super().__init__();self.world=UnifiedWorldModel(token_dim,state_dim,slots);self.local_queries=nn.Parameter(torch.randn(1,local_queries,state_dim)*.02);self.local_key=nn.Linear(token_dim,state_dim);self.local_value=nn.Linear(token_dim,state_dim);self.local_attention=nn.MultiheadAttention(state_dim,8,batch_first=True);self.local_norm=nn.LayerNorm(state_dim);self.verb=nn.Linear(state_dim,verb_classes);self.noun=nn.Sequential(nn.LayerNorm(2*state_dim),nn.Linear(2*state_dim,noun_classes));self.action=nn.Sequential(nn.Linear(3*state_dim,state_dim),nn.GELU(),nn.Linear(state_dim,len(pairs)));p=torch.tensor(pairs,dtype=torch.long);self.register_buffer('pv',p[:,0]);self.register_buffer('pn',p[:,1])
 def local_object_evidence(self,grid,return_attention=False):
  # Fine 8x8 tokens and 2x2 pooled tokens make small objects visible while
  # retaining context needed to disambiguate visually similar utensils.
  b,t,n,d=grid.shape;s=math.isqrt(n);assert s*s==n and s%2==0
  fine=grid.reshape(b,t*n,d);coarse=grid.reshape(b,t,s,s,d).reshape(b,t,s//2,2,s//2,2,d).mean((3,5)).reshape(b,t*(s//2)**2,d);tokens=torch.cat((fine,coarse),1);q=self.local_queries.expand(b,-1,-1);z,attention=self.local_attention(q,self.local_key(tokens),self.local_value(tokens),need_weights=return_attention,average_attn_weights=False);evidence=self.local_norm(z).mean(1);return (evidence,attention) if return_attention else evidence
 def forward(self,grid,return_analysis=False):
  state,future=self.world(grid.float());relation=state.relations.mean((1,2));future_scene=future.scene;verb_feature=state.dynamics+.1*future_scene;slot_feature=state.objects.amax((1,2))+.1*relation;local=self.local_object_evidence(grid.float(),return_attention=return_analysis);local_feature,local_attention=local if return_analysis else (local,None);noun_feature=torch.cat((slot_feature,local_feature),1);verb=self.verb(verb_feature);noun=self.noun(noun_feature);joint=torch.cat((verb_feature,slot_feature,local_feature),1);action=verb[:,self.pv]+noun[:,self.pn]+.5*self.action(joint);consistency=1-torch.nn.functional.cosine_similarity(state.objects[:,:-1],state.objects[:,1:],dim=-1).mean()
  if return_analysis:return verb,noun,action,consistency,{"objects":state.objects,"dynamics":state.dynamics,"relations":state.relations,"local_attention":local_attention,"verb_feature":verb_feature,"slot_feature":slot_feature,"local_feature":local_feature}
  return verb,noun,action,consistency
