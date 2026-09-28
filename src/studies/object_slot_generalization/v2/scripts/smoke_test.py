import argparse, torch
from src.studies.object_slot_generalization.v2 import OCSE, ocse_loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--input-dim',type=int,default=32);p.add_argument('--hidden-dim',type=int,default=64);p.add_argument('--batch',type=int,default=2);p.add_argument('--frames',type=int,default=3);p.add_argument('--patches',type=int,default=25);a=p.parse_args()
    d=torch.device(a.device);m=OCSE(a.input_dim,a.hidden_dim).to(d);x=torch.randn(a.batch,a.frames,a.patches,a.input_dim,device=d,requires_grad=True);o=m(x);l=ocse_loss(o,target_features=x)['total'];l.backward();print({'input':list(x.shape),'slots':list(o.slots.shape),'assignment':list(o.assignment.shape),'contribution':list(o.contribution.shape),'loss':float(l.detach())})
if __name__=='__main__':main()
