from __future__ import annotations
import csv
from pathlib import Path
import numpy as np, torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

class EK100Dataset(Dataset):
    def __init__(self, annotations, video_root, indices=None, crop_size=224):
        rows=list(csv.DictReader(Path(annotations).open(newline=''))); self.rows=rows if indices is None else [rows[int(i)] for i in indices]; self.root=Path(video_root); self.crop=int(crop_size)
        if not self.rows: raise ValueError(f'empty EK100 dataset: {annotations}')
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; path=self.root/r['participant_id']/'videos'/f"{r['video_id']}.MP4"
        if not path.exists(): path=path.with_suffix('.mp4')
        vr=VideoReader(str(path),num_threads=1,ctx=cpu(0)); start=max(0,min(int(r['start_frame'])-1,len(vr)-1)); stop=max(start,min(int(r['stop_frame'])-1,len(vr)-1)); idx=np.linspace(start,stop,16).round().astype(np.int64); x=torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0,3,1,2).float()/255.; x=torch.nn.functional.interpolate(x,(self.crop,self.crop),mode='bilinear',align_corners=False); mean=x.new_tensor((.485,.456,.406)).view(1,3,1,1); std=x.new_tensor((.229,.224,.225)).view(1,3,1,1); x=((x-mean)/std).permute(1,0,2,3).contiguous()
        return {'video':x,'verb':int(r['verb_class']),'noun':int(r['noun_class']),'video_id':r['video_id'],'narration_id':r['narration_id']}

def read_rows(path): return list(csv.DictReader(Path(path).open(newline='')))
def split_indices(path, fraction=.1, seed=239):
    rows=read_rows(path); vids=sorted({r['video_id'] for r in rows}); g=np.random.default_rng(seed); g.shuffle(vids); val=set(vids[:max(1,int(round(len(vids)*fraction)))]); return [i for i,r in enumerate(rows) if r['video_id'] not in val],[i for i,r in enumerate(rows) if r['video_id'] in val]
