"""Synthetic slashed-zero robustness training; new holdout remains sealed."""
from __future__ import annotations
import json,random,time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
import train_model as t

HERE=Path(__file__).resolve().parent
OUT=HERE/'model31_robust';SEG=HERE/'segments_v3';INITIAL=HERE/'model31_refined/best.pt'

def slash_zeros(images,labels,seed=2026111781):
    rng=np.random.default_rng(seed);out=images.copy()
    for i in np.flatnonzero(labels==0):
        # A short diagonal mark inside the glyph; never uses real camera pixels.
        cx=int(rng.integers(12,16));cy=int(rng.integers(12,16));span=int(rng.integers(3,6))
        cv2.line(out[i],(cx-span,cy+span),(cx+span,cy-span),int(rng.integers(0,35)),int(rng.integers(1,3)),cv2.LINE_AA)
    return out

def main():
    if (OUT/'best.pt').exists():raise FileExistsError('Robust training already exists')
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    torch.manual_seed(2026111780);np.random.seed(2026111780);random.seed(2026111780)
    t.SEG=SEG;x,y,_=t.load_data('train');vx,vy,vrows=t.load_data('validation');assert (y>=0).all()
    t.SEG=HERE/'segments';x2,y2,_=t.load_data('train');t.SEG=SEG
    rng=np.random.default_rng(2026111782);take=rng.choice(len(x2),min(10000,len(x2)),replace=False)
    _,oldx,oldy=t.load_split(t.OLD,'train')
    x=np.concatenate([x,x2[take],oldx]);y=np.concatenate([y,y2[take],oldy]);valid=vy>=0
    stress=slash_zeros(vx,vy);ck=torch.load(INITIAL,map_location='cpu',weights_only=False)
    model=t.LeNet5K3(len(t.CLASSES),16,28);model.load_state_dict(ck['model'])
    t.dump(OUT/'validation_before.json',{'clean':t.val_stats(t.infer(model,vx),vrows,vy),
                                          'slashed_zero':t.val_stats(t.infer(model,stress),vrows,vy)})
    opt=torch.optim.AdamW(model.parameters(),lr=.00008,weight_decay=.0001)
    counts=np.bincount(y,minlength=len(t.CLASSES));weights=np.clip(np.sqrt(counts.mean()/np.maximum(counts,1)),.5,3.2)
    wt=torch.tensor(weights,dtype=torch.float32);tx=torch.from_numpy(x).float().unsqueeze(1)/127.5-1;ty=torch.from_numpy(y)
    cfg={'seed':2026111780,'epochs':12,'classes':t.CLASSES,'initial_model_sha256':t.digest(INITIAL),
         'new_train_glyphs':len(x)-len(oldx)-len(take),'v2_synthetic_replay':len(take),'v4_synthetic_replay':len(oldx),
         'augmentation':'random short diagonal marks on some synthetic zeros, local stroke dropout, horizontal scaling, noise',
         'camera_training_images':0,'source':'synthetic only','selection':'minimum of clean and slashed-zero validation macro raw plate accuracy, then validation NLL',
         'heldout_test_used_in_training':False,'dataset_v3_sha256':t.digest(HERE/'dataset_v3/manifest.json')}
    t.dump(OUT/'training_configuration.json',cfg);best=(-1.,float('-inf'));history=[]
    for epoch in range(cfg['epochs']):
        model.train();started=time.perf_counter();total=0
        for ix in torch.randperm(len(ty)).split(512):
            batch=tx[ix];n=len(ix);labels=ty[ix]
            theta=torch.zeros((n,2,3));theta[:,0,0]=1/(.89+.22*torch.rand(n));theta[:,1,1]=1
            batch=F.grid_sample(batch,F.affine_grid(theta,batch.shape,align_corners=False),padding_mode='border',align_corners=False)
            batch=(batch+.012*torch.randn_like(batch)).clamp(-1,1)
            xx=torch.arange(28).view(1,1,28);yy=torch.arange(28).view(1,28,1)
            x0=torch.randint(3,24,(n,1,1));y0=torch.randint(3,24,(n,1,1));ww=torch.randint(1,5,(n,1,1));hh=torch.randint(1,5,(n,1,1))
            cut=(xx>=x0)&(xx<x0+ww)&(yy>=y0)&(yy<y0+hh)&(torch.rand(n,1,1)<.16)
            batch=torch.where(cut.unsqueeze(1),batch*.35+.65,batch)
            centre=27+torch.randint(-2,3,(n,1,1));thickness=torch.randint(0,2,(n,1,1))
            slash=(torch.abs(xx+yy-centre)<=thickness)&(xx>=9)&(xx<=19)&(yy>=9)&(yy<=19)
            slash=slash&(labels.view(n,1,1)==0)&(torch.rand(n,1,1)<.38)
            batch=torch.where(slash.unsqueeze(1),batch*.12-.88,batch)
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(batch),labels,weight=wt,label_smoothing=.01)
            loss.backward();opt.step();total+=float(loss.detach())*n
        clean=t.infer(model,vx);challenged=t.infer(model,stress)
        cs=t.val_stats(clean,vrows,vy);ss=t.val_stats(challenged,vrows,vy)
        nll=(float(F.cross_entropy(torch.from_numpy(clean[valid]),torch.from_numpy(vy[valid])))+
             float(F.cross_entropy(torch.from_numpy(challenged[valid]),torch.from_numpy(vy[valid]))))/2
        entry={'epoch':epoch+1,'train_loss':total/len(y),'val_nll_mean':nll,'seconds':round(time.perf_counter()-started,3),
               'clean':cs,'slashed_zero':ss}
        history.append(entry);rank=(min(cs['macro_raw_plate_accuracy'],ss['macro_raw_plate_accuracy']),-nll)
        if rank>best:
            best=rank;torch.save({'model':model.state_dict(),'classes':list(t.CLASSES),'input_size':28,'conv2_channels':16,
              'epoch':epoch+1,'synthetic_only':True,'camera_images_in_training':0},OUT/'best.pt');t.dump(OUT/'best_validation.json',entry)
        t.dump(OUT/'history.json',history)
        print(json.dumps({'epoch':epoch+1,'clean_plate_macro':round(cs['macro_raw_plate_accuracy'],4),
          'slashed_plate_macro':round(ss['macro_raw_plate_accuracy'],4),'seconds':entry['seconds']}),flush=True)
    t.dump(OUT/'selection_freeze.json',{'checkpoint_sha256':t.digest(OUT/'best.pt'),
      'source_sha256':t.digest(Path(__file__)),'validation_dataset_sha256':t.digest(HERE/'dataset_v3/manifest.json'),
      'selection':'clean and synthetic slashed-zero validation; independent test not opened'})
    print(json.dumps({'best_rank':best,'checkpoint':str(OUT/'best.pt')}),flush=True)

if __name__=='__main__':main()
