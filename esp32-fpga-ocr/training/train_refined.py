"""Refine the 31-class candidate for both motorcycle-series layouts."""
from __future__ import annotations
import json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import train_model as t

HERE=Path(__file__).resolve().parent
OUT=HERE/'model31_refined'
DATA=HERE/'dataset_v3'
SEG=HERE/'segments_v3'
V2=HERE/'model31'

def main():
    if (OUT/'best.pt').exists():raise FileExistsError('Refined training already exists')
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    np.random.seed(2026111714);torch.manual_seed(2026111714);random.seed(2026111714)
    t.SEG=SEG
    x,y,rows=t.load_data('train');vx,vy,vrows=t.load_data('validation');assert (y>=0).all()
    # Replay only synthetic training examples from earlier runs; no test/camera images.
    t.SEG=HERE/'segments';v2x,v2y,_=t.load_data('train');t.SEG=SEG
    rng=np.random.default_rng(2026111715);take=rng.choice(len(v2x),min(10000,len(v2x)),replace=False)
    _,old_x,old_y=t.load_split(t.OLD,'train')
    x=np.concatenate([x,v2x[take],old_x]);y=np.concatenate([y,v2y[take],old_y])
    checkpoint=torch.load(V2/'best.pt',map_location='cpu',weights_only=False)
    model=t.LeNet5K3(len(t.CLASSES),16,28);model.load_state_dict(checkpoint['model'])
    t.dump(OUT/'validation_before_refinement.json',t.val_stats(t.infer(model,vx),vrows,vy))
    opt=torch.optim.AdamW(model.parameters(),lr=.00010,weight_decay=.0001)
    counts=np.bincount(y,minlength=len(t.CLASSES));weights=np.clip(np.sqrt(counts.mean()/np.maximum(counts,1)),.5,3.2)
    wt=torch.tensor(weights,dtype=torch.float32);tx=torch.from_numpy(x).float().unsqueeze(1)/127.5-1;ty=torch.from_numpy(y)
    cfg={'seed':2026111714,'epochs':12,'classes':t.CLASSES,'initial_model_sha256':t.digest(V2/'best.pt'),
         'dataset_manifest_sha256':t.digest(DATA/'manifest.json'),'new_glyphs':len(x)-len(old_x)-len(take),
         'previous_new_synthetic_train_replay':len(take),'old_v4_train_replay':len(old_x),
         'source':'synthetic plates only; camera photos excluded',
         'motorcycle_top_row':'two letters or one letter plus one digit',
         'selection':'validation macro raw whole-plate accuracy, then validation NLL',
         'class_counts':dict(zip(t.CLASSES,counts.tolist())),'heldout_test_used':False}
    t.dump(OUT/'training_configuration.json',cfg);history=[];best=(-1.,float('-inf'));valid=vy>=0
    for epoch in range(cfg['epochs']):
        model.train();start=time.perf_counter();total=0
        for ix in torch.randperm(len(ty)).split(512):
            batch=tx[ix];n=len(ix);theta=torch.zeros((n,2,3));theta[:,0,0]=1/(.89+.22*torch.rand(n));theta[:,1,1]=1
            batch=F.grid_sample(batch,F.affine_grid(theta,batch.shape,align_corners=False),padding_mode='border',align_corners=False)
            batch=(batch+.012*torch.randn_like(batch)).clamp(-1,1)
            x0=torch.randint(3,24,(n,1,1));y0=torch.randint(3,24,(n,1,1));ww=torch.randint(1,5,(n,1,1));hh=torch.randint(1,5,(n,1,1))
            xx=torch.arange(28).view(1,1,28);yy=torch.arange(28).view(1,28,1)
            mask=(xx>=x0)&(xx<x0+ww)&(yy>=y0)&(yy<y0+hh)&(torch.rand(n,1,1)<.16)
            batch=torch.where(mask.unsqueeze(1),batch*.35+.65,batch)
            opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(batch),ty[ix],weight=wt,label_smoothing=.01)
            loss.backward();opt.step();total+=float(loss.detach())*n
        logits=t.infer(model,vx);stats=t.val_stats(logits,vrows,vy)
        nll=float(F.cross_entropy(torch.from_numpy(logits[valid]),torch.from_numpy(vy[valid])))
        entry={'epoch':epoch+1,'train_loss':total/len(y),'val_nll':nll,'seconds':round(time.perf_counter()-start,3),**stats}
        history.append(entry);rank=(stats['macro_raw_plate_accuracy'],-nll)
        if rank>best:
            best=rank;torch.save({'model':model.state_dict(),'classes':list(t.CLASSES),'input_size':28,'conv2_channels':16,
                                  'epoch':epoch+1,'synthetic_only':True,'camera_images_in_training':0},OUT/'best.pt')
            t.dump(OUT/'best_validation.json',entry)
        t.dump(OUT/'history.json',history)
        print(json.dumps({'epoch':epoch+1,'train_loss':round(entry['train_loss'],4),'val_plate_macro':round(stats['macro_raw_plate_accuracy'],4),
                          'val_character':round(stats['glyph_raw_accuracy_conditional'],4),'seconds':entry['seconds']}),flush=True)
    t.dump(OUT/'selection_freeze.json',{'checkpoint_sha256':t.digest(OUT/'best.pt'),'dataset_manifest_sha256':t.digest(DATA/'manifest.json'),
       'source_sha256':t.digest(Path(__file__)),'selection':'validation only','test_not_opened':True})
    print(json.dumps({'best_rank':best,'checkpoint':str(OUT/'best.pt')}),flush=True)

if __name__=='__main__':main()
