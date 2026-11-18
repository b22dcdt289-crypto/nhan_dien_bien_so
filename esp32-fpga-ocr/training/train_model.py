"""Fine-tune a 31-class LeNet K3 on synthetic glyphs; select on validation only."""
from __future__ import annotations
import hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
UART=HERE.parent
ROOT=UART.parent
PROJECT=Path(r'D:\TOTNGHIEP\license_plate_ai')
sys.path.insert(0,str(UART));sys.path.insert(0,str(PROJECT))
from lenet5_k3 import LeNet5K3
from one_row_recovery import load_split

BASE='0123456789ABCDEFGHKLMNPSTUVXYZ'
CLASSES=BASE+'O'
OLD=UART/'one_row_recovery_20261101_v4'
SEG=HERE/'segments'
OUT=HERE/'model31'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def load_data(split):
    x=np.load(SEG/f'{split}_glyphs.npz')['x']
    rows=json.loads((SEG/f'{split}_rows.json').read_text(encoding='utf-8'))
    labels=np.full(len(x),-1,dtype=np.int64)
    for r in rows:
        if r['count']==len(r['truth_posthoc']):
            labels[r['start']:r['start']+r['count']]=[CLASSES.index(c) for c in r['truth_posthoc']]
    return x,labels,rows

def val_stats(logits,rows,labels):
    pred=logits.argmax(1);groups={}
    for r in rows:
        typ=r['layout'];g=groups.setdefault(typ,{'plates':0,'correct_raw':0,'segmentation_failures':0})
        g['plates']+=1;g['segmentation_failures']+=int(not r['segmentation']['ok'])
        if r['count']==len(r['truth_posthoc']):
            text=''.join(CLASSES[j] for j in pred[r['start']:r['start']+r['count']])
            g['correct_raw']+=int(text==r['truth_posthoc'])
    for g in groups.values():g['raw_exact_rate']=g['correct_raw']/g['plates']
    valid=labels>=0
    return {'per_layout':groups,'macro_raw_plate_accuracy':float(np.mean([g['raw_exact_rate'] for g in groups.values()])),
            'glyph_raw_accuracy_conditional':float(np.mean(pred[valid]==labels[valid]))}

def infer(model,x):
    model.eval();chunks=[]
    with torch.inference_mode():
        for ix in range(0,len(x),512):
            t=torch.from_numpy(x[ix:ix+512]).float().unsqueeze(1)/127.5-1
            chunks.append(model(t).numpy())
    return np.concatenate(chunks)

def main():
    if (OUT/'best.pt').exists():raise FileExistsError('Training result already exists; use a new run directory')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);np.random.seed(2026111703);torch.manual_seed(2026111703);random.seed(2026111703)
    x,y,train_rows=load_data('train');assert np.all(y>=0)
    vx,vy,vrows=load_data('validation');valid=vy>=0
    old_rows,old_x,old_y=load_split(OLD,'train')
    assert len(old_x)==len(old_y) and all(r['split']=='train' for r in old_rows)
    x=np.concatenate([x,old_x]);y=np.concatenate([y,old_y])
    counts=np.bincount(y,minlength=len(CLASSES)); weights=np.clip(np.sqrt(counts.mean()/np.maximum(counts,1)),.5,3.2)
    checkpoint=torch.load(OLD/'best.pt',map_location='cpu',weights_only=False)
    base=LeNet5K3(len(BASE),16,28);base.load_state_dict(checkpoint['model'])
    baseline=val_stats(infer(base,vx),vrows,vy)
    dump(OUT/'validation_baseline_30class.json',baseline)
    model=LeNet5K3(len(CLASSES),16,28)
    state=checkpoint['model'];target=model.state_dict()
    for key in target:
        if key.startswith('classifier.4.'):
            target[key][:30]=state[key]
        else:target[key]=state[key]
    target['classifier.4.weight'][30]=state['classifier.4.weight'][0].clone()
    target['classifier.4.bias'][30]=state['classifier.4.bias'][0].clone()-.15
    model.load_state_dict(target)
    opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.0001)
    weight_tensor=torch.tensor(weights,dtype=torch.float32)
    tx=torch.from_numpy(x).float().unsqueeze(1)/127.5-1;ty=torch.from_numpy(y)
    config={'seed':2026111703,'epochs':16,'architecture':'LeNet K3, 6/16 conv channels, 120/84 FC, 31 outputs',
            'classes':CLASSES,'old_model_sha256':digest(OLD/'best.pt'),
            'dataset_manifest_sha256':digest(HERE/'dataset_v2/manifest.json'),
            'new_train_glyphs':int(len(x)-len(old_x)),'replay_old_synthetic_train_glyphs':int(len(old_x)),
            'validation_glyphs':int(len(vx)),'class_counts':dict(zip(CLASSES,counts.tolist())),
            'synthetic_training_only':True,'camera_images_in_training':0,
            'early_selection':'macro raw whole-plate accuracy across three validation layouts, then validation NLL',
            'augmentations':'horizontal scale, mild noise, local partial stroke erasure; no geometric tilt',
            'test_used_in_training_or_selection':False}
    dump(OUT/'training_configuration.json',config)
    history=[];best=(-1.,float('-inf'))
    for epoch in range(config['epochs']):
        model.train();started=time.perf_counter();loss_sum=0
        for ix in torch.randperm(len(ty)).split(512):
            batch=tx[ix];n=len(ix)
            theta=torch.zeros((n,2,3));theta[:,0,0]=1/(.89+.22*torch.rand(n));theta[:,1,1]=1
            batch=F.grid_sample(batch,F.affine_grid(theta,batch.shape,align_corners=False),padding_mode='border',align_corners=False)
            batch=(batch+.012*torch.randn_like(batch)).clamp(-1,1)
            # Small, randomly placed partial erasure independent of identity.
            x0=torch.randint(3,24,(n,1,1));y0=torch.randint(3,24,(n,1,1))
            ww=torch.randint(1,5,(n,1,1));hh=torch.randint(1,5,(n,1,1))
            xx=torch.arange(28).view(1,1,28);yy=torch.arange(28).view(1,28,1)
            region=(xx>=x0)&(xx<x0+ww)&(yy>=y0)&(yy<y0+hh)&(torch.rand(n,1,1)<.16)
            batch=torch.where(region.unsqueeze(1),batch*.35+.65,batch)
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(batch),ty[ix],weight=weight_tensor,label_smoothing=.01)
            loss.backward();opt.step();loss_sum+=float(loss.detach())*n
        v_logits=infer(model,vx);stats=val_stats(v_logits,vrows,vy)
        val_nll=float(F.cross_entropy(torch.from_numpy(v_logits[valid]),torch.from_numpy(vy[valid])))
        entry={'epoch':epoch+1,'train_loss':loss_sum/len(y),'val_nll':val_nll,
               'seconds':round(time.perf_counter()-started,3),**stats}
        history.append(entry);rank=(stats['macro_raw_plate_accuracy'],-val_nll)
        if rank>best:
            best=rank
            torch.save({'model':model.state_dict(),'classes':list(CLASSES),'input_size':28,'conv2_channels':16,
                        'epoch':epoch+1,'synthetic_only':True,'camera_images_in_training':0},OUT/'best.pt')
            dump(OUT/'best_validation.json',entry)
        dump(OUT/'history.json',history)
        print(json.dumps({'epoch':epoch+1,'train_loss':round(entry['train_loss'],4),
                          'val_plate_macro':round(stats['macro_raw_plate_accuracy'],4),
                          'val_character':round(stats['glyph_raw_accuracy_conditional'],4),
                          'seconds':entry['seconds']},ensure_ascii=False),flush=True)
    dump(OUT/'selection_freeze.json',{'checkpoint_sha256':digest(OUT/'best.pt'),
         'dataset_manifest_sha256':digest(HERE/'dataset_v2/manifest.json'),
         'source_sha256':digest(Path(__file__)),'selected_using':'validation only','test_not_opened':True})
    print(json.dumps({'best_validation_rank':best,'checkpoint':str(OUT/'best.pt')}),flush=True)

if __name__=='__main__':main()
