"""Calibrate on synthetic calibration only; evaluate the frozen test once.

Candidate selection uses validation only. Only pixels reach the OCR pipeline;
truth, class frequency and source filenames are used solely for later scoring.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from pathlib import Path
import cv2
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
UART=HERE.parent
PROJECT=Path(r'D:\TOTNGHIEP\license_plate_ai')
sys.path.insert(0,str(UART));sys.path.insert(0,str(PROJECT))
from lenet5_k3 import LeNet5K3
from one_row_recovery import IntegerModel
from segment import segment
from train_model import BASE,CLASSES,OLD,OUT,SEG,digest,infer,load_data

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def load_models():
    checkpoint=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False)
    model=LeNet5K3(len(CLASSES),16,28);model.load_state_dict(checkpoint['model']);model.eval()
    integer=IntegerModel(checkpoint['model']);integer.manifest['class_order']=list(CLASSES)
    return model,integer

def softmax(logits,temp):
    z=logits.astype(np.float64)/temp;z-=np.max(z,axis=1,keepdims=True)
    e=np.exp(z);return e/e.sum(axis=1,keepdims=True)

def temperature(logits,y):
    valid=y>=0;y=y[valid];logits=logits[valid]
    vals=np.geomspace(.15,5,101);best=(float('inf'),1.)
    for t in vals:
        p=softmax(logits,t);nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())
        if nll<best[0]:best=(nll,float(t))
    return {'temperature':best[1],'calibration_nll':best[0],'samples':len(y)}

def proposals(p,rows):
    """Apply only the scoped character position, no labels or identity."""
    pred=p.argmax(axis=1);chosen=np.zeros(len(p),np.int64);margin=np.zeros(len(p));ratio=np.zeros(len(p));score=np.zeros(len(p))
    for r in rows:
        a,n=r['start'],r['count']
        for pos in range(n):
            i=a+pos;allowed=np.arange(10,len(CLASSES)) if pos==2 else np.arange(10)
            order=allowed[np.argsort(-p[i,allowed],kind='stable')];chosen[i]=order[0]
            score[i]=p[i,order[0]];margin[i]=p[i,order[0]]-p[i,order[1]]
            ratio[i]=p[i,order[0]]/max(p[i,pred[i]],1e-12)
    return pred,chosen,score,margin,ratio

def plate_scores(rows,chosen,accept):
    correct=accepted=wrong=0
    for r in rows:
        a,n=r['start'],r['count'];ok=n==len(r['truth_posthoc']) and n>0
        text=''.join(CLASSES[int(k)] for k in chosen[a:a+n]) if n else ''
        yes=bool(ok and accept[a:a+n].all()); hit=bool(ok and text==r['truth_posthoc'])
        accepted+=yes;correct+=yes and hit;wrong+=yes and not hit
    return correct,accepted,wrong

def select_policy(logits,y,rows):
    fit=temperature(logits,y);p=softmax(logits,fit['temperature'])
    raw,chosen,score,margin,ratio=proposals(p,rows)
    best=None
    for conf in (.3,.4,.5,.6,.7):
      for gap in (.05,.1,.2,.3):
       for ratio_min in (.7,.85,1.):
        accept=(score>=conf)&(margin>=gap)&((raw==chosen)|(ratio>=ratio_min))
        correct,n,wrong=plate_scores(rows,chosen,accept)
        if n and wrong/n<=.01:
            rank=(correct,-wrong,n,conf,gap,ratio_min)
            if best is None or rank>best[0]:best=(rank,conf,gap,ratio_min,correct,n,wrong)
    if best is None:
        raise RuntimeError('No calibration threshold achieved <=1% accepted errors')
    _,conf,gap,ratio_min,correct,n,wrong=best
    return {**fit,'minimum_score':conf,'minimum_allowed_margin':gap,'minimum_alternative_ratio':ratio_min,
            'calibration_plates':len(rows),'calibration_correct_accepted':correct,'calibration_accepted':n,
            'calibration_wrong_accepted':wrong,'selection':'maximize correct accepted subject to <=1% wrong among accepted on calibration',
            'scope':'only given 8/9-character one/two-row synthetic layouts; score is calibrated model output, not guaranteed probability'}

def calibrate():
    freeze=json.loads((OUT/'selection_freeze.json').read_text(encoding='utf-8'))
    if freeze['checkpoint_sha256']!=digest(OUT/'best.pt'):raise ValueError('checkpoint changed')
    model,integer=load_models();x,y,rows=load_data('calibration')
    print('Calibrating FP32 on calibration split...',flush=True)
    fp=infer(model,x);fp_policy=select_policy(fp,y,rows)
    print('Calibrating INT8 on calibration split...',flush=True)
    ip=integer.raw(x)/integer.scale;ip_policy=select_policy(ip,y,rows)
    integer.export(OUT/'generated31')
    contract={'checkpoint_sha256':digest(OUT/'best.pt'),'dataset_manifest_sha256':digest(HERE/'dataset_v2/manifest.json'),
              'segment_source_sha256':digest(HERE/'segment.py'),'evaluation_source_sha256':digest(Path(__file__)),
              'fp32_policy':fp_policy,'int8_policy':ip_policy,
              'label_use':'calibration only for temperature and thresholds; never inference',
              'rtl_deployment_ready':False,'reason':'current FPGA has 30 output classes and one-row/8-character grammar'}
    dump(OUT/'calibration_freeze.json',contract)
    print(json.dumps({'fp32':fp_policy,'int8':ip_policy},indent=2),flush=True)

def per_sample(logits,rows,policy):
    p=softmax(logits,policy['temperature']);raw,chosen,score,margin,ratio=proposals(p,rows)
    accepted=(score>=policy['minimum_score'])&(margin>=policy['minimum_allowed_margin'])&((raw==chosen)|(ratio>=policy['minimum_alternative_ratio']))
    return raw,chosen,accepted,p

def evaluate(logits,rows,policy):
    raw,chosen,accept,p=per_sample(logits,rows,policy)
    confusion=np.zeros((len(CLASSES),len(CLASSES)),np.int64)
    slices={'all':{'plates':0,'segmented':0,'raw_exact':0,'candidate_exact':0,'accepted':0,'accepted_correct':0,
                   'accepted_wrong':0,'character_total':0,'character_correct_raw':0,'character_correct_candidate':0}}
    records=[]
    for r in rows:
        a,n=r['start'],r['count'];truth=r['truth_posthoc'];ok=n==len(truth) and n>0
        rr=''.join(CLASSES[int(k)] for k in raw[a:a+n]) if n else ''
        candidate=''.join(CLASSES[int(k)] for k in chosen[a:a+n]) if n else ''
        hit_raw=ok and rr==truth;hit=ok and candidate==truth;yes=bool(ok and accept[a:a+n].all())
        rc=sum(x==y for x,y in zip(rr,truth)) if ok else 0
        cc=sum(x==y for x,y in zip(candidate,truth)) if ok else 0
        tags=[r['layout'],'unseen_font' if r['font_holdout'] else 'seen_font',
              'with_stroke_damage' if r['damage_count'] else 'no_recorded_stroke_damage',
              'O_stress' if 'O' in truth else 'without_O']
        for tag in ['all']+tags:
            if tag not in slices:slices[tag]={'plates':0,'segmented':0,'raw_exact':0,'candidate_exact':0,'accepted':0,'accepted_correct':0,'accepted_wrong':0,'character_total':0,'character_correct_raw':0,'character_correct_candidate':0}
            d=slices[tag];d['plates']+=1;d['segmented']+=int(ok);d['raw_exact']+=int(hit_raw);d['candidate_exact']+=int(hit)
            d['accepted']+=int(yes);d['accepted_correct']+=int(yes and hit);d['accepted_wrong']+=int(yes and not hit)
            d['character_total']+=len(truth);d['character_correct_raw']+=rc;d['character_correct_candidate']+=cc
        if ok:
            for ch,pred in zip(truth,raw[a:a+n]):confusion[CLASSES.index(ch),pred]+=1
        records.append({'id':r['id'],'truth_posthoc':truth,'raw':rr,'candidate':candidate,'accepted':yes,
                        'correct':hit,'segmented':ok,'layout':r['layout'],'font_holdout':r['font_holdout']})
    for d in slices.values():
        n=d['plates'];c=d['character_total']
        d['raw_whole_plate_accuracy']=d['raw_exact']/n;d['candidate_whole_plate_accuracy']=d['candidate_exact']/n
        d['accepted_correct_plate_accuracy_including_rejects']=d['accepted_correct']/n
        d['coverage']=d['accepted']/n;d['accepted_precision']=d['accepted_correct']/d['accepted'] if d['accepted'] else None
        d['raw_character_accuracy_including_failed_segmentation']=d['character_correct_raw']/c
        d['candidate_character_accuracy_including_failed_segmentation']=d['character_correct_candidate']/c
    return {'slices':slices,'policy':policy,'synthetic_only':True,'real_camera_accuracy':None,
            'labels_only_used_posthoc':True,'note':'actual camera/ESP32/FPGA two-row path not measured'},confusion,records

def test():
    contract=json.loads((OUT/'calibration_freeze.json').read_text(encoding='utf-8'))
    for key,path in [('checkpoint_sha256',OUT/'best.pt'),('dataset_manifest_sha256',HERE/'dataset_v2/manifest.json'),
                     ('segment_source_sha256',HERE/'segment.py'),('evaluation_source_sha256',Path(__file__))]:
        if contract[key]!=digest(path):raise ValueError(f'Frozen file changed: {key}')
    if (OUT/'test_fp32_metrics.json').exists():raise FileExistsError('Test already opened; do not tune against it')
    audit=json.loads((HERE/'dataset_audit.json').read_text(encoding='utf-8'))
    if audit['test_identity_overlap_any_known_historic'] or any(audit['split_identity_overlap'].values()):
        raise ValueError('Identity overlap detected; do not evaluate')
    model,integer=load_models();x,y,rows=load_data('test')
    oldck=torch.load(OLD/'best.pt',map_location='cpu',weights_only=False)
    baseline=LeNet5K3(len(BASE),16,28);baseline.load_state_dict(oldck['model']);old=infer(baseline,x)
    old=np.pad(old,((0,0),(0,1)),constant_values=-1e9)
    fp=infer(model,x)
    print('Computing held-out INT8 logits...',flush=True)
    integer_logits=integer.raw(x)/integer.scale
    for name,logits,policy in [('baseline30',old,contract['fp32_policy']),('fp32',fp,contract['fp32_policy']),('int8',integer_logits,contract['int8_policy'])]:
        metrics,cm,records=evaluate(logits,rows,policy)
        if name=='baseline30':metrics['note']='baseline lacks O; candidate policy belongs to new model, so compare only raw results'
        dump(OUT/f'test_{name}_metrics.json',metrics)
        dump(OUT/f'test_{name}_predictions.json',records)
        with (OUT/f'test_{name}_confusion.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['truth\\prediction']+list(CLASSES))
            for label,row in zip(CLASSES,cm):w.writerow([label]+row.tolist())
        print(json.dumps({'model':name,'all':metrics['slices']['all'],
                          'by_layout':{k:metrics['slices'][k] for k in ('car_one_row','car_two_row','motorcycle_two_row')}},ensure_ascii=False),flush=True)

def diagnostic(path):
    contract=json.loads((OUT/'calibration_freeze.json').read_text(encoding='utf-8'))
    _,integer=load_models();glyphs,info=segment(cv2.imread(str(path),cv2.IMREAD_GRAYSCALE))
    if glyphs is None:print(json.dumps({'error':'segmentation_failed','details':info},ensure_ascii=False));return
    logits=integer.raw(glyphs)/integer.scale
    rows=[{'start':0,'count':len(glyphs)}];r,c,a,p=per_sample(logits,rows,contract['int8_policy'])
    out={'raw':''.join(CLASSES[i] for i in r),'candidate':''.join(CLASSES[i] for i in c),'accepted':bool(a.all()),
         'per_character':[{'raw':CLASSES[int(r[i])],'candidate':CLASSES[int(c[i])],'accepted':bool(a[i]),
                           'top3':[[CLASSES[int(j)],round(float(p[i,j]),5)] for j in np.argsort(-p[i])[:3]]} for i in range(len(r))],
         'segmentation':info,'labels_supplied_to_model':False}
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4);cv2.setNumThreads(1)
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('calibrate','test','diagnostic'));ap.add_argument('--image',type=Path);args=ap.parse_args()
    if args.stage=='calibrate':calibrate()
    elif args.stage=='test':test()
    else:
        if args.image is None:ap.error('--image required')
        diagnostic(args.image)
