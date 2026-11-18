"""Calibrate robust candidate, then score a newly created synthetic holdout."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import train_model as t
import evaluate as e
from evaluate_refined import proposals_refined
from train_confusion import slash_zeros

HERE=Path(__file__).resolve().parent;OUT=HERE/'model31_robust';SEG=HERE/'segments_v3'
e.proposals=proposals_refined

def fair_plate_scores(rows,chosen,accept):
    """The runtime decision must not consult the reference string length."""
    correct=accepted=wrong=0
    for r in rows:
        a,n=r['start'],r['count'];truth=r['truth_posthoc']
        text=''.join(t.CLASSES[int(k)] for k in chosen[a:a+n]) if n else ''
        yes=bool(n>0 and r['segmentation']['ok'] and accept[a:a+n].all())
        hit=bool(n==len(truth) and n>0 and text==truth)
        accepted+=yes;correct+=yes and hit;wrong+=yes and not hit
    return correct,accepted,wrong

e.plate_scores=fair_plate_scores

def select_policy_strict(logits,y,rows):
    """Choose only on calibration, and fail instead of silently relaxing precision."""
    fit=e.temperature(logits,y);p=e.softmax(logits,fit['temperature'])
    raw,chosen,score,margin,ratio=proposals_refined(p,rows);best=None
    for conf in (.3,.4,.5,.6,.7,.8,.9,.95,.98,.99,.995,.999,.9999):
      for gap in (.05,.1,.2,.3,.5,.7,.9,.98,.995):
       for ratio_min in (.7,.85,1.):
        accept=(score>=conf)&(margin>=gap)&((raw==chosen)|(ratio>=ratio_min))
        correct,n,wrong=fair_plate_scores(rows,chosen,accept)
        if n and wrong/n<=.01:
            rank=(correct,-wrong,n,conf,gap,ratio_min)
            if best is None or rank>best[0]:best=(rank,conf,gap,ratio_min,correct,n,wrong)
    if best is None:raise RuntimeError('No non-empty calibration policy reached <=1% wrong accepted')
    _,conf,gap,ratio_min,correct,n,wrong=best
    return {**fit,'minimum_score':conf,'minimum_allowed_margin':gap,'minimum_alternative_ratio':ratio_min,
       'calibration_plates':len(rows),'calibration_correct_accepted':correct,'calibration_accepted':n,
       'calibration_wrong_accepted':wrong,'selection':'maximize correct accepted subject to <=1% wrong among accepted on calibration',
       'scope':'synthetic calibration; probability and precision are not guaranteed on new domains'}

def fair_evaluate(logits,rows,policy):
    raw,chosen,accept,p=e.per_sample(logits,rows,policy)
    confusion=np.zeros((len(t.CLASSES),len(t.CLASSES)),np.int64)
    slices={};records=[]
    for r in rows:
        a,n=r['start'],r['count'];truth=r['truth_posthoc']
        segmented=bool(n>0 and r['segmentation']['ok']);align=n==len(truth) and segmented
        rr=''.join(t.CLASSES[int(k)] for k in raw[a:a+n]) if n else ''
        candidate=''.join(t.CLASSES[int(k)] for k in chosen[a:a+n]) if n else ''
        hit_raw=align and rr==truth;hit=align and candidate==truth
        yes=bool(segmented and accept[a:a+n].all())
        rc=sum(x==y for x,y in zip(rr,truth)) if align else 0
        cc=sum(x==y for x,y in zip(candidate,truth)) if align else 0
        tags=['all',r['layout'],'unseen_font' if r['font_holdout'] else 'seen_font',
              'with_stroke_damage' if r['damage_count'] else 'no_recorded_stroke_damage',
              'O_stress' if 'O' in truth else 'without_O']
        if r['layout']=='motorcycle_two_row':tags.append('motorcycle_LL' if truth[3].isalpha() else 'motorcycle_LD')
        for tag in tags:
            d=slices.setdefault(tag,{'plates':0,'segmented':0,'raw_exact':0,'candidate_exact':0,'accepted':0,
              'accepted_correct':0,'accepted_wrong':0,'character_total':0,'character_correct_raw':0,'character_correct_candidate':0})
            d['plates']+=1;d['segmented']+=int(segmented);d['raw_exact']+=int(hit_raw);d['candidate_exact']+=int(hit)
            d['accepted']+=int(yes);d['accepted_correct']+=int(yes and hit);d['accepted_wrong']+=int(yes and not hit)
            d['character_total']+=len(truth);d['character_correct_raw']+=rc;d['character_correct_candidate']+=cc
        if align:
            for ch,pred in zip(truth,raw[a:a+n]):confusion[t.CLASSES.index(ch),pred]+=1
        records.append({'id':r['id'],'truth_posthoc':truth,'raw':rr,'candidate':candidate,'accepted':yes,
                        'correct':hit,'segmented':segmented,'layout':r['layout'],'font_holdout':r['font_holdout']})
    for d in slices.values():
        n=d['plates'];c=d['character_total']
        d['raw_whole_plate_accuracy']=d['raw_exact']/n;d['candidate_whole_plate_accuracy']=d['candidate_exact']/n
        d['accepted_correct_plate_accuracy_including_rejects']=d['accepted_correct']/n
        d['coverage']=d['accepted']/n;d['accepted_precision']=d['accepted_correct']/d['accepted'] if d['accepted'] else None
        d['raw_character_accuracy_including_failed_segmentation']=d['character_correct_raw']/c
        d['candidate_character_accuracy_including_failed_segmentation']=d['character_correct_candidate']/c
    return {'slices':slices,'policy':policy,'synthetic_only':True,'real_camera_accuracy':None,
       'labels_only_used_posthoc':True,'accepted_decision_depends_on_truth_length':False,
       'note':'PC prototype; camera/ESP32/FPGA two-row path not measured'},confusion,records

def models():
    ck=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False)
    model=t.LeNet5K3(len(t.CLASSES),16,28);model.load_state_dict(ck['model']);model.eval()
    integer=e.IntegerModel(ck['model']);integer.manifest['class_order']=list(t.CLASSES)
    return model,integer

def calibrate():
    select=json.loads((OUT/'selection_freeze.json').read_text(encoding='utf-8'))
    if select['checkpoint_sha256']!=t.digest(OUT/'best.pt'):raise ValueError('model changed after selection')
    model,integer=models();t.SEG=SEG;x,y,rows=t.load_data('calibration')
    challenged=slash_zeros(x,y,2026111789)
    twin=[{**row,'start':row['start']+len(x)} for row in rows]
    x=np.concatenate([x,challenged]);y=np.concatenate([y,y]);rows=rows+twin
    print('Calibrating FP32 on clean plus synthetic slashed-zero crops...',flush=True)
    fp=select_policy_strict(t.infer(model,x),y,rows)
    print('Calibrating INT8 on clean plus synthetic slashed-zero crops...',flush=True)
    ip=select_policy_strict(integer.raw(x)/integer.scale,y,rows)
    integer.export(OUT/'generated31')
    paths={'checkpoint':OUT/'best.pt','training_data':HERE/'dataset_v3/manifest.json',
           'segmenter':HERE/'segment.py','evaluation':Path(__file__),'policy':HERE/'evaluate_refined.py',
           'shared_eval':HERE/'evaluate.py','holdout_builder':HERE/'build_final_holdout.py',
           'render_source':HERE/'build_data.py'}
    contract={'sha256':{name:t.digest(path) for name,path in paths.items()},
      'fp32_policy':fp,'int8_policy':ip,'calibration':'v3 calibration, original and synthetic slashed-zero duplicate',
      'test_status':'new independent synthetic holdout not generated yet','rtl_ready':False,
      'scope':'one-row car, two-row car, 9-glyph motorcycle LL and LD; O is synthetic OCR stress class'}
    t.dump(OUT/'calibration_freeze.json',contract);print(json.dumps(contract,indent=2),flush=True)

def test():
    frozen=json.loads((OUT/'calibration_freeze.json').read_text(encoding='utf-8'))
    paths={'checkpoint':OUT/'best.pt','training_data':HERE/'dataset_v3/manifest.json',
           'segmenter':HERE/'segment.py','evaluation':Path(__file__),'policy':HERE/'evaluate_refined.py',
           'shared_eval':HERE/'evaluate.py','holdout_builder':HERE/'build_final_holdout.py',
           'render_source':HERE/'build_data.py'}
    for k,path in paths.items():
        if frozen['sha256'][k]!=t.digest(path):raise ValueError('Frozen source changed: '+k)
    if (OUT/'test_int8_metrics.json').exists():raise FileExistsError('Holdout already evaluated')
    audit=json.loads((HERE/'dataset_final_holdout_audit.json').read_text(encoding='utf-8'))
    if audit['duplicate_identities_within_test'] or audit['identities_in_known_prior_data'] or audit['heldout_font_overlap_known_training']:
        raise ValueError('Holdout overlap')
    model,integer=models();t.SEG=HERE/'segments_final_holdout';x,y,rows=t.load_data('test')
    oldck=torch.load(HERE/'model31_refined/best.pt',map_location='cpu',weights_only=False)
    old=t.LeNet5K3(len(t.CLASSES),16,28);old.load_state_dict(oldck['model'])
    baseline=t.infer(old,x);fp=t.infer(model,x)
    print('Running INT8 on fresh held-out glyphs...',flush=True);ip=integer.raw(x)/integer.scale
    for name,z,policy in [('before_slash_training',baseline,frozen['fp32_policy']),
                          ('fp32',fp,frozen['fp32_policy']),('int8',ip,frozen['int8_policy'])]:
        metrics,cm,records=fair_evaluate(z,rows,policy)
        if name=='before_slash_training':metrics['note']='Only raw comparison is meaningful; decision thresholds calibrated for robust model'
        t.dump(OUT/f'test_{name}_metrics.json',metrics);t.dump(OUT/f'test_{name}_predictions.json',records)
        with (OUT/f'test_{name}_confusion.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['truth\\prediction']+list(t.CLASSES))
            for label,row in zip(t.CLASSES,cm):w.writerow([label]+row.tolist())
        print(json.dumps({'model':name,'all':metrics['slices']['all'],
          'by_layout':{k:metrics['slices'][k] for k in ('car_one_row','car_two_row','motorcycle_two_row')}},ensure_ascii=False),flush=True)

def diagnostic(path):
    import cv2
    from segment import segment
    frozen=json.loads((OUT/'calibration_freeze.json').read_text(encoding='utf-8'));_,integer=models()
    glyphs,info=segment(cv2.imread(str(path),cv2.IMREAD_GRAYSCALE))
    if glyphs is None:print(json.dumps({'error':'segmentation_failed','details':info},ensure_ascii=False));return
    z=integer.raw(glyphs)/integer.scale;rows=[{'start':0,'count':len(glyphs)}]
    raw,pred,accepted,p=e.per_sample(z,rows,frozen['int8_policy'])
    result={'raw':''.join(t.CLASSES[i] for i in raw),'candidate':''.join(t.CLASSES[i] for i in pred),
      'accepted':bool(accepted.all()),'scores':[{'raw':t.CLASSES[int(raw[i])],'choice':t.CLASSES[int(pred[i])],
      'accepted':bool(accepted[i]),'top3':[[t.CLASSES[int(j)],round(float(p[i,j]),5)] for j in np.argsort(-p[i])[:3]]} for i in range(len(raw))],
      'segmentation':info,'labels_supplied_to_model':False}
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('calibrate','test','diagnostic'));ap.add_argument('--image',type=Path);a=ap.parse_args()
    if a.stage=='calibrate':calibrate()
    elif a.stage=='test':test()
    else:
        if a.image is None:ap.error('--image required')
        diagnostic(a.image)
