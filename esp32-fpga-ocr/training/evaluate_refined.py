"""Fresh calibration/test for refined 31-class, multi-row OCR candidate."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
import train_model as t
import evaluate as e

HERE=Path(__file__).resolve().parent
OUT=HERE/'model31_refined'; DATA=HERE/'dataset_v3'; SEG=HERE/'segments_v3'

def proposals_refined(p,rows):
    """Scoped position filter; row-two motorcycle position 3 allows digit OR letter.

    0/O scores are combined only when both have nontrivial visual support and
    the position has exactly one allowed alternative. This is a format prior,
    not evidence that a plate is registered or a replacement for rejection.
    """
    raw=p.argmax(1);chosen=np.zeros(len(p),np.int64);score=np.zeros(len(p));margin=np.zeros(len(p));ratio=np.zeros(len(p))
    zero=t.CLASSES.index('0');letter_o=t.CLASSES.index('O')
    for r in rows:
        a,n=r['start'],r['count']
        for pos in range(n):
            i=a+pos; unrestricted=n==9 and pos==3
            allowed=(np.arange(len(t.CLASSES)) if unrestricted else
                     np.arange(10,len(t.CLASSES)) if pos==2 else np.arange(10))
            work=p[i].copy()
            pair=work[zero]+work[letter_o]
            if not unrestricted and pair>=.8 and min(work[zero],work[letter_o])>=.12:
                target=letter_o if pos==2 else zero
                work[target]=pair
            order=allowed[np.argsort(-work[allowed],kind='stable')];chosen[i]=order[0]
            score[i]=work[order[0]];margin[i]=work[order[0]]-work[order[1]]
            ratio[i]=work[order[0]]/max(p[i,raw[i]],1e-12)
    return raw,chosen,score,margin,ratio

e.proposals=proposals_refined
t.SEG=SEG

def models():
    ck=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False)
    model=t.LeNet5K3(len(t.CLASSES),16,28);model.load_state_dict(ck['model']);model.eval()
    integer=e.IntegerModel(ck['model']);integer.manifest['class_order']=list(t.CLASSES)
    return model,integer

def calibrate():
    freeze=json.loads((OUT/'selection_freeze.json').read_text(encoding='utf-8'))
    if freeze['checkpoint_sha256']!=t.digest(OUT/'best.pt'):raise ValueError('checkpoint changed')
    model,integer=models();x,y,rows=t.load_data('calibration')
    print('Calibrating FP32...',flush=True);fp=e.select_policy(t.infer(model,x),y,rows)
    print('Calibrating INT8...',flush=True);ip=e.select_policy(integer.raw(x)/integer.scale,y,rows)
    integer.export(OUT/'generated31')
    contract={'checkpoint_sha256':t.digest(OUT/'best.pt'),'dataset_manifest_sha256':t.digest(DATA/'manifest.json'),
              'segment_sha256':t.digest(HERE/'segment.py'),'evaluation_source_sha256':t.digest(Path(__file__)),
              'shared_evaluation_sha256':t.digest(HERE/'evaluate.py'),
              'fp32_policy':fp,'int8_policy':ip,'0O_rule':'merge only if pair >= .8 and each >= .12, only at a restricted position',
              'motorcycle_position_3':'digit OR letter','test_not_opened':True,
              'rtl_deployment_ready':False,'reason':'existing FPGA output count and ESP32 8-glyph code require changes'}
    t.dump(OUT/'calibration_freeze.json',contract);print(json.dumps(contract,indent=2),flush=True)

def test():
    contract=json.loads((OUT/'calibration_freeze.json').read_text(encoding='utf-8'))
    for key,path in [('checkpoint_sha256',OUT/'best.pt'),('dataset_manifest_sha256',DATA/'manifest.json'),
                     ('segment_sha256',HERE/'segment.py'),('evaluation_source_sha256',Path(__file__)),
                     ('shared_evaluation_sha256',HERE/'evaluate.py')]:
        if contract[key]!=t.digest(path):raise ValueError('frozen source changed: '+key)
    if (OUT/'test_int8_metrics.json').exists():raise FileExistsError('Test already opened')
    audit=json.loads((HERE/'dataset_v3_audit.json').read_text(encoding='utf-8'))
    if audit['test_overlap_prior_datasets'] or audit['heldout_font_overlap_training'] or audit['heldout_font_overlap_known_prior_train'] or any(audit['split_identity_overlap'].values()):raise ValueError('Data overlap')
    model,integer=models();x,y,rows=t.load_data('test')
    prev=torch.load(HERE/'model31/best.pt',map_location='cpu',weights_only=False)
    baseline=t.LeNet5K3(len(t.CLASSES),16,28);baseline.load_state_dict(prev['model'])
    logits0=t.infer(baseline,x);logits=t.infer(model,x)
    print('Computing held-out INT8 logits...',flush=True);ilogits=integer.raw(x)/integer.scale
    for name,z,policy in [('before_refinement',logits0,contract['fp32_policy']),('fp32',logits,contract['fp32_policy']),('int8',ilogits,contract['int8_policy'])]:
        metrics,cm,records=e.evaluate(z,rows,policy)
        if name=='before_refinement':metrics['note']='only raw comparison is meaningful; policy was calibrated on refined model'
        t.dump(OUT/f'test_{name}_metrics.json',metrics);t.dump(OUT/f'test_{name}_predictions.json',records)
        with (OUT/f'test_{name}_confusion.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['truth\\prediction']+list(t.CLASSES))
            for label,row in zip(t.CLASSES,cm):w.writerow([label]+row.tolist())
        print(json.dumps({'model':name,'all':metrics['slices']['all'],
                          'by_layout':{k:metrics['slices'][k] for k in ('car_one_row','car_two_row','motorcycle_two_row')}},ensure_ascii=False),flush=True)

def diagnostic(path):
    import cv2
    from segment import segment
    contract=json.loads((OUT/'calibration_freeze.json').read_text(encoding='utf-8'));_,integer=models()
    glyphs,info=segment(cv2.imread(str(path),cv2.IMREAD_GRAYSCALE))
    if glyphs is None:print(json.dumps({'error':'segmentation_failed','details':info},ensure_ascii=False));return
    z=integer.raw(glyphs)/integer.scale; rows=[{'start':0,'count':len(glyphs)}]
    raw,pred,accepted,p=e.per_sample(z,rows,contract['int8_policy'])
    print(json.dumps({'raw':''.join(t.CLASSES[i] for i in raw),'candidate':''.join(t.CLASSES[i] for i in pred),
      'accepted':bool(accepted.all()),'scores':[{'raw':t.CLASSES[int(raw[i])],'choice':t.CLASSES[int(pred[i])],
       'accepted':bool(accepted[i]),'top3':[[t.CLASSES[int(j)],round(float(p[i,j]),5)] for j in np.argsort(-p[i])[:3]]} for i in range(len(raw))],
      'segmentation':info,'labels_supplied_to_model':False},ensure_ascii=False,indent=2))

if __name__=='__main__':
    torch.set_num_threads(4)
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('calibrate','test','diagnostic'));ap.add_argument('--image',type=Path);a=ap.parse_args()
    if a.stage=='calibrate':calibrate()
    elif a.stage=='test':test()
    else:
        if a.image is None:ap.error('--image required')
        diagnostic(a.image)
