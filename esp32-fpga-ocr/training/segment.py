"""PC-only pixel-based plate segmentation prototype, supporting 8/9 glyphs.

Only image pixels enter segment(). Ground truth is used after segmentation for
training alignment and evaluation. This is not the ESP32 firmware implementation.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

HERE=Path(__file__).resolve().parent
DATA=HERE/'dataset_v2'
OUT=HERE/'segments'

def enhance(gray):
    # Approximate bounded contrast/sharpening used by the existing ESP32 code.
    p05,p95=np.percentile(gray,[5,95]); contrast=p95-p05
    if contrast<25:return None
    gain=np.clip(215/max(contrast,1),1,1.5)
    src=np.clip(np.rint((gray.astype(np.float32)-p05)*gain+16),0,255).astype(np.uint8)
    blurred=cv2.GaussianBlur(src,(3,3),0,borderType=cv2.BORDER_REPLICATE)
    detail=src.astype(np.int16)-blurred.astype(np.int16)
    correction=np.where(np.abs(detail)>3,np.clip(np.trunc(detail*0.75),-18,18),0)
    adjusted=src.astype(np.int16)+correction.astype(np.int16)
    lo=cv2.erode(src,np.ones((3,3),np.uint8));hi=cv2.dilate(src,np.ones((3,3),np.uint8))
    return np.clip(adjusted,lo,hi).astype(np.uint8)

def normalize(gray,box):
    x0,y0,x1,y1=box;h,w=gray.shape
    x0=max(0,x0-1);y0=max(0,y0-1);x1=min(w,x1+1);y1=min(h,y1+1)
    crop=gray[y0:y1,x0:x1]
    if crop.size==0:raise ValueError('empty glyph crop')
    ch,cw=crop.shape;scale=min(22/ch,22/cw)
    nh=max(1,round(ch*scale));nw=max(1,round(cw*scale))
    scaled=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA)
    out=np.full((28,28),255,np.uint8);oy=(28-nh)//2;ox=(28-nw)//2
    out[oy:oy+nh,ox:ox+nw]=scaled
    return out

def segment(image):
    if image is None:return None,{'ok':False,'reason':'image_not_found'}
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image.copy()
    h0,w0=gray.shape;scale=min(1.,480/w0,160/h0)
    if scale<1:gray=cv2.resize(gray,(round(w0*scale),round(h0*scale)),interpolation=cv2.INTER_AREA)
    h,w=gray.shape
    if w<80 or h<20:return None,{'ok':False,'reason':'roi_too_small'}
    processed=enhance(gray)
    if processed is None:return None,{'ok':False,'reason':'insufficient_contrast'}
    is_two=w/h<3.0
    otsu,_=cv2.threshold(processed,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    for off in (0,-8,8,-16,16):
        threshold=int(np.clip(otsu+off,20,235))
        binary=(processed<=threshold).astype(np.uint8)
        n,_,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8)
        boxes=[]
        for x,y,bw,bh,area in stats[1:n]:
            if area<max(7,w*h//4500) or bw<2 or bw>w*.24:continue
            if x<=w*.025 or x+bw>=w*.975 or y<=h*.035 or y+bh>=h*.965:continue
            if is_two:
                if bh<h*.18 or bh>h*.48:continue
            else:
                if bh<h*.32 or bh>h*.85:continue
            boxes.append([int(x),int(y),int(x+bw),int(y+bh)])
        if not is_two:
            if len(boxes)!=8:continue
            boxes.sort(key=lambda b:b[0])
            centers=[(b[1]+b[3])/2 for b in boxes]
            if max(centers)-min(centers)>h*.18:continue
            rows=[boxes]
        else:
            if len(boxes) not in (8,9):continue
            by_y=sorted(boxes,key=lambda b:(b[1]+b[3])/2)
            centers=np.array([(b[1]+b[3])/2 for b in by_y]);gaps=np.diff(centers)
            cut=int(np.argmax(gaps))+1
            if gaps[cut-1]<h*.18 or cut not in (3,4) or len(boxes)-cut!=5:continue
            rows=[sorted(by_y[:cut],key=lambda b:b[0]),sorted(by_y[cut:],key=lambda b:b[0])]
        ordered=[box for row in rows for box in row]
        glyphs=np.stack([normalize(processed,b) for b in ordered])
        return glyphs,{'ok':True,'boxes':ordered,'rows':[len(r) for r in rows],
                       'threshold':threshold,'size':[w,h],'scale':scale}
    return None,{'ok':False,'reason':'cannot_find_expected_components',
                 'layout_guess':'two_rows' if is_two else 'one_row','size':[w,h]}

def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    inter=ix*iy;aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return inter/max(1,aa+bb-inter)

def prepare(split,root=DATA,out=OUT):
    records=json.loads((root/'manifest.json').read_text(encoding='utf-8'))['records']
    records=[r for r in records if r['split']==split];images=[];rows=[];counts=Counter()
    for i,row in enumerate(records):
        glyphs,info=segment(cv2.imread(str(root/row['file']),cv2.IMREAD_GRAYSCALE))
        start=len(images); count=0 if glyphs is None else len(glyphs)
        gold=row['box_annotations'];scale=info.get('scale',1.)
        original=[np.array(b)*scale for b in gold]
        aligned=count==len(gold) and all(iou(a,b)>=.25 for a,b in zip(info.get('boxes',[]),original))
        if split in ('train','calibration') and not aligned:
            glyphs=None;count=0
        if count:images.extend(glyphs)
        counts['total']+=1;counts['segmented']+=int(info['ok']);counts['aligned_posthoc']+=int(aligned)
        counts[row['layout']+'_total']+=1;counts[row['layout']+'_segmented']+=int(info['ok'])
        rows.append({'id':row['id'],'layout':row['layout'],'truth_posthoc':row['text_posthoc'],
                     'font_holdout':row['font_holdout'],'damage_count':len(row['damaged_positions']),
                     'start':start,'count':count,'segmentation':info,'aligned_posthoc':bool(aligned)})
        if (i+1)%500==0:print(json.dumps({'split':split,'done':i+1,'total':len(records)}),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    array=np.asarray(images,dtype=np.uint8).reshape(-1,28,28)
    np.savez_compressed(out/f'{split}_glyphs.npz',x=array)
    (out/f'{split}_rows.json').write_text(json.dumps(rows,ensure_ascii=False),encoding='utf-8')
    (out/f'{split}_segmentation.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(json.dumps({'split':split,'glyphs':len(array),**counts},indent=2),flush=True)

if __name__=='__main__':
    cv2.setNumThreads(1)
    p=argparse.ArgumentParser();p.add_argument('split',choices=('preview','train','validation','calibration','test'));args=p.parse_args()
    prepare(args.split,HERE/'preview_v4' if args.split=='preview' else DATA, HERE/'preview_v4_segments' if args.split=='preview' else OUT)
