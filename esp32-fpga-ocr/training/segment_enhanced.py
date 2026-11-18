"""PC-only segmentation: remove a thin frame band if the initial parse fails."""
from pathlib import Path
import argparse
import cv2
import numpy as np
import segment as base
HERE=Path(__file__).resolve().parent

def segment(image):
    if image is None:return None,{'ok':False,'reason':'image_not_found'}
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image.copy()
    h0,w0=gray.shape;scale=min(1.,480/w0,160/h0)
    if scale<1:gray=cv2.resize(gray,(round(w0*scale),round(h0*scale)),interpolation=cv2.INTER_AREA)
    h,w=gray.shape
    if w<80 or h<20:return None,{'ok':False,'reason':'roi_too_small'}
    processed=base.enhance(gray)
    if processed is None:return None,{'ok':False,'reason':'insufficient_contrast'}
    two=w/h<3.;otsu,_=cv2.threshold(processed,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    for off in (0,-8,8,-16,16):
      threshold=int(np.clip(otsu+off,20,235));original=(processed<=threshold).astype(np.uint8)
      for band in ((0.,.08) if two else (0.,.08,.12,.16)):
        binary=original.copy()
        if band:
            border=int(h*band);binary[:border,:]=0;binary[h-border:,:]=0
        n,_,stats,_=cv2.connectedComponentsWithStats(binary,connectivity=8);boxes=[]
        for x,y,bw,bh,area in stats[1:n]:
            if area<max(7,w*h//4500) or bw<2 or bw>w*.24:continue
            if x<=w*.025 or x+bw>=w*.975 or y<=h*.035 or y+bh>=h*.965:continue
            if two:
                if bh<h*.18 or bh>h*.48:continue
            else:
                if bh<h*.32 or bh>h*.85:continue
            boxes.append([int(x),int(y),int(x+bw),int(y+bh)])
        if not two:
            if len(boxes)!=8:continue
            boxes.sort(key=lambda b:b[0]);centers=[(b[1]+b[3])/2 for b in boxes]
            if max(centers)-min(centers)>h*.18:continue
            rows=[boxes]
        else:
            if len(boxes) not in (8,9):continue
            ys=sorted(boxes,key=lambda b:(b[1]+b[3])/2);centers=np.array([(b[1]+b[3])/2 for b in ys]);gaps=np.diff(centers)
            cut=int(np.argmax(gaps))+1
            if gaps[cut-1]<h*.18 or cut not in (3,4) or len(boxes)-cut!=5:continue
            rows=[sorted(ys[:cut],key=lambda b:b[0]),sorted(ys[cut:],key=lambda b:b[0])]
        ordered=[b for row in rows for b in row];glyphs=np.stack([base.normalize(processed,b) for b in ordered])
        return glyphs,{'ok':True,'boxes':ordered,'rows':[len(r) for r in rows],'threshold':threshold,
                       'size':[w,h],'scale':scale,'removed_border_fraction':band}
    return None,{'ok':False,'reason':'cannot_find_expected_components','layout_guess':'two_rows' if two else 'one_row','size':[w,h]}

if __name__=='__main__':
    cv2.setNumThreads(1);p=argparse.ArgumentParser();p.add_argument('split',choices=('train','validation','calibration','test'))
    p.add_argument('--development',action='store_true');a=p.parse_args();base.segment=segment
    root=HERE/'dataset_final_holdout' if a.development else HERE/'dataset_v4'
    out=HERE/'dev_enhanced_segments' if a.development else HERE/'segments_v4'
    base.prepare(a.split,root,out)
