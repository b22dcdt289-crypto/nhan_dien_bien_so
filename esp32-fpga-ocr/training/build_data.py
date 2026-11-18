"""Synthetic, frontal one/two-row plate corpus; no camera images in training.

Each plate identity is assigned before rendering and never crosses splits.
Glyph boxes are annotations for supervision and post-hoc segmentation audit,
not inputs to the runtime recognizer. O is an explicit OCR stress class; its
presence here does not imply all such strings are officially issued plates.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
UART=HERE.parent
BASE='0123456789ABCDEFGHKLMNPSTUVXYZ'
CLASSES=BASE+'O'
FONT_ROOT=Path(r'D:\TOTNGHIEP\license_plate_ai\data\frontal_reference_fonts')
TRAIN_FONTS=[str(FONT_ROOT/n) for n in ('D-DINCondensed-Bold.ttf','D-DINCondensed.ttf','D-DIN.ttf','Anton-Regular.ttf','BebasNeue-Regular.ttf')]
TRAIN_FONTS += [str(Path('C:/Windows/Fonts')/n) for n in ('arial.ttf','arialbd.ttf','arialn.ttf','tahoma.ttf','segoeui.ttf','bahnschrift.ttf')]
TRAIN_FONTS += [str(Path('C:/Windows/Fonts')/n) for n in ('AGENCYB.TTF','AGENCYR.TTF','calibri.ttf','calibrib.ttf')]
HOLDOUT_FONTS=[str(Path('C:/Windows/Fonts')/n) for n in ('FRADM.TTF','FRADMCN.TTF','consola.ttf','consolab.ttf')]
TEMPLATE_ROOT=Path(r'D:\TOTNGHIEP\license_plate_ai\data\photoreal_straight_20261023_v2\templates')
TRAIN_FONTS=[x for x in TRAIN_FONTS if Path(x).exists()]
HOLDOUT_FONTS=[x for x in HOLDOUT_FONTS if Path(x).exists()]
LAYOUTS=('car_one_row','car_two_row','motorcycle_two_row')
SPLITS={'train':(600,900,1200),'validation':(120,140,200),
        'calibration':(100,120,160),'test':(250,250,300)}

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

@lru_cache(maxsize=8)
def metal_template(name):
    raw=cv2.imread(str(TEMPLATE_ROOT/f'metal_{name}.png'),cv2.IMREAD_GRAYSCALE)
    if raw is None:raise FileNotFoundError(f'metal_{name}.png')
    band=raw[:,raw.shape[1]//10:-raw.shape[1]//10]
    dark=np.flatnonzero(np.quantile(band,.75,axis=1)<95)
    if not len(dark):raise ValueError('Cannot locate plate holder')
    return raw[int(dark[0]):int(dark[-1])+1]

@lru_cache(maxsize=32)
def load_font(path): return ImageFont.truetype(path,360)
@lru_cache(maxsize=400)
def mask_for_char(char,path):
    font=load_font(path); box=font.getbbox(char)
    image=Image.new('L',(box[2]-box[0]+20,box[3]-box[1]+20),0)
    ImageDraw.Draw(image).text((10-box[0],10-box[1]),char,font=font,fill=255)
    return image.crop(image.getbbox())

def draw_line(text,font_path,rng,arr,region,per_char):
    x0,y0,x1,y1=region; W=x1-x0; H=y1-y0
    glyph_h=round(H*rng.uniform(.74,.86)); gap=round(glyph_h*rng.uniform(.07,.16)); items=[]
    for ch in text:
        if ch=='-': gw=max(3,round(glyph_h*.23));gh=max(2,round(glyph_h*.07));
        elif ch=='.': gw=gh=max(2,round(glyph_h*.08))
        else:
            native=mask_for_char(ch,font_path);gh=glyph_h
            gw=max(3,round(gh*(native.width/native.height)*rng.uniform(.94,1.06)))
        items.append([ch,gw,gh])
    span=sum(z[1] for z in items)+gap*(len(items)-1)
    shrink=min(1.0,W*.92/max(span,1));gap=round(gap*shrink)
    for z in items:z[1]=max(2,round(z[1]*shrink))
    span=sum(z[1] for z in items)+gap*(len(items)-1)
    cursor=x0+(W-span)//2; top=y0+(H-glyph_h)//2
    for ch,gw,gh in items:
        gx=round(cursor)
        if ch in '-.':
            gy=top+round(glyph_h*(.50 if ch=='-' else .91)); mask=np.full((gh,gw),255,np.uint8)
        else:
            gy=top;mask=np.asarray(mask_for_char(ch,font_path).resize((gw,gh),Image.Resampling.LANCZOS))
        arr[gy:gy+gh,gx:gx+gw]=np.maximum(arr[gy:gy+gh,gx:gx+gw],mask)
        if ch not in '-.': per_char.append([gx,gy,gx+gw,gy+gh])
        cursor+=gw+gap

def render(task):
    root,split,layout,idx,text,seed,font_path,holdout=task
    cv2.setNumThreads(1);rng=random.Random(seed);noise=np.random.default_rng(seed)
    if layout=='car_one_row': W,H=480,120; rows=[text[:3]+'-'+text[3:6]+'.'+text[6:]]
    else:
        W,H=(380,280) if layout=='motorcycle_two_row' else (400,200)
        cut=4 if layout=='motorcycle_two_row' else 3
        top=text[:2]+'-'+text[2:cut] if layout=='motorcycle_two_row' else text[:cut]
        rows=[top,text[cut:cut+3]+'.'+text[cut+3:]]
    alpha=np.zeros((H,W),np.uint8);boxes=[]
    if len(rows)==1:draw_line(rows[0],font_path,rng,alpha,(15,12,W-15,H-12),boxes)
    else:
        margin=round(H*.15);gap=round(H*.025)
        draw_line(rows[0],font_path,rng,alpha,(20,margin,W-20,H//2-gap),boxes)
        draw_line(rows[1],font_path,rng,alpha,(20,H//2+gap,W-20,H-margin),boxes)
    assert len(boxes)==len(text),(layout,text,boxes)
    # Mild interrupted strokes are applied independently of character identity.
    damaged=[]
    for j,(x0,y0,x1,y1) in enumerate(boxes):
        if rng.random()<(.19 if split=='train' else .22):
            bw=x1-x0;bh=y1-y0
            rw=max(1,round(bw*rng.uniform(.06,.18)));rh=max(1,round(bh*rng.uniform(.04,.14)))
            xx=rng.randrange(x0,max(x0+1,x1-rw+1)); yy=rng.randrange(y0,max(y0+1,y1-rh+1))
            alpha[yy:yy+rh,xx:xx+rw]=(alpha[yy:yy+rh,xx:xx+rw].astype(np.float32)*rng.uniform(.1,.6)).astype(np.uint8)
            damaged.append(j)
    yy,xx=np.mgrid[:H,:W]
    coarse=cv2.resize(noise.normal(0,1,(8,15)).astype(np.float32),(W,H),interpolation=cv2.INTER_CUBIC)
    is_metal=rng.random()<.78
    template_id=('AB'[rng.randrange(2)] if split in ('train','preview') else 'C' if split in ('validation','calibration') else 'D') if is_metal else None
    if is_metal:
        base=cv2.resize(metal_template(template_id),(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)
        base+=coarse*rng.uniform(0,2)+(xx/W-.5)*rng.uniform(-9,9)
    else:
        base=rng.uniform(202,246)+coarse*rng.uniform(0,3)+(xx/W-.5)*rng.uniform(-22,22)+(yy/H-.5)*rng.uniform(-12,12)
        cv2.rectangle(base,(1,1),(W-2,H-2),rng.randrange(8,42),rng.choice([3,4,5]))
        if rng.random()<.5:cv2.rectangle(base,(6,6),(W-7,H-7),rng.randrange(105,178),1)
    base+=noise.normal(0,rng.uniform(.2,1.5),(H,W)).astype(np.float32)
    base=np.clip(base,0,255)
    a=alpha.astype(np.float32)/255.
    # Thin metal lip / contact shadow, plus tonal variability of black paint.
    dil=cv2.dilate(alpha,np.ones((3,3),np.uint8)).astype(np.float32)/255.
    outer=np.maximum(0,dil-a)
    base=base*(1-.20*outer)
    paint=rng.uniform(9,46)+noise.normal(0,rng.uniform(.4,2.0),(H,W))
    if is_metal:
        # A highlight at the upper-left edge gives a restrained embossed lip.
        up=np.zeros_like(a);up[1:,1:]=a[:-1,:-1]
        lip=np.maximum(0,a-up)
        paint+=lip*rng.uniform(18,42)
    image=base*(1-a)+paint*a
    image=np.clip(image,0,255).astype(np.uint8)
    blur=rng.uniform(.18,.85 if split=='train' else 1.0)
    image=cv2.GaussianBlur(image,(0,0),blur)
    contrast=rng.uniform(.76,1.05);bright=rng.uniform(-9,9)
    image=np.clip((image.astype(np.float32)-128)*contrast+128+bright,0,255).astype(np.uint8)
    quality=rng.randrange(68,97);ok,enc=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,quality])
    if not ok:raise RuntimeError('JPEG encoding failed')
    path=Path(root)/'images'/split/layout/f'{idx:05d}.jpg';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(enc.tobytes())
    return {'id':f'{split}_{layout}_{idx:05d}','split':split,'layout':layout,'text_posthoc':text,
            'file':path.relative_to(root).as_posix(),'font':font_path,'font_holdout':holdout,'seed':seed,
            'box_annotations':boxes,'size':[W,H],'damaged_positions':damaged,'blur_sigma':blur,
            'contrast':contrast,'jpeg_quality':quality,'template':template_id or 'screen_flat',
            'sha256':sha(path),'synthetic':True}

def prior_identities():
    excluded=set()  # Supply the held-out real plate identities before generating data.
    for p in UART.glob('one_row_recovery_20261101*/manifest.json'):
        data=json.loads(p.read_text(encoding='utf-8'))
        excluded.update(r.get('text','') for r in data.get('records',[]))
    for p in (HERE/'dataset_v2/manifest.json',HERE/'preview_v4/manifest.json'):
        if p.exists():excluded.update(r.get('text_posthoc','') for r in json.loads(p.read_text(encoding='utf-8')).get('records',[]))
    return excluded

def make_tasks(root,counts,preview=False):
    if not TRAIN_FONTS or not HOLDOUT_FONTS:raise RuntimeError('Missing fonts for separated test')
    rng=random.Random(20261117129);used=prior_identities();tasks=[]; assignments=[]
    for split,quantities in counts.items():
        for layout,count in zip(LAYOUTS,quantities):
            for i in range(count):
                while True:
                    # O is deliberately included as a distinct OCR stress label.
                    letter='O' if rng.random()<.20 else rng.choice(BASE[10:])
                    second=(rng.choice(BASE[10:]+'O') if i%2==0 else str(rng.randrange(10))) if layout=='motorcycle_two_row' else ''
                    prefix=f'{rng.randrange(10,100):02d}'+letter+second
                    text=prefix+''.join(str(rng.randrange(10)) for _ in range(5))
                    if text not in used:used.add(text);break
                # Independent of motorcycle letter/digit variant (both parities occur in each font group).
                holdout=split=='test' and i%4 in (0,1)
                font_path=rng.choice(HOLDOUT_FONTS if holdout else TRAIN_FONTS)
                seed=rng.randrange(2**31)
                tasks.append((str(root),split,layout,i,text,seed,font_path,holdout))
                assignments.append({'split':split,'layout':layout,'id':i,'text':text,'seed':seed})
    return tasks,assignments

def generate(root,preview=False):
    if (root/'manifest.json').exists():raise FileExistsError('Frozen dataset already exists')
    counts={'preview':(4,4,4)} if preview else SPLITS
    tasks,assignments=make_tasks(root,counts,preview)
    root.mkdir(parents=True,exist_ok=True);dump(root/'partition_before_render.json',{'counts':counts,'assignments':assignments})
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(render,tasks,chunksize=8),1):
            rows.append(row)
            if i%200==0:print(json.dumps({'rendered':i,'total':len(tasks),'seconds':round(time.perf_counter()-start,1)}),flush=True)
    groups={split:{r['text_posthoc'] for r in rows if r['split']==split} for split in counts}
    assert all(not groups[a]&groups[b] for a in groups for b in groups if a!=b)
    meta={'version':'multirow_20261117_v1','synthetic_only':True,'camera_images_in_training':0,
          'classes':CLASSES,'front_facing':True,'perspective':False,'split_counts':counts,
          'font_assignment_independent_of_class':True,'train_fonts':TRAIN_FONTS,'test_holdout_fonts':HOLDOUT_FONTS,
          'o_class':'diagnostic OCR stress; not a claim of valid plate registration',
          'split_identity_overlap':0,'records':rows}
    dump(root/'manifest.json',meta)
    print(json.dumps({'output':str(root),'plates':len(rows),'seconds':round(time.perf_counter()-start,1)},ensure_ascii=False),flush=True)

def contact_sheet(root):
    rows=json.loads((root/'manifest.json').read_text(encoding='utf-8'))['records'];width=960;cellh=250
    canvas=Image.new('RGB',(width,((len(rows)+1)//2)*cellh),'white');draw=ImageDraw.Draw(canvas)
    f=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for i,row in enumerate(rows):
        im=Image.open(root/row['file']).convert('RGB');im.thumbnail((450,205))
        x=12+(i%2)*480;y=8+(i//2)*cellh;canvas.paste(im,(x,y+23))
        draw.text((x,y),row['layout']+' | '+row['text_posthoc'],font=f,fill=(25,25,25))
    canvas.save(root/'preview.jpg',quality=90)
    print(str(root/'preview.jpg'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('preview','generate'));args=parser.parse_args()
    out=HERE/('preview_v5' if args.stage=='preview' else 'dataset_v3')
    generate(out,preview=args.stage=='preview')
    if args.stage=='preview':contact_sheet(out)
