"""Generate fresh identities with diverse fonts; no observed image is replayed."""
from pathlib import Path
import json
import build_data as b
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    prior=b.prior_identities
    def excluded():
        ids=prior()
        for p in (HERE/'dataset_v3/manifest.json',HERE/'dataset_final_holdout/manifest.json',HERE/'preview_v5/manifest.json'):
            ids.update(r['text_posthoc'] for r in json.loads(p.read_text(encoding='utf-8'))['records'])
        return ids
    b.prior_identities=excluded
    extra=('consola.ttf','consolab.ttf','OCRAEXT.TTF','lucon.ttf','cour.ttf','trebuc.ttf','FRADM.TTF','FRADMCN.TTF')
    b.TRAIN_FONTS+= [str(Path('C:/Windows/Fonts')/n) for n in extra if (Path('C:/Windows/Fonts')/n).exists()]
    b.HOLDOUT_FONTS=[str(Path('C:/Windows/Fonts')/n) for n in ('impact.ttf','gadugi.ttf','GOTHIC.TTF','verdana.ttf')]
    if not all(Path(p).exists() for p in b.HOLDOUT_FONTS):raise FileNotFoundError('Missing held-out font')
    b.SPLITS={'train':(600,900,1200),'validation':(120,140,200),'calibration':(100,120,160),'test':(250,250,300)}
    b.generate(HERE/'dataset_v4')
