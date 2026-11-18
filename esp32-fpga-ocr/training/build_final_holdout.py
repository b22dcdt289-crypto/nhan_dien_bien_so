"""Render a fresh test set only after the robust model and policy are frozen."""
from pathlib import Path
import json
import build_data as b

HERE=Path(__file__).resolve().parent
OUT=HERE/'dataset_final_holdout'
if __name__=='__main__':
    if not (HERE/'model31_robust/calibration_freeze.json').exists():
        raise RuntimeError('Freeze model and calibration before generating holdout')
    previous=b.prior_identities
    def excluded():
        ids=previous()
        for p in (HERE/'dataset_v3/manifest.json',HERE/'preview_v5/manifest.json'):
            ids.update(r['text_posthoc'] for r in json.loads(p.read_text(encoding='utf-8'))['records'])
        return ids
    b.prior_identities=excluded
    b.HOLDOUT_FONTS=[str(Path('C:/Windows/Fonts')/n) for n in ('lucon.ttf','cour.ttf','OCRAEXT.TTF','trebuc.ttf')]
    if not all(Path(p).exists() for p in b.HOLDOUT_FONTS):raise FileNotFoundError('Missing held-out font')
    b.SPLITS={'test':(250,250,300)}
    b.generate(OUT)
