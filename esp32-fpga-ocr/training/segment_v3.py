"""Apply the frozen pixel-only segmenter to the refined synthetic corpus."""
import argparse
from pathlib import Path
import cv2
from segment import prepare

HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    cv2.setNumThreads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('split',choices=('preview','train','validation','calibration','test'))
    a=parser.parse_args()
    prepare(a.split,HERE/'preview_v5' if a.split=='preview' else HERE/'dataset_v3',
            HERE/'preview_v5_segments' if a.split=='preview' else HERE/'segments_v3')
