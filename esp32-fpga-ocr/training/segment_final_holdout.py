"""Apply the frozen pixel-only segmentation to the final holdout."""
from pathlib import Path
import cv2
from segment import prepare
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    cv2.setNumThreads(1)
    prepare('test',HERE/'dataset_final_holdout',HERE/'segments_final_holdout')
