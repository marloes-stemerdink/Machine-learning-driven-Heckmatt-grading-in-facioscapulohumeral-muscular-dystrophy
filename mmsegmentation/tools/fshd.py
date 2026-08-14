## FSHD dataset registry
from mmengine.logging import print_log

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

classes = ('background',
        'muscle')

paletteFSHD = [
        (0, 0, 0),       # black
    (128, 0, 128),   # purple
    ]  
        
@DATASETS.register_module()
class FSHD(BaseSegDataset):
    METAINFO = dict(classes = classes, palette = paletteFSHD)
    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png',
                        seg_map_suffix='.png',
                        reduce_zero_label = False,
                        ignore_index=255,
                        **kwargs)