from fastai.vision.all import *

__all__ = ['MaskBlockCustom', 'WSISEG_STATS', 'get_segmentation_dls']

WSISEG_STATS = [0.3114, 0.3166, 0.3946],[0.2587, 0.2598, 0.2958]

set_seed(2021)


def MaskBlockCustom(codes=None, div=100):
    "A `TransformBlock` for segmentation masks, potentially with `codes`"
    return TransformBlock(type_tfms=PILMask.create, 
                          item_tfms=AddMaskCodes(codes=codes), 
                          batch_tfms=IntToFloatTensor(div_mask=div))

def _get_segmentation_block(label_func, bs, val_pct, crop_size=256, div_factor=255, pad_mode='zeros'):
    "Get segmentation datablock"
    tfms = aug_transforms(flip_vert=True, max_zoom=1.5, pad_mode=pad_mode)
    block = DataBlock(blocks=(ImageBlock, MaskBlockCustom(div=div_factor)), 
                      get_items=get_image_files,
                      get_y=label_func,
                      item_tfms=RandomCrop(crop_size) if crop_size else None,
                      batch_tfms=tfms + [Normalize.from_stats(*WSISEG_STATS)],
                      splitter=RandomSplitter(val_pct, seed=2021))
    return block

def get_segmentation_dls(path, label_func, crop_size=128, bs=16, val_pct=0.2, div_factor=255, pad_mode='zeros'):
    "Get a dataloader with correspoding image sizes, set size=None for full size"
    block = _get_segmentation_block(label_func, bs, val_pct, crop_size, div_factor, pad_mode)
    return block.dataloaders(path, bs=bs)