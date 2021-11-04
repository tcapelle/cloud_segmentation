from fastai.vision.all import *
from dl_lib.data import get_segmentation_dls
from dl_lib.utils import is_gpu
from dl_lib.losses import CombinedLoss, foreground_acc

__all__ = ['swimseg_label_func', 'pretrain_swimseg']


def swimseg_label_func(fn): 
    return (fn.parent.parent/'GTmaps')/(fn.with_suffix('').name + '_GT.png')

def _pretrain_swimseg_stage(path, model, loss_func, metrics, splitter=None, crop_szie=256, bs=32, fp16=True, **kwargs):
    "Train model on swimseg patches for pretraining"
    dls = get_segmentation_dls(path=path, 
                               label_func=swimseg_label_func, 
                               crop_size=256,
                               bs=32,
                               div_factor=255, 
                               pad_mode='reflection')

    learn = Learner(dls, 
                    model, 
                    splitter=ifnone(splitter, trainable_params),
                    metrics=metrics, 
                    loss_func=loss_func,
                    opt_func=ranger,
                    path = Path.cwd().parent,
                    model_dir = 'model_checkpoints',
                    **kwargs)
    if fp16: 
        learn = learn.to_fp16()

    learn.fit_flat_cos(5, 1e-3)
    return learn

@delegates(Learner)
def pretrain_swimseg(path, 
                     model, 
                     splitter=None, 
                     loss_func = CombinedLoss(axis=1, alpha=0.4),
                     metrics=[foreground_acc, DiceMulti(axis=1)],
                     bs=32, 
                     fp16=True,
                     fine_tune=3,
                     **kwargs):
    """Pretrain swimseg model incrementally on 256, then full size images.
    path:  Path to the folder containing the swimseg swimseg dataset
    model: A segmentation model with binary output (2 classes)
    splitter (optional): A model split for fastai incremental training.
    fp16: Use mixed precision training
    
    returns: A fastai learner object.
    
    You will need to modify the trained model's head to use with our 4 or more
    classes data"""
    learn = _pretrain_swimseg_stage(path, model, loss_func, metrics, splitter, crop_szie=256, bs=bs, 
                                    fp16=fp16, **kwargs)

    if fine_tune:
        learn.dls = get_segmentation_dls(path=path, 
                                     label_func=swimseg_label_func, 
                                     crop_size=None,
                                     bs=bs//4,
                                     div_factor=255, 
                                     pad_mode='reflection')
        learn.fine_tune(fine_tune, 1e-5)
    return learn