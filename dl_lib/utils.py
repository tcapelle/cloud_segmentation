import shutil
from pathlib import Path
import torch
from fastcore.utils import run
from PIL import Image
from fastai.vision.utils import resize_images

__all__ = ['is_gpu', 'show_gpus', 'download_wsiseg', 'resize_dataset', 'get_wsiseg']

def is_gpu(device=0):
    "check if gpu is available"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        show_gpus()
        return True
    else: 
        return False

def show_gpus():
    "Show available cuda devices"
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        devs = [torch.cuda.get_device_properties(n) for n in range(n_devices)]
        for i, dev in enumerate(devs):
            mem_gb = dev.total_memory / (1024*1024*1e3)
            dev_name = (dev.name + ',')
            if torch.cuda.current_device()==i:
                print(f'{i}: *{dev_name:24s} {mem_gb:4.1f}GB, tensor_cores={dev.multi_processor_count}')
            else:
                print(f'{i}: {dev_name:24s} {mem_gb:4.1f}GB, tensor_cores={dev.multi_processor_count}')

    else:
        print('No GPU found')

def download_wsiseg(dest_path, force=False):
    "Download WSISEG dataset at dest"
    dest_path = Path(dest_path)
    if dest_path.exists(): 
        if force:
            shutil.rmtree(dest_path)
        else:
            return
    if not dest_path.exists():
        print(f'Cloning repo WSISEG into {dest_path}')
        code = run(f'git clone https://github.com/CV-Application/WSISEG-Database {dest_path}')


def fix_dataset(path):
    "Fix dataset filenames"
    image_path = (path/'whole sky images').rename(path/'images')
    annotation_path = (path/'annotation').rename(path/'masks')
    print('Fixing image extenstions')
    for folder in [image_path, annotation_path]:
        for im in folder.ls():
            if not im.name.endswith('.png'):
                im.rename(im.with_suffix('.png'))
    
    print('delete .git folder')
    shutil.rmtree(path/'.git')


def resize_dataset(path, sizes = [128, 256]):
    "resize dataset to sizes"
    image_path = path/'images'
    annotation_path = path/'masks'
    for size in sizes:
        if not image_path.with_name(f'images{size}').exists():
            print(f'Resizing images to {size}')
            resize_images(image_path, dest=image_path.with_name(f'images{size}'), resume=True, max_size=size) 
            resize_images(annotation_path, dest=annotation_path.with_name(f'masks{size}'), n_channels=1, 
                        resample=Image.NEAREST, resume=False, max_size=size)
                
def get_wsiseg(path, force=False, sizes=None):
    "Download and prepare the dataset"
    path = Path(path)

    download_wsiseg(path, force)
    fix_dataset(path)

    if isinstance(sizes, (list, tuple)):
        resize_dataset(path, sizes=sizes)
    
