import pathlib
from smart_open import open
from tqdm import tqdm
import os
import numpy as np
import math
from PIL import Image, ImageEnhance
from io import BytesIO
import imageio

def write_image(dest: str, im: np.ndarray):
    imageio.imwrite(dest, (im * 255).astype(np.uint8))

def compress_image(im: np.ndarray, quality=1) -> np.ndarray:
    im = Image.fromarray((im * 255).astype(np.uint8))
    out = BytesIO()
    im.save(out, format="JPEG", quality=quality)
    out.seek(0)
    return np.asarray(Image.open(out)) / 255.0

def sharpen_image(im: np.ndarray, amount=1) -> np.ndarray:
    im = Image.fromarray((im * 255).astype(np.uint8))
    return np.asarray(ImageEnhance.Sharpness(im).enhance(amount)) / 255.0


def crop_image(im, scale):
    ''' the goal of this function is to get images that scale to integers when scaled down'''
    height, width, _ = im.shape

    resulting_height = int(height / scale)
    resulting_width = int(width / scale)

    upscaled_height = resulting_height * scale
    upscaled_width = resulting_width * scale

    height_diff = height - upscaled_height
    width_diff = width - upscaled_width

    start_height = math.floor(height_diff / 2)
    end_height = height - (height_diff - start_height)

    start_width = math.floor(width_diff / 2)
    end_width = width - (width_diff - start_width)
    
    return im[start_height:end_height, start_width:end_width, :]

def get_files(folder):
    files = []
    for file in os.listdir(folder):
        if file.endswith('.gif') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
            files.append(file)
    return files


def prepare_data(dirs, scale=None):
    '''
    Provide it a list of tuples. 
    
    Each tuple is:

    (
        src - the directory containing the HR photos
        dest - the directory to which BOTH the originals (cropped) and the downscaled images will be written to
    )
    '''
    if scale is None:
        raise Exception('Provide scale')

    dest_dirs = []
    errors = []
    for src, dest in dirs:
        src = pathlib.Path(src)
        dest = pathlib.Path(dest)
        if str(dest).startswith('s3://'):
            HR = f'{dest}/hr'
            LR = f'{dest}/lr'
            dest_dirs.append((HR, LR))
            for imgname in tqdm(get_files(src), desc=f'Processing files in {src}'):
                try:
                    im = Image.open(src / imgname).convert('RGB')
                    cropped_im = crop_image(np.array(im), scale)
                    im = Image.fromarray(cropped_im)
                    smallim = im.resize((int(im.size[0] / scale),int(im.size[1] / scale)), Image.ANTIALIAS)

                    with open(f'{HR}/{imgname}', 'wb') as f:
                        im.save(f)
                    with open(f'{LR}/{imgname}', 'wb') as f:
                        smallim.save(f)
                except Exception as e:
                    raise Exception(src / imgname)

        else:
            HR = (dest / 'hr')
            LR = (dest / 'lr')
            HR.mkdir(exist_ok=True, parents=True)
            LR.mkdir(exist_ok=True, parents=True)
            dest_dirs.append((HR, LR))
            for imgname in tqdm(get_files(src), desc=f'Processing files in {src}'):
                try:
                    im = Image.open(src / imgname).convert('RGB')
                    cropped_im = crop_image(np.array(im), scale)
                    im = Image.fromarray(cropped_im)
                    im.save(dest / 'hr' / imgname)

                    smallim = im.resize((int(im.size[0] / scale),int(im.size[1] / scale)), Image.ANTIALIAS)
                    # smallim = compress_image(smallim, quality=quality)
                    smallim.save(dest / 'lr' / imgname)
                except Exception as e:
                    errors.append((src / imgname, e))
    print('The following errors:', errors)

    return dest_dirs
