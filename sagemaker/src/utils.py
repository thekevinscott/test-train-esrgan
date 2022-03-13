from tqdm import tqdm
import os
import numpy as np
import math
from PIL import Image
from io import BytesIO

def compress_image(im, quality=1):
    out = BytesIO()
    im.save(out, format="JPEG", quality=quality)
    out.seek(0)
    return Image.open(out)


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


def prepare_data(dirs, scale=None, quality=50):
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
    for src, dest in dirs:
        HR = (dest / 'hr')
        LR = (dest / 'lr')
        HR.mkdir(exist_ok=True, parents=True)
        LR.mkdir(exist_ok=True, parents=True)
        dest_dirs.append((HR, LR))
        for imgname in tqdm(os.listdir(src), desc=f'Processing files in {src}'):
            im = Image.open(src / imgname).convert('RGB')
            cropped_im = crop_image(np.array(im), scale)
            im = Image.fromarray(cropped_im)
            im.save(dest / 'hr' / imgname)

            smallim = im.resize((int(im.size[0] / scale),int(im.size[1] / scale)), Image.ANTIALIAS)
            print('**** Compression quality not (yet) being applied')
            # smallim = compress_image(smallim, quality=quality)
            smallim.save(dest / 'lr' / imgname)

    return dest_dirs
