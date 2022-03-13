import os
import pathlib
import numpy as np
import shutil
import tempfile
from utils import crop_image, prepare_data
from PIL import Image

ROOT = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
SAMPLE_IMAGES = ROOT / 'sample_images'

def test_it_crops_an_image_already_at_correct_size():
    im = np.ones(shape=(2, 2, 3))
    assert crop_image(im, 2).shape == (2, 2, 3)

def test_it_crops_an_image_at_scale_2_with_bigger_width():
    im = np.ones(shape=(3, 2, 3))
    assert crop_image(im, 2).shape == (2, 2, 3)

def test_it_crops_an_image_at_scale_2_with_bigger_height():
    im = np.ones(shape=(2, 3, 3))
    assert crop_image(im, 2).shape == (2, 2, 3)

def test_it_crops_an_image_at_scale_2_with_bigger_width_and_height():
    im = np.ones(shape=(3, 3, 3))
    assert crop_image(im, 2).shape == (2, 2, 3)


def test_it_crops_an_image_at_scale_4_with_bigger_width():
    im = np.ones(shape=(17, 4, 3))
    assert crop_image(im, 4).shape == (16, 4, 3)

def test_it_crops_an_image_at_scale_4_with_bigger_height():
    im = np.ones(shape=(4, 17, 3))
    assert crop_image(im, 4).shape == (4, 16, 3)

def test_it_crops_an_image_at_scale_4_with_bigger_width_and_height():
    im = np.ones(shape=(17, 17, 3))
    assert crop_image(im, 4).shape == (16, 16, 3)

def test_it_crops_an_image_at_scale_4_with_even_size():
    im = np.ones(shape=(18, 18, 3))
    assert crop_image(im, 4).shape == (16, 16, 3)

def test_it_prepares_data():
    with tempfile.TemporaryDirectory() as src:
        with tempfile.TemporaryDirectory() as dest:
            src = pathlib.Path(src)
            dest = pathlib.Path(dest)
            Image.fromarray(np.ones(dtype=np.uint8, shape=(16, 16, 3)) * 255).save(src / 'sample1.png')
            Image.fromarray(np.ones(dtype=np.uint8, shape=(17, 17, 3)) * 255).save(src / 'sample2.png')
            Image.fromarray(np.ones(dtype=np.uint8, shape=(18, 18, 3)) * 255).save(src / 'sample3.png')
            dirs = prepare_data([(src, dest)], scale=4)
            assert len(os.listdir(src)) == len(os.listdir(dest / 'hr'))
            assert np.array(Image.open(dest / 'hr' / 'sample1.png').convert('RGB')).shape == (16, 16, 3)
            assert np.array(Image.open(dest / 'hr' / 'sample2.png').convert('RGB')).shape == (16, 16, 3)
            assert np.array(Image.open(dest / 'hr' / 'sample3.png').convert('RGB')).shape == (16, 16, 3)
            assert np.array(Image.open(dest / 'lr' / 'sample1.png').convert('RGB')).shape == (4, 4, 3)
            assert np.array(Image.open(dest / 'lr' / 'sample2.png').convert('RGB')).shape == (4, 4, 3)
            assert np.array(Image.open(dest / 'lr' / 'sample3.png').convert('RGB')).shape == (4, 4, 3)
            assert dirs == [(dest / 'hr', dest / 'lr')]

def test_it_prepares_data_for_multiple_folders():
    with tempfile.TemporaryDirectory() as train_src:
        with tempfile.TemporaryDirectory() as valid_src:
            with tempfile.TemporaryDirectory() as dest:
                train_src = pathlib.Path(train_src)
                valid_src = pathlib.Path(valid_src)
                dest = pathlib.Path(dest)

                Image.fromarray(np.ones(dtype=np.uint8, shape=(16, 16, 3)) * 255).save(train_src / 'train.png')
                Image.fromarray(np.ones(dtype=np.uint8, shape=(12, 12, 3)) * 255).save(valid_src / 'valid.png')
                dirs = prepare_data([
                    (train_src, dest / 'train'),
                    (valid_src, dest / 'valid'),
                ], scale=4)
                assert len(os.listdir(train_src)) == len(os.listdir(dest / 'train/hr'))
                assert len(os.listdir(valid_src)) == len(os.listdir(dest / 'valid/hr'))
                assert np.array(Image.open(dest / 'train/hr/train.png').convert('RGB')).shape == (16, 16, 3)
                assert np.array(Image.open(dest / 'valid/hr/valid.png').convert('RGB')).shape == (12, 12, 3)
