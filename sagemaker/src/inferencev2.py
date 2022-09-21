from tempfile import NamedTemporaryFile, TemporaryFile
import base64
import hashlib
  
import json
from typing import List, Tuple, Any, Union, Optional
from IPython.display import display, HTML
import datetime
from tqdm import tqdm
import os
import yaml
import numpy as np
from numpy import ndarray
from types import SimpleNamespace
import pathlib
from PIL import Image
import statistics
from ISR.models import RRDN, RDN
import cv2
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from utils import crop_image
import boto3
from smart_open import open

ACCESS_KEY=''
SECRET_KEY=''
client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
)

class Cache:
    cache_dir = '.cache'
    def __init__(self, cache_dir = '.cache'):
        self.cache_dir = cache_dir
        try:
            os.mkdir(self.cache_dir)
        except:
            pass

    def get(self, trained_job):
        hash = trained_job.get_hash()
        try:
            with open(f'{self.cache_dir}/{hash}.json', 'r') as f:
                data = json.load(f)
                return EvaluationResult(data.get('output_dir'), data.get('dataset'), data)
        except:
            return None
    
    def set(self, trained_job, r):
        hash = trained_job.get_hash()
        filename = f'{self.cache_dir}/{hash}.json'
        # print(f'writing to {filename}')
        with open(filename, 'w') as f:
            f.write(json.dumps(r.dump()))
cache = Cache()

class EvaluationResult:
    output_dir = None
    dataset = None

    def __init__(self, output_dir=None, dataset=None, data={}):
        self.durations = data.get('durations', [])
        self.psnr = data.get('psnr', [])
        self.ssim = data.get('ssim', [])
        self.total_pixels = data.get('total_pixels', 0)
        self.srs = []
        self.lrs = []
        self.originals = []
        self.dataset = dataset
        if output_dir:
            self.output_dir = pathlib.Path(output_dir)
            (self.output_dir / 'upscaled').mkdir(exist_ok=True, parents=True)
            (self.output_dir / 'original').mkdir(exist_ok=True, parents=True)

    def add_evaluation(self, duration, num_pixels, image, sr_img, lr_img, original_img):
        self.durations.append(duration / datetime.timedelta(milliseconds=1))
        self.srs.append(sr_img)
        self.lrs.append(lr_img)
        self.originals.append(original_img)
        self.total_pixels += num_pixels
        psnr, ssim = self._calculate_metrics_(original_img, sr_img)
        self.psnr.append(psnr)
        self.ssim.append(ssim)
        if self.output_dir:
            sr_img.save(self.output_dir / 'upscaled' / image.name)
            lr_img.save(self.output_dir / 'original' / image.name)

    def _calculate_metrics_(self, original, upscaled):
        original_arr = np.array(original)
        upscaled_arr = np.array(upscaled)
        ssim = structural_similarity(original_arr, upscaled_arr, channel_axis=2, multichannel=3)
        original = to_cv2(original)
        upscaled = to_cv2(upscaled)

        psnr = cv2.PSNR(original, upscaled)
        return psnr, ssim

    def get_results(self):
        total_duration = 0
        for d in self.durations:
            total_duration += d
        
        return (
            total_duration, 
            float(total_duration) / self.total_pixels * 100,
            statistics.mean(self.psnr),
            statistics.median(self.psnr),
            statistics.mean(self.ssim),
            statistics.median(self.ssim),
        )

    def __str__(self) -> str:
        total_duration, time_per_100, psnr_mean, psnr_median, ssim_mean, ssim_median = self.get_results()
        
        return '\n'.join([
            f'Total duration in ms: {total_duration:9.4f}',
            f'Time per 100 pixels in ms: {time_per_100:9.4f}',
            f'PSNR mean: {psnr_mean:9.4f}',
            f'PSNR median: {psnr_median:9.4f}',
            f'SSIM mean: {ssim_mean:9.4f}',
            f'SSIM median: {ssim_median:9.4f}',
        ])

    def dump(self):
        return {
            'output_dir': str(self.output_dir),
            'durations': self.durations,
            'total_pixels': self.total_pixels,
            'ssim': self.ssim,
            'psnr': self.psnr,
            'dataset': str(self.dataset),
        }



    def get_sorted_metrics(self, metric, highest_first=True):
        metrics = None
        if metric == 'ssim':
            metrics = self.ssim
        elif metric == 'psnr':
            metrics = self.psnr
        if metrics is None:
            raise Exception(f'Unsupported metric specified: {metric}')
        metrics = [(m, i) for i, m in enumerate(metrics)]
        metrics.sort()
        if highest_first:
            metrics.reverse()
        return [m for m in metrics]

    def show_best_n(self, metric='psnr', n=5, width=20, row_size=3):
        indices = [i for _, i in self.get_sorted_metrics(metric=metric, highest_first=True)]
        self.show_n(indices[:n], width=width, row_size=row_size)

    def show_worst_n(self, metric='psnr', n=5, width=20, row_size=3):
        indices = [i for _, i in self.get_sorted_metrics(metric=metric, highest_first=False)]
        self.show_n(indices[:n], width=width, row_size=row_size)

    def show_n(self, _indices: Union[int, List[int]], n=5, width=20, row_size=3):
        indices = get_indices(_indices, len(self.lrs))
        _, axs = plt.subplots(n, 3, figsize=(width, row_size * len(indices)))
        for i in range(n):
            index = indices[i]
            lr_img = self.lrs[index]
            sr_img = self.srs[index]
            original_img = self.originals[index]

            axs[i, 0].imshow(lr_img)
            axs[i, 0].set_title('Low Resolution')
            axs[i, 1].imshow(sr_img)
            axs[i, 1].set_title('Upscaled')
            axs[i, 2].imshow(original_img)
            axs[i, 2].set_title('Original image')

        # for ax in axs.flat:
        #     ax.set(xlabel='x-label', ylabel='y-label')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
        plt.show(block=True)

class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def evaluate(self, model, scale, dataset=None, output_dir=None, n=None, show_tqdm=True):
        if dataset is None:
            dataset = self.dataset
        evaluation_result = EvaluationResult(output_dir, dataset)
        images = list(dataset.get_images())
        if n is not None:
            images = images[:n]
        
        pbar = None if show_tqdm is False else tqdm(total=len(images))
        for sample_image in images:
            evaluation = self._evaluate_image(model, sample_image, scale)
            evaluation_result.add_evaluation(*evaluation)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()
        
        return evaluation_result
        
    def _evaluate_image(self, model, image, scale):
        original, downscaled = image.get(scale)
        lr_img = np.array(downscaled)
        start = datetime.datetime.now()
        sr_img = Image.fromarray(model.predict(lr_img))
        sr_img_shape = np.array(sr_img).shape
        orig_shape = np.array(original).shape
        assert sr_img_shape == orig_shape, f'Shapes did not match, sr: {sr_img_shape}, orig: {orig_shape}'
        duration = datetime.datetime.now() - start

        lr_img = Image.fromarray(lr_img)
        return duration, image.get_num_pixels(), image, sr_img, lr_img, original

class SampleImage:
    name: str
    fullpath: pathlib.Path
    size: Tuple[int, int]
    array: ndarray
    def __init__(self, name, fullpath):
        self.fullpath = fullpath
        self.name = name

    def get(self, scale):
        im = Image.open(self.fullpath).convert('RGB')
        cropped_im = crop_image(np.array(im), scale)
        cropped_im_size = 72
        start_x = int((cropped_im.shape[0] - cropped_im_size) / 2)
        start_y = int((cropped_im.shape[1] - cropped_im_size) / 2)

        cropped_im = cropped_im[start_x:start_x+cropped_im_size, start_y:start_y+cropped_im_size]
        im = Image.fromarray(cropped_im)
        smallim = im.resize((int(im.size[0] / scale),int(im.size[1] / scale)), Image.ANTIALIAS)
        self.size = smallim.size
        return im, smallim

    def get_num_pixels(self):
        return self.size[0] * self.size[1]

class Dataset:
    root_folder: pathlib.Path

    def __init__(self, root):
        self.gather_dataset(root)

    def gather_dataset(self, root):
        self.root_folder = root
        self.files = [SampleImage(f, root / f) for f in os.listdir(self.root_folder)]

    def get_images(self):
        for file in self.files:
            filename = file.name
            if filename.endswith('jpg') or filename.endswith('jpeg') or filename.endswith('gif') or filename.endswith('png'):
                yield file

    def __str__(self):
        return str(self.root_folder)

class TrainedJob:
    root_folder: Union[pathlib.Path, str]
    run_folders = None
    generators: List[Union[pathlib.Path, str]]
    discriminators: List[Union[pathlib.Path, str]]
    configs: List[Union[pathlib.Path, str]]
    is_s3: bool = False
    bucket: Optional[str]

    def __init__(self, root: Union[str, pathlib.Path], keys: Optional[List[str]], bucket: Optional[str]) -> None:
        self.root_folder = root
        self.keys = keys
        self._parse_path = get_parse_path(root, keys, bucket)
        self.generators = []
        self.discriminators = []
        self.configs = []
        if keys is None:
            self.gather_weights(root)
        elif type(root)is str:
            # is s3
            self.is_s3 = True
            self.bucket = bucket
            self.gather_weights_from_s3(root, keys)
        else:
            raise Exception(f'Unexpected root provided to Training Job: {root}')

    def __str__(self):
        try:
            params = self.get_params()
            return f'{self.get_name()}: {params} : {self.get_compression()} | {self.get_sharpen()} | {self.get_dataset()}'
        except Exception as e:
            print(e)
            return 'error'

    def get_hash(self):
        base64_bytes = base64.b64encode(str(self).encode())
        encoded = base64_bytes.decode("ascii")
        result = hashlib.md5(encoded.encode())
        return result.hexdigest()

    def get_attribute_from_name(self, attb):
        for part in self.get_name().split('-'):
            if part.startswith(attb):
                return part.split(attb).pop()

    def get_vary_compression(self):
        return self.get_attribute_from_name('vary_c')

    def get_compression(self):
        return self.get_attribute_from_name('compress')

    def get_sharpen(self):
        return self.get_attribute_from_name('sharpen')

    def get_dataset(self):
        return self.get_attribute_from_name('data')

    def get_name(self):
        return str(self.root_folder).split('/').pop()

    def gather_weights_from_s3(self, root: str, keys: List[str]):
        for key in keys:
            self.__parse_file(key, key)

    def __parse_file(self, filename: Union[str, pathlib.Path], filepath: Union[str, pathlib.Path]):
        filename = str(filename)
        if filename.endswith('yml'):
            self.configs.append(filepath)
            # with open(filepath, 'r') as f:
        elif filename.endswith('hdf5'):
            if filename.startswith('rdn') or filename.startswith('rrdn'):
                self.generators.append(filepath)
            elif filename.startswith('srgan') or 'generator' in filename:
                self.discriminators.append(filepath)
        else:
            print(f'Unsupported file: {filename}')

    def gather_weights(self, root):
        self.run_folders = os.listdir(root)

        for run_folder in self.run_folders:
            run_folder = pathlib.Path(run_folder)
            with self._parse_path(root / run_folder) as parsed_path:
                for file in os.listdir(parsed_path):
                    filepath = pathlib.Path(parsed_path) / file
                    self.__parse_file(file, filepath)

    def _choose_best_epoch(self, arr):
        if len(arr) == 0:
            raise Exception('Cannot choose best epoch as array is empty')
        arr = [(int(str(a).split('epoch').pop().split('.hdf5')[0]), a) for a in arr]
        arr.sort()
        return arr.pop()[1]

    def get_generator(self, metric=None):
        if len(self.generators) == 0:
            raise Exception(f'No generators saved. keys were: {self.keys}')
        if metric:
            matching_generators = [g for g in self.generators if metric in str(g)]
            if len(matching_generators) == 0:
                raise Exception(f'No generators found for metric {metric}')
            return self._choose_best_epoch(matching_generators)
        return self._choose_best_epoch(self.generators)

    def get_discriminator(self, metric=None):
        if len(self.discriminators) == 0:
            raise Exception(f'No discriminators saved. keys were: {self.keys}')
        if metric:
            matching_discriminators = [g for g in self.discriminators if metric in str(g)]
            if len(matching_discriminators) == 0:
                raise Exception(f'No discriminators found for metric {metric}')
            return self._choose_best_epoch(matching_discriminators)
        return self._choose_best_epoch(self.discriminators)


    # def _parse_path(self, pathname: Union[str, pathlib.Path]):
    #     if self.is_s3:
    #         return f's3://{self.bucket}/{str(self.root_folder)}/{str(pathname)}'

    #     return f'{str(self.root_folder)}/{str(pathname)}'

    def get_model_name(self):
        return str(self.get_model()[1]).split('/').pop()
        pass

    def get_models(self, check_every_model=False, metric=None):
        if check_every_model is True:
            models = []
            for generator in self.generators:
                config = self.get_config()
                keys = config.keys()
                generator_config = config.get(list(keys)[0], {}).get('generator', {})
                params = SimpleNamespace(**generator_config.get('parameters', {}))
                model = get_model(generator_config.get('name'), params)
                with self._parse_path(generator) as parsed_path:
                    model.model.load_weights(str(parsed_path))
                models.append((model, generator))
            return models
        generator = self.get_generator(metric)
        config = self.get_config()
        keys = config.keys()
        generator_config = config.get(list(keys)[0], {}).get('generator', {})
        params = SimpleNamespace(**generator_config.get('parameters', {}))
        model = get_model(generator_config.get('name'), params)
        with self._parse_path(generator) as parsed_path:
            model.model.load_weights(str(parsed_path))
        return [(model, generator)]

    def get_params(self):
        config_all = self.get_config()
        keys = list(config_all.keys())
        config = None
        for key in keys:
            if config is None:
                config = json.dumps(config_all[key].get('generator').get('parameters'))
            elif config != json.dumps(config_all[key].get('generator').get('parameters')):
                raise Exception('Mismatch in configurations')
        return config

    def get_config(self):
        if len(self.configs) > 1:
            config = {}
            for _config in self.configs:
                config = {
                    **config,
                    **self._load_config_file(_config),
                }
            return config
        elif len(self.configs) == 1:
            config = self.configs[0]
            return self._load_config_file(config)
        else:
            if self.run_folders:
                for run_folder in self.run_folders:
                    with self._parse_path(run_folder) as parsed_path:
                        print(os.listdir(parsed_path))
            raise Exception(f'No config files found, {self.run_folders}')

    def _load_config_file(self, configpath):
        with self._parse_path(configpath) as parsed_path:
            with open(parsed_path, 'r', transport_params=dict(client=client)) as f:
                return yaml.load(f, Loader=yaml.BaseLoader)

def get_parse_path(root: Union[str, pathlib.Path], keys: Optional[List[str]], bucket: Optional[str]):
    class _parse_path():
        tempfile = None
        def __init__(self, pathname: Union[str, pathlib.Path]):
            self.tempfile = None
            self.pathname = pathname

        def __enter__(self):
            if bucket:
                fullpath = f's3://{bucket}/{root}/{self.pathname}'
                self.tempfile = NamedTemporaryFile()
                with open(fullpath, 'rb', transport_params=dict(client=client)) as f:
                    self.tempfile.write(f.read())
                    self.tempfile.seek(0)
                    return self.tempfile.name
            return self.pathname

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.tempfile:
                self.tempfile.close()

    return _parse_path

def get_model(model, args):
    C=args.C
    D=args.D
    G=args.G
    T=args.T
    G0=args.G0
    x=args.x
    if T == '0.0':
        T = '0'

    arch_params = {
        'C': int(C), 
        'D': int(D), 
        'G': int(G), 
        'G0':int(G0), 
        'T':int(T), 
        'x':int(x)
    }
    if model == 'rdn':
        return RDN(arch_params=arch_params)
    elif model == 'rrdn':
        return RRDN(arch_params=arch_params)
    raise Exception(f'No valid model found for {args.model}')

def get_indices(indices: Union[int, List[int]], total: int) -> List[int]:
    if type(indices) is list:
        return indices
    if type(indices) is int:
        indices = indices if indices <= total else total
        return list(range(indices))
    raise Exception(f'Unsupported type of indices provided, must be an int or list of ints: {type(indices)}')

def load_trained_jobs(ROOT_VOLUME):
    if str(ROOT_VOLUME).startswith('s3://'):
        bucket, files = get_files_from_s3(ROOT_VOLUME)
        return [TrainedJob(folder, keys, bucket) for folder, keys in files]
    ROOT_VOLUME = pathlib.Path(ROOT_VOLUME)
    return [TrainedJob(ROOT_VOLUME / architecture_type, None, None) for architecture_type in os.listdir(ROOT_VOLUME)]



def get_files_from_s3(path: str, ACCESS_KEY='AKIA3S5BOJMD6S5N2P54', SECRET_KEY='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'):
    client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    path = path.split('s3://').pop()
    parts = path.split('/')
    Bucket = parts[0]
    Prefix = '/'.join(parts[1:])

    # Create a reusable Paginator
    paginator = client.get_paginator('list_objects_v2')

    # Create a PageIterator from the Paginator
    page_iterator = paginator.paginate(Bucket=Bucket, Prefix=Prefix)

    contents = []
    for page in page_iterator:
        contents += [c.get('Key') for c in page['Contents']]
        
    grouped_contents = {}
    for file in contents:
        parts = file.split('/')
        folder_name = '/'.join(parts[0:-1])
        grouped_contents[folder_name] = grouped_contents.get(folder_name, []) + [parts[-1]]
    return Bucket, list(grouped_contents.items())
        

def to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def should_evaluate(trained_job, validparams=None):
    if validparams is None:
        return True
    params = json.loads(trained_job.get_params())

    try:
        for key, val in validparams.items():
            if key == 'arch':
                if str(trained_job).startswith(val) is False:
                    return False
            # elif key == 'compress':
            #     compression = trained_job.get_compression()
            #     if compression != val:
            #         return False
            else:
                paramsval = params.get(key)
                if paramsval == '0.0':
                    paramsval = '0'
                if val != int(paramsval):
                    return False
        return True
    except Exception as e:
        print(validparams, trained_job)
        raise e

def evaluate_jobs(evaluator, folder, n=None, show_n=0, show_best_n=0, show_worst_n=0, validparams=None, reset_cache=False, check_every_model=False):
    ts = load_trained_jobs(folder)
    if validparams is None:
        matching_training_jobs = ts
    else:
        matching_training_jobs = []
        for trained_job in ts:
            try:
                if should_evaluate(trained_job, validparams):
                    matching_training_jobs.append(trained_job)
            except:
                pass
    if len(matching_training_jobs) == 0:
        if validparams is None:
            print(f'No training jobs found for folder {folder}')           
        else:
            print(f'No training jobs found for folder {folder} that match params provided: {validparams}')

    print('Number of matching training jobs:', len(matching_training_jobs))        
    results = []
    for trained_job in matching_training_jobs:
        # print(trained_job)
        params = trained_job.get_params()
        scale = int(json.loads(params).get('x'))
  

        models = trained_job.get_models(check_every_model=check_every_model)
        pbar = None if check_every_model is False else tqdm(total=len(models))
        for model, modelname in models:
            try:
                r = cache.get(trained_job)
                if r is None or reset_cache is True:
                    r = evaluator.evaluate(model, output_dir='./outputs', scale=scale, n=n, show_tqdm=check_every_model is False)
                    cache.set(trained_job, r)
                    # print(r)
                    if show_n > 0:
                        print('Comparisons')
                        r.show_n(show_n, row_size=5)
                    if show_best_n > 0:                
                        print('Best')
                        r.show_best_n(n=show_best_n, row_size=5)
                    if show_worst_n > 0:                
                        print('Worst')
                        r.show_worst_n(n=show_worst_n, row_size=5)
                else:
                    # print(r)
                    if show_n > 0 or show_best_n > 0 or show_worst_n > 0:
                        print('To show images, run with "reset_cache" set to True')
                # r = evaluator.evaluate(RDN(weights='psnr-small'), output_dir='./outputs', scale=2)    
                results.append((trained_job, r, str(modelname).split('/').pop()))
                
            except Exception as e:
                print(trained_job)
                print(e)
            if pbar is not None:
                pbar.update(1)
        display(HTML('<hr />'))
        if pbar is not None:
            pbar.close();
    return results
