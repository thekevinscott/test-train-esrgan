from typing import List, Tuple, Any
from IPython.display import display
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

def to_cv2(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class EvaluationResult:
    output_dir = None

    def __init__(self, output_dir=None):
        self.durations = []
        self.total_pixels = 0
        self.psnr = []
        self.ssim = []
        self.srs = []
        self.lrs = []
        self.originals = []
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
        # print('calculate metrics')
        original_arr = np.array(original)
        # print(original_arr.shape)
        upscaled_arr = np.array(upscaled)
        # print(upscaled_arr.shape)
        ssim = structural_similarity(original_arr, upscaled_arr, channel_axis=2)
        original = to_cv2(original)
        upscaled = to_cv2(upscaled)

        psnr = cv2.PSNR(original, upscaled)
        return psnr, ssim

    def __str__(self) -> str:
        total_duration = 0
        for d in self.durations:
            total_duration += d
        
        return '\n'.join([
            f'Total duration in ms: {total_duration:9.4f}',
            f'Time per 100 pixels in ms: {(float(total_duration) / self.total_pixels * 100):9.4f}',
            f'PSNR mean: {statistics.mean(self.psnr):9.4f}',
            f'PSNR median: {statistics.median(self.psnr):9.4f}',
            f'SSIM mean: {statistics.mean(self.ssim):9.4f}',
            f'SSIM median: {statistics.median(self.ssim):9.4f}',
        ])

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
        indices = [i for m, i in self.get_sorted_metrics(metric=metric, highest_first=True)]
        self.show_n(indices, n=n, width=width, row_size=row_size)
    def show_worst_n(self, metric='psnr', n=5, width=20, row_size=3):
        indices = [i for m, i in self.get_sorted_metrics(metric=metric, highest_first=False)]
        self.show_n(indices, n=n, width=width, row_size=row_size)


    def show_n(self, indices, n=5, width=20, row_size=3):
        n = n if n <= len(self.lrs) else len(self.lrs)
        fig, axs = plt.subplots(n, 3, figsize=(width, row_size * n))
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

    def evaluate(self, model, scale, dataset=None, output_dir=None, n=None):
        if dataset is None:
            dataset = self.dataset
        evaluation_result = EvaluationResult(output_dir)
        images = list(dataset.get_images())
        if n is not None:
            images = images[:n]
        for sample_image in tqdm(images):
            evaluation = self._evaluate_image(model, sample_image, scale)
            evaluation_result.add_evaluation(*evaluation)
        
        return evaluation_result
        
    def _evaluate_image(self, model, image, scale):
        original, downscaled = image.get(scale)
        lr_img = np.array(downscaled)
        start = datetime.datetime.now()
        sr_img = Image.fromarray(model.predict(lr_img))
        duration = datetime.datetime.now() - start

        lr_img = Image.fromarray(lr_img)
        return duration, image.get_num_pixels(), image, sr_img, lr_img, original

class SampleImage:
    fullpath: pathlib.Path
    size: Tuple[int, int]
    array: ndarray
    def __init__(self, name, fullpath):
        self.fullpath = fullpath
        self.name = name

    def get(self, scale):
        im = Image.open(self.fullpath).convert('RGB')
        cropped_im = crop_image(np.array(im), scale)
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
            yield file

class TrainedJob:
    root_folder: pathlib.Path
    weights_folder: pathlib.Path
    model_folder: pathlib.Path
    run_folders = None
    generators: List[pathlib.Path]
    discriminators: List[pathlib.Path]
    configs: List[pathlib.Path]

    def __init__(self, root) -> None:
        self.generators = []
        self.discriminators = []
        self.configs = []
        self.gather_weights(root)

    def gather_weights(self, root):
        self.root_folder = root
        if 'checkpoints' in str(root):
            return
        self.weights_folder = pathlib.Path('checkpoints/weights')
        if '4e34b0c3' not in str(root):
            return
        # print(root / self.weights_folder)
        subfolders = os.listdir(root / self.weights_folder)
        if len(subfolders) == 0:
            return
        assert len(subfolders) == 1, f'{subfolders} for {root / self.weights_folder}'

        self.model_folder = pathlib.Path(subfolders[0])
        self.run_folders = os.listdir(self.root_folder / self.weights_folder / self.model_folder)

        # files are organized by discriminator / generator / config,
        # further organized by metric,
        # further organized by epoch
        for run_folder in self.run_folders:
            run_folder = pathlib.Path(run_folder)
            for file in os.listdir(self._parse_path(run_folder)):
                print('file', file)
                filepath = run_folder / file
                if file.endswith('yml'):
                    self.configs.append(filepath)
                    # with open(filepath, 'r') as f:
                elif file.endswith('hdf5'):
                    if file.startswith('rdn') or file.startswith('rrdn'):
                        self.generators.append(filepath)
                    elif file.startswith('srgan') or 'generator' in file:
                        self.discriminators.append(filepath)
                else:
                    print(f'Unsupported file: {file}')

    def _choose_best_epoch(self, arr):
        arr = [(int(str(a).split('epoch').pop().split('.hdf5')[0]), a) for a in arr]
        arr.sort()
        return arr.pop()[1]

    def get_generator(self, metric=None):
        if metric:
            matching_generators = [g for g in self.generators if metric in str(g)]
            if len(matching_generators) == 0:
                raise Exception(f'No generators found for metric {metric}')
            return self._choose_best_epoch(matching_generators)
        return self._choose_best_epoch(self.generators)

    def _parse_path(self, pathname):
        return self.root_folder / self.weights_folder / self.model_folder / pathname

    def get_model(self, metric=None):
        weightspath = self.get_generator(metric)
        config = self.get_config(weightspath)
        key = os.path.dirname(weightspath)
        config = config.get(key)
        generatorparams = config.get('generator')
        model = get_model(generatorparams.get('name'), SimpleNamespace(**generatorparams.get('parameters')))
        model.model.load_weights(self._parse_path(weightspath))
        return model

    def get_config(self, generator_path = None):
        if len(self.configs) > 1:
            if generator_path is None:
                raise Exception(f'Multiple config files found, you must provide a corresponding generator path')

            directory_of_generator = os.path.dirname(generator_path)
            print(self.configs)
            matching_configs = [c for c in self.configs if directory_of_generator == os.path.dirname(c)]
            if len(matching_configs) != 1:
                raise Exception('Either too many configs match, or none at all')
            matching_config = matching_configs[0]
            return self._load_config_file(matching_config)
        elif len(self.configs) == 1:
            config = self.configs[0]
            return self._load_config_file(config)
        else:
            if self.run_folders:
                for run_folder in self.run_folders:
                    print(os.listdir(self._parse_path(run_folder)))
            raise Exception(f'No config files found, {self.run_folders}')

    def _load_config_file(self, configpath):
        with open(self._parse_path(configpath), 'r') as f:
            return yaml.load(f, Loader=yaml.BaseLoader)

def load_trained_jobs(ROOT_VOLUME):
    return [TrainedJob(ROOT_VOLUME / tag_name) for tag_name in os.listdir(ROOT_VOLUME)]

def get_model(model, args):
    C=args.C
    D=args.D
    G=args.G
    T=args.T
    G0=args.G0
    x=args.x

    arch_params = {
        'C': int(C), 
        'D': int(D), 
        'G': float(G), 
        'G0':float(G0), 
        'T':float(T), 
        'x':int(x)
    }
    if model == 'rdn':
        return RDN(arch_params=arch_params)
    elif model == 'rrdn':
        return RRDN(arch_params=arch_params)
    raise Exception(f'No valid model found for {args.model}')
