import pprint
import numpy as np
import math
import json
import pathlib
import os
import argparse
import datetime
from ISR.train import Trainer
from ISR.models import RRDN, RDN
from ISR.models import Discriminator
from ISR.models import Cut_VGG19
from utils import prepare_data
import tensorflow as tf

print('================================================================')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('================================================================')


# pp = pprint.PrettyPrinter(indent=4)

INTERNAL_DATA = pathlib.Path('/data')
ROOT = pathlib.Path('/opt/ml')
CODE = pathlib.Path('/code')
INPUT = ROOT / 'input/data/inputData'
HYPERPARAMETERS_PATH = ROOT / 'input/config/hyperparameters.json'
OUTPUT = ROOT / 'output' # dunno wtf this is for
CHECKPOINTS = ROOT / 'checkpoints' # save checkpoints here
MODEL = ROOT / 'model' # save the model here


LOGS_DIR = pathlib.Path('/logs')
WEIGHTS_DIR = CHECKPOINTS / 'weights'

OUTPUT.mkdir(exist_ok=True, parents=True)
CHECKPOINTS.mkdir(exist_ok=True, parents=True)
MODEL.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)
INTERNAL_DATA.mkdir(exist_ok=True, parents=True)

log_dirs = {'logs': str(LOGS_DIR), 'weights': str(WEIGHTS_DIR)}
for d in [
    '/opt/ml/input/data/inputData/',
    '/opt/ml/input/data/inputData/train',
    '/opt/ml/input/data/inputData/train/hr',
    CHECKPOINTS,
    WEIGHTS_DIR,
]:
    print(str(d), os.listdir(str(d)))

with open(HYPERPARAMETERS_PATH, 'r') as f:
    HYPERPARAMETERS = json.load(f)

print('********************************************************')
print('hyperparams', HYPERPARAMETERS)
print('********************************************************')

def get_arch_name(args):
    return '-'.join([
        args.model,
        f'C{args.C}',
        f'D{args.D}',
        f'G{args.G}',
        f'G0{args.G0}',
        f'T{args.T}',
        f'x{args.scale}',
    ])

def get_checkpoints_if_exist(args, monitored_metric):
    arch_name = get_arch_name(args)
    if (WEIGHTS_DIR / arch_name).exists():
        weights_generators = []
        weights_discriminators = []
        for directory in os.listdir(WEIGHTS_DIR / arch_name):
            files = os.listdir(WEIGHTS_DIR / arch_name / directory)
            for file in files:
                if file.endswith('hdf5') and monitored_metric in file:
                    file_path = f'{directory}/{file}'
                    if arch_name in file:
                        weights_generators.append(file_path)
                    else:
                        weights_discriminators.append(file_path)

        weights_discriminators.sort()
        weights_generators.sort()
        print('Generator Weights', weights_generators)
        print('Discriminator Weights', weights_discriminators)
        return str(WEIGHTS_DIR / arch_name / weights_generators.pop()), str(WEIGHTS_DIR / arch_name / weights_discriminators.pop())

    return None, None

def make_basename(generator, args, hr_train_patch_size):
    gen_name = generator.name
    params = [gen_name]
    for param in np.sort(list(generator.params.keys())):
        params.append('{g}{p}'.format(g=param, p=generator.params[param]))
    params.append('{g}{p}'.format(g='patchsize', p=hr_train_patch_size))
    params.append('{g}{p}'.format(g='compress', p=args.compression_quality))
    params.append('{g}{p}'.format(g='sharpen', p=args.sharpen_amount))
    return '-'.join(params)

        # ('batches_per_epoch', '20', int),
        # ('hr_patch_size', '128', int),
        # ('batch_size', '16', int),
        # ('epsilon', '0.1', float),
        # ('beta1', '0.9', float),
        # ('beta2', '0.999', float),
        # ('lr', '0.0004', float),
        # ('loss_weight_generator', '1', int),
        # ('loss_weight_discriminator', '0.0', float),
        # ('loss_weight_feature_extractor', '0.0', float),
        # ('lr_decay_factor', '0.5', float),
        # ('lr_decay_frequency', '100', int),
        # ('n_validation', '100', int),
        # ('compression_quality', '50', int),
        # ('sharpen_amount', '1', int),

def train(
    generator,
    args,
    hr_train_patch_size,
):
    hr_train_dir=pathlib.Path(args.hr_train_dir)
    hr_valid_dir=pathlib.Path(args.hr_valid_dir)
    lr_train_dir=pathlib.Path(args.lr_train_dir)
    lr_valid_dir=pathlib.Path(args.lr_valid_dir)
    epochs=args.epochs
    # scale=args.scale
    batches_per_epoch=args.batches_per_epoch
    batch_size=args.batch_size
    n_validation = args.n_validation
    # compression_quality = args.compression_quality

    layers_to_extract = [5, 9]

    f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
    discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

    loss_weights = {
        'generator': args.loss_weight_generator,
        'feature_extractor': args.loss_weight_feature_extractor,
        'discriminator': args.loss_weight_discriminator,
    }
    losses = {
        'generator': 'mae',
        'feature_extractor': 'mse',
        'discriminator': 'binary_crossentropy'
    }

    learning_rate = {'initial_value': args.lr, 'decay_factor': args.lr_decay_factor, 'decay_frequency': args.lr_decay_frequency}

    flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

    print('****************************************')
    weights_generator, weights_discriminator = get_checkpoints_if_exist(args, 'val_generator_PSNR_Y')
    if weights_generator:
        print(f'Checkpoint was found for weights generator: {weights_generator}')
    else:
        print('Checkpoint was not found for weights generator, starting from scratch')
    if weights_discriminator:
        print(f'Checkpoint was found for weights discriminator: {weights_discriminator}')
    else:
        print('Checkpoint was not found for weights discriminator, starting from scratch')

    trainer = Trainer(
        generator=generator,
        discriminator=discr,
        feature_extractor=f_ext,
        lr_train_dir=lr_train_dir,
        hr_train_dir=hr_train_dir,
        lr_valid_dir=lr_valid_dir,
        hr_valid_dir=hr_valid_dir,
        loss_weights=loss_weights,
        losses=losses,
        learning_rate=learning_rate,
        flatness=flatness,
        dataname='image_dataset',
        log_dirs=log_dirs,
        weights_generator=weights_generator,
        weights_discriminator=weights_discriminator,
        n_validation=n_validation,
        basename=make_basename(generator, args, hr_train_patch_size),
    )

    print(f'Starting training now for {epochs} epochs')
    trainer.train(
        epochs=epochs,
        steps_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        monitored_metrics={
            'val_loss': 'min',
            'val_PSNR_Y': 'max',
            'val_generator_loss': 'min',
            'val_generator_PSNR_Y': 'max',
        }
    )
    print('Training complete')

def get_model(args, lr_train_patch_size):
    C=args.C
    D=args.D
    G=args.G
    T=args.T
    G0=args.G0
    scale=args.scale

    arch_params = {
        'C': C, 
        'D': D, 
        'G': G, 
        'G0':G0, 
        'T':T, 
        'x':scale
    }
    if args.model == 'rdn':
        return RDN(arch_params=arch_params, patch_size=lr_train_patch_size)
    elif args.model == 'rrdn':
        return RRDN(arch_params=arch_params, patch_size=lr_train_patch_size)
    raise Exception(f'No valid model found for {args.model}')

def main(args):
    with open(OUTPUT / 'args.json', 'w') as f:
        args_config = vars(args)
        f.write(json.dumps(args_config))
    start = datetime.datetime.now()
    scale = args.scale
    hr_train_patch_size = args.hr_patch_size
    if hr_train_patch_size / scale != math.floor(hr_train_patch_size / scale):
        raise Exception(f'Patch size must be divisible by scale which is {scale}')
    lr_train_patch_size = int(hr_train_patch_size / scale)
    generator = get_model(args, lr_train_patch_size)
    train(generator, args, hr_train_patch_size)
    elapsed_time = datetime.datetime.now() - start
    print('Elapsed seconds', elapsed_time.total_seconds())
    with open(OUTPUT / 'output.json', 'w') as f:
        output_config = {
            'total_seconds': elapsed_time.total_seconds()
        }
        f.write(json.dumps(output_config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ESRGAN')
    # 'C':2, 'D':3, 'G':32, 'G0':32, 'T':5,
    # 'C':4, 'D':3, 'G':32, 'G0':32, 'T': 1,
    # 'C':4, 'D':3, 'G':32, 'G0':32, 'T': 10,
    # 'C':4, 'D':3, 'G':32, 'G0':32, 'T':10,
    # 'C':4, 'D':3, 'G':32, 'G0':32, 'T':10,
    # rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5
    # rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5
    # rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5
    # rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5
    for key, default, type in [
        ('lr_train_dir', str(INPUT / 'train/lr'), str),
        ('lr_valid_dir', str(INPUT / 'valid/lr'), str),
        ('hr_train_dir', str(INPUT / 'train/hr'), str),
        ('hr_valid_dir', str(INPUT / 'valid/hr'), str),
        ('model', 'rrdn', str),
        ('C', '6', int),
        ('D', '20', int),
        ('G', '64', int),
        ('T', '10', float),
        ('G0', '64', int),
        ('epochs', '1', int),
        ('scale', '2', int),
        ('batches_per_epoch', '500', int),
        ('hr_patch_size', '128', int),
        ('batch_size', '16', int),
        ('epsilon', '0.1', float),
        ('beta1', '0.9', float),
        ('beta2', '0.999', float),
        ('lr', '0.0004', float),
        ('loss_weight_generator', '1', int),
        ('loss_weight_discriminator', '0.0', float),
        ('loss_weight_feature_extractor', '0.0', float),
        ('lr_decay_factor', '0.5', float),
        ('lr_decay_frequency', '100', int),
        ('n_validation', '100', int),
        ('compression_quality', '50', int),
        ('sharpen_amount', '1', int),
    ]:
        parser.add_argument(f'--{key}', type=type, default=HYPERPARAMETERS.get(key, default))

    args = parser.parse_args()
    print('training script', args)
    main(args)
    # run_inference(args)
