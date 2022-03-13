import pprint
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

pp = pprint.PrettyPrinter(indent=4)

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

with open(OUTPUT / 'output-sample-file.txt', 'w') as f:
    f.write('foo')
with open(MODEL / 'model-sample-file.txt', 'w') as f:
    f.write('foo')

log_dirs = {'logs': str(LOGS_DIR), 'weights': str(WEIGHTS_DIR)}
for d in [
    '/opt/ml/input/data/inputData/DIV2K_valid_LR_bicubic',
    '/opt/ml/input/data/inputData/DIV2K_valid_LR_bicubic/X2',
    CHECKPOINTS,
    WEIGHTS_DIR,
]:
    print(str(d), os.listdir(str(d)))

with open(HYPERPARAMETERS_PATH, 'r') as f:
    HYPERPARAMETERS = json.load(f)

print('hyperparams', HYPERPARAMETERS)


# def train_rrdn(
#     args,
# ):
#     lr_train_dir=args.lr_train_dir
#     lr_valid_dir=args.lr_valid_dir
#     hr_train_dir=args.hr_train_dir
#     hr_valid_dir=args.hr_valid_dir
#     C=args.C
#     D=args.D
#     G=args.G
#     T=args.T
#     G0=args.G0
#     epochs=args.epochs
#     scale=args.scale
#     batches_per_epoch=args.batches_per_epoch
#     patch_size=args.patch_size
#     batch_size=args.batch_size
#     print('train rrdn')
#     lr_train_patch_size = patch_size
#     layers_to_extract = [5, 9]
#     hr_train_patch_size = lr_train_patch_size * scale
#     arch_params = {
#         'C': C,
#         'D': D,
#         'G': G,
#         'G0':G0,
#         'T':T,
#         'x':scale
#     }
#     generator  = RRDN(arch_params=arch_params, patch_size=lr_train_patch_size)
#     f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
#     discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

#     loss_weights = {
#         'generator': 0.0,
#         'feature_extractor': 0.0833,
#         'discriminator': 0.01
#     }
#     losses = {
#         'generator': 'mae',
#         'feature_extractor': 'mse',
#         'discriminator': 'binary_crossentropy'
#     }

#     learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

#     flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

#     adam_optimizer = {'beta1': 0.9, 'beta2': 0.999, 'epsilon': None}

#     trainer = Trainer(
#         generator=generator,
#         discriminator=discr,
#         feature_extractor=f_ext,
#         lr_train_dir=lr_train_dir,
#         hr_train_dir=hr_train_dir,
#         lr_valid_dir=lr_valid_dir,
#         hr_valid_dir=hr_valid_dir,
#         loss_weights=loss_weights,
#         losses=losses,
#         learning_rate=learning_rate,
#         flatness=flatness,
#         dataname='div2k',
#         log_dirs=log_dirs,
#         weights_generator=None,
#         weights_discriminator=None,
#         n_validation=40,
#         adam_optimizer=adam_optimizer,
#     )

#     print(f'Starting training now for {epochs} epochs')
#     trainer.train(
#         epochs=epochs,
#         steps_per_epoch=batches_per_epoch,
#         batch_size=batch_size,
#         monitored_metrics={
#             'val_generator_PSNR_Y': 'max',
#             'val_PSNR_Y': 'max',
#             'val_generator_loss': 'min',
#         }
#     )

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

def train(
    generator,
    args,
):
    hr_train_dir=pathlib.Path(args.hr_train_dir)
    hr_valid_dir=pathlib.Path(args.hr_valid_dir)
    epochs=args.epochs
    scale=args.scale
    batches_per_epoch=args.batches_per_epoch
    patch_size=args.patch_size
    batch_size=args.batch_size
    n_validation = args.n_validation
    compression_quality = args.compression_quality

    lr_train_patch_size = patch_size
    layers_to_extract = [5, 9]
    hr_train_patch_size = lr_train_patch_size * scale
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
    print('****************************************')
    print('Preparing data')
    train_dirs, valid_dirs = prepare_data([
        (hr_train_dir, INTERNAL_DATA / 'train'),
        (hr_valid_dir, INTERNAL_DATA / 'valid'),
    ], scale=scale, quality=compression_quality)
    processed_hr_train_dir, processed_lr_train_dir = train_dirs
    processed_hr_valid_dir, processed_lr_valid_dir = valid_dirs
    print('****************************************')

    trainer = Trainer(
        generator=generator,
        discriminator=discr,
        feature_extractor=f_ext,
        lr_train_dir=processed_lr_train_dir,
        hr_train_dir=processed_hr_train_dir,
        lr_valid_dir=processed_lr_valid_dir,
        hr_valid_dir=processed_hr_valid_dir,
        loss_weights=loss_weights,
        losses=losses,
        learning_rate=learning_rate,
        flatness=flatness,
        dataname='image_dataset',
        log_dirs=log_dirs,
        weights_generator=weights_generator,
        weights_discriminator=weights_discriminator,
        n_validation=n_validation,
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

def get_model(args):
    if args.model == 'rdn':
        C=args.C
        D=args.D
        G=args.G
        T=args.T
        G0=args.G0
        scale=args.scale
        patch_size=args.patch_size

        lr_train_patch_size = patch_size
        arch_params = {
            'C': C, 
            'D': D, 
            'G': G, 
            'G0':G0, 
            'T':T, 
            'x':scale
        }
        return RDN(arch_params=arch_params, patch_size=lr_train_patch_size)
    elif args.model == 'rrdn':
        C=args.C
        D=args.D
        G=args.G
        T=args.T
        G0=args.G0
        scale=args.scale
        patch_size=args.patch_size

        lr_train_patch_size = patch_size
        arch_params = {
            'C': C, 
            'D': D, 
            'G': G, 
            'G0':G0, 
            'T':T, 
            'x':scale
        }
        return RRDN(arch_params=arch_params, patch_size=lr_train_patch_size)
    raise Exception(f'No valid model found for {args.model}')

def main(args):
    with open(OUTPUT / 'args.json', 'w') as f:
        args_config = vars(args)
        f.write(json.dumps(args_config))
    start = datetime.datetime.now()
    generator = get_model(args)
    train(generator, args)
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
        # ('lr_train_dir', str(INPUT / 'DIV2K_train_LR_bicubic/X2'), str),
        # ('lr_valid_dir', str(INPUT / 'DIV2K_valid_LR_bicubic/X2'), str),
        ('hr_train_dir', str(INPUT / 'DIV2K_train_HR'), str),
        ('hr_valid_dir', str(INPUT / 'DIV2K_valid_HR'), str),
        ('model', 'rrdn', str),
        ('C', '6', int),
        ('D', '20', int),
        ('G', '64', int),
        ('T', '10', float),
        ('G0', '64', int),
        ('epochs', '1', int),
        ('scale', '2', int),
        ('batches_per_epoch', '20', int),
        ('patch_size', '16', int),
        ('batch_size', '2', int),
        ('epsilon', '0.1', float),
        ('beta1', '0.9', float),
        ('beta2', '0.999', float),
        ('lr', '0.0004', float),
        ('loss_weight_generator', '1', int),
        ('loss_weight_discriminator', '0.0', float),
        ('loss_weight_feature_extractor', '0.0', float),
        ('lr_decay_factor', '0.5', float),
        ('lr_decay_frequency', '100', int),
        ('n_validation', '40', int),
        ('compression_quality', '50', int),
    ]:
        parser.add_argument(f'--{key}', type=type, default=HYPERPARAMETERS.get(key, default))

    args = parser.parse_args()
    print('training script', args)
    main(args)
    # run_inference(args)
