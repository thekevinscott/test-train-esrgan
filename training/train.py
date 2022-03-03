import argparse
from isr.ISR.train import Trainer
from isr.ISR.models import RRDN, RDN
from isr.ISR.models import Discriminator
from isr.ISR.models import Cut_VGG19

def train_rrdn(
    lr_train_dir,
    lr_valid_dir,
    hr_train_dir,
    hr_valid_dir,
    C=6,
    D=20,
    G=64,
    G0=64,
    T=10,
    epochs=300,
    scale=4,
    batches_per_epoch=1000,
    patch_size=32,
    batch_size=8,
):
    print('train rrdn')
    lr_train_patch_size = patch_size
    layers_to_extract = [5, 9]
    hr_train_patch_size = lr_train_patch_size * scale
    arch_params = {
        'C': C, 
        'D': D, 
        'G': G, 
        'G0':G0, 
        'T':T, 
        'x':scale
    }
    rrdn  = RRDN(arch_params=arch_params, patch_size=lr_train_patch_size)
    f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
    discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

    loss_weights = {
        'generator': 0.0,
        'feature_extractor': 0.0833,
        'discriminator': 0.01
    }
    losses = {
        'generator': 'mae',
        'feature_extractor': 'mse',
        'discriminator': 'binary_crossentropy'
    }

    log_dirs = {'logs': './logs', 'weights': './weights'}

    learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

    flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

    trainer = Trainer(
        generator=rrdn,
        discriminator=discr,
        feature_extractor=f_ext,
        lr_train_dir=lr_train_dir,
        hr_train_dir=hr_train_dir,
        lr_valid_dir=lr_valid_dir,
        hr_valid_dir=hr_valid_dir,
        loss_weights=loss_weights,
        learning_rate=learning_rate,
        flatness=flatness,
        dataname='image_dataset',
        log_dirs=log_dirs,
        weights_generator=None,
        weights_discriminator=None,
        n_validation=40,
    )

    print(f'Starting training now for {epochs} epochs')
    trainer.train(
        epochs=epochs,
        steps_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        monitored_metrics={
            'val_generator_PSNR_Y': 'max',
            'val_PSNR_Y': 'max',
        }
    )

def train_rdn(
    lr_train_dir,
    lr_valid_dir,
    hr_train_dir,
    hr_valid_dir,
    C=6,
    D=20,
    G=64,
    G0=64,
    T=10,
    epochs=86,
    scale=4,
    batches_per_epoch=1000,
    patch_size=32,
    batch_size=8,
):
    print('train rdn')
    lr_train_patch_size = patch_size
    layers_to_extract = [5, 9]
    hr_train_patch_size = lr_train_patch_size * scale
    arch_params = {
        'C': C, 
        'D': D, 
        'G': G, 
        'G0':G0, 
        'T':T, 
        'x':scale
    }
    rdn  = RDN(arch_params=arch_params, patch_size=lr_train_patch_size)
    f_ext = Cut_VGG19(patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract)
    discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)

    loss_weights = {
        'generator': 0.0,
        'feature_extractor': 0.0833,
        'discriminator': 0.01
    }
    losses = {
        'generator': 'mae',
        'feature_extractor': 'mse',
        'discriminator': 'binary_crossentropy'
    }

    log_dirs = {'logs': './logs', 'weights': './weights'}

    learning_rate = {'initial_value': 0.0004, 'decay_factor': 0.5, 'decay_frequency': 30}

    flatness = {'min': 0.0, 'max': 0.15, 'increase': 0.01, 'increase_frequency': 5}

    trainer = Trainer(
        generator=rdn,
        discriminator=discr,
        feature_extractor=f_ext,
        lr_train_dir=lr_train_dir,
        hr_train_dir=hr_train_dir,
        lr_valid_dir=lr_valid_dir,
        hr_valid_dir=hr_valid_dir,
        loss_weights=loss_weights,
        learning_rate=learning_rate,
        flatness=flatness,
        dataname='image_dataset',
        log_dirs=log_dirs,
        weights_generator=None,
        weights_discriminator=None,
        n_validation=40,
    )

    print(f'Starting training now for {epochs} epochs')
    trainer.train(
        epochs=epochs,
        steps_per_epoch=batches_per_epoch,
        batch_size=batch_size,
        monitored_metrics={'val_PSNR_Y': 'max'}
    )

if __name__ == '__main__':
    print('training script')
    parser = argparse.ArgumentParser(description='Train ESRGAN')
    parser.add_argument('--model')
    parser.add_argument('--lr_train_dir')
    parser.add_argument('--lr_valid_dir')
    parser.add_argument('--hr_train_dir')
    parser.add_argument('--hr_valid_dir')
    # 'C':2, 'D':3, 'G':32, 'G0':32, 'T':5,
    # 'C':4, 'D':3, 'G':32, 'G0':32, 'T': 1,
    # 'C':4, 'D':3, 'G':32, 'G0':32, 'T': 10,
    # 'C':4, 'D':3, 'G':32, 'G0':32, 'T':10,
    # 'C':4, 'D':3, 'G':32, 'G0':32, 'T':10,
    # rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5
    # rrdn-C4-D3-G32-G032-T10-x4_epoch299.hdf5
    # rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5
    # rdn-C6-D20-G64-G064-x2_PSNR_epoch086.hdf5
    parser.add_argument('--C', type=int, default=6)
    parser.add_argument('--D', type=int, default=20)
    parser.add_argument('--G', type=int, default=64)
    parser.add_argument('--T', type=int, default=10)
    parser.add_argument('--G0', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--batches_per_epoch', type=int, default=1000)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    print(args)
    if args.model == 'rdn':
        train_rdn(
            args.lr_train_dir,
            args.lr_valid_dir,
            args.hr_train_dir,
            args.hr_valid_dir,
            args.C,
            args.D,
            args.G,
            args.T,
            args.G0,
            args.epochs,
            args.scale,
            args.batches_per_epoch,
            args.patch_size,
            args.batch_size,
        )
    elif args.model == 'rrdn':
        train_rrdn(
            args.lr_train_dir,
            args.lr_valid_dir,
            args.hr_train_dir,
            args.hr_valid_dir,
            args.C,
            args.D,
            args.G,
            args.T,
            args.G0,
            args.epochs,
            args.scale,
            args.batches_per_epoch,
            args.patch_size,
            args.batch_size,
        )
    else:
        raise Exception(f'Unsupported model {args.model}')
