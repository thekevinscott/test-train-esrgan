import pathlib
import os
import numpy as np
import yaml
from PIL import Image
from types import SimpleNamespace

ROOT = pathlib.Path('/opt/ml')
CODE = pathlib.Path('/code')
INPUT = ROOT / 'input/data/inputData'
HYPERPARAMETERS_PATH = ROOT / 'input/config/hyperparameters.json'
OUTPUT = ROOT / 'output' # dunno wtf this is for
CHECKPOINTS = ROOT / 'checkpoints' # save checkpoints here
MODEL = ROOT / 'model' # save the model here


LOGS_DIR = pathlib.Path('/logs')
WEIGHTS_DIR = CHECKPOINTS / 'weights'

def get_files(directory):
    model_weight = None
    generator_weight = None
    config = None
    for file in os.listdir(directory):
        if file.startswith('rdn') or file.startswith('rrdn'):
            model_weight = file
        elif file.startswith('srgan') or 'generator' in file:
            generator_weight = file
        elif file.endswith('yml'):
            with open(directory / file, 'r') as f:
                config = yaml.load(f, Loader=yaml.Loader)
    if model_weight is None:
        print(f'Could not find model weight in {directory}', os.listdir(directory))
    elif generator_weight is None:
        print(f'Could not find generator weight in {directory}', os.listdir(directory))
    elif config is None:
        print(f'Could not config file in {directory}', os.listdir(directory))

    return model_weight, generator_weight, config
# ['srgan-large_best-val_generator_PSNR_Y_epoch001.hdf5', 'rrdn-C6-D20-G64-G010-T64-x2_best-val_generator_PSNR_Y_epoch001.hdf5', 'session_config.yml']

def get_config_files_for_directory(directory):
    for timestamp in os.listdir(directory):
        config_file_path = directory / timestamp / 'session_config.yml'
        if os.path.exists(config_file_path) is False:
            raise Exception(f'No file found for {config_file_path}')

        with open(config_file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

        config = config.get(timestamp)

        discriminator = config.get('discriminator')
        feature_extractor = config.get('feature_extractor')
        generator = config.get('generator')
        training_parameters = config.get('training_parameters')

        yield training_parameters, generator, discriminator, feature_extractor

        #     # {   'discriminator': {'name': 'srgan-large', 'weights_discriminator': None},
    # # 'feature_extractor': {'layers': [5, 9], 'name': 'vgg19'},
    # # 'generator': {   'name': 'rrdn',
        #     #          'parameters': {   'C': 6,
        #     #                            'D': 20,
        #     #                            'G': 64,
        #     #                            'G0': 10,
        #     #                            'T': 64,
        #     #                            'x': 2},
        #     #          'weights_generator': None},
    # # 'training_parameters'
        #     return
        #     print('config', pp.pprint(config.get(timestamp_dir)))

def load_sample_images():
    directory = CODE / 'data/input/sample'
    for imgname in os.listdir(directory):
        img = Image.open(directory / imgname)
        lr_img = np.array(img)
        yield lr_img, imgname


def run_inference(args):
    for weight in os.listdir(WEIGHTS_DIR):
        weight_path_directory = WEIGHTS_DIR / weight
        for training_parameters, generator, discriminator, feature_extractor in get_config_files_for_directory(weight_path_directory):
            params = generator.get('parameters')
            model = get_model(SimpleNamespace(**params, model=generator.get('name')))
            for lr_img, imgname in load_sample_images():
                img_directory = OUTPUT / 'samples' / weight
                img_directory.mkdir(exist_ok=True, parents=True)
                imgsavepath = img_directory / imgname
                print('saving image to', imgsavepath)
                sr_img = Image.fromarray(model.predict(lr_img))
                print('made from array')
                sr_img.save(str(imgsavepath))
                print('saved it')
            print('done looping through images')
        print('done looping through config files')
    print('done looping through folders')
