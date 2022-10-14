import os
import argparse
import pathlib
from ISR.models import RDN, RRDN
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

tf.get_logger().setLevel('ERROR')

print('================================================================')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print('================================================================')

def get_model(model, arch_params):
    if model == 'rdn':
        return RDN(arch_params=arch_params)
    elif model == 'rrdn':
        return RRDN(arch_params=arch_params)
    raise Exception('No valid model found for ' + model)
    
def get_params(folder):
    arch, C, D, G, G0, T, x, _, _2, _3, _4, _5 = folder.split('-')

    arch_params = {
        'C': int(C[1:]),
        'D': int(D[1:]),
        'G': int(G[1:]),
        'G0':int(G0[2:]),
        'x':int(x[1:])
    }
    if arch == 'rrdn':
        arch_params['T'] = int(T[1:])
    return arch, arch_params

# def save_model(weights, output, arch, x, C, D, G, G0, T):
#     model.model.load_weights('/code/weights/' + weights)
#     model.model.save('/code/weights/' + output)

def get_weights(folder):
    weights = []
    for date_folder in os.listdir(folder):
        date_folder = folder / date_folder
        weights += [str(date_folder / f) for f in os.listdir(date_folder) if 'srgan' not in f and f.endswith('hdf5')]
    return weights

def convert_weight_files_to_model_files():
    output = pathlib.Path('/code/output')
    root = output / 'weights'
    target = output / 'models'

    if len(os.listdir(target)) != 0:
        raise Exception('target directory is not empty')
    if len(os.listdir(root)) == 0:
        raise Exception('weights directory is empty. have you bound a local volume of weights to the docker container?')

    weights = []
    errs = []

    for folder in os.listdir(root):
        arch, arch_params = get_params(folder)
        weights += [(w, arch, arch_params) for w in get_weights(root / folder)]
        
    weights = weights[0:]
        
    i = 0
    for weight, arch, arch_params in tqdm(weights):
        try:
            tf.keras.backend.clear_session() # needed for https://github.com/tensorflow/tfjs/issues/755#issuecomment-489665951
            model = get_model(arch, arch_params)
            model.model.load_weights(weight)
            weight_name = weight.split('/')[-3:]
            weight_name = '/'.join(weight_name).split('.')[0] + '.h5'
            target_path = target / weight_name
            os.makedirs('/'.join(str(target_path).split('/')[0:-1]), exist_ok=True)
            model.model.save(target_path)       
            i += 1
        except Exception as e:
            errs += [(weight, e)]
            
    print(f'Successfully processed {i} files')
    if len(errs) > 0:
        print(f'The following {len(errs)} weights could not be processed\n-------------------------------')
        for err, e in errs:
            print(err, e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ESRGAN weights to models')
    parser.add_argument(f'--root', type=str)
    args = parser.parse_args()

    convert_weight_files_to_model_files()

