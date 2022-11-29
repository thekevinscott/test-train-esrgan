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
    parts = folder.split('-')
    arch = parts[0]
    C = parts[1]
    D = parts[2]
    G = parts[3]
    G0 = parts[4]
    T = parts[5]
    x = parts[6]
    _patchsize = parts[7]
    _compress = parts[8]
    _sharpen = parts[9]
    data = parts[10].split('data').pop()
    _vary_compression = parts[11]

    arch_params = {
        'C': int(C[1:]),
        'D': int(D[1:]),
        'G': int(G[1:]),
        'G0':int(G0[2:]),
        'x':int(x[1:])
    }
    if arch == 'rrdn':
        arch_params['T'] = int(T[1:])
    return arch, arch_params, data

# def save_model(weights, output, arch, x, C, D, G, G0, T):
#     model.model.load_weights('/code/weights/' + weights)
#     model.model.save('/code/weights/' + output)

def get_weights(folder):
    weights = []
    for date_folder in os.listdir(folder):
        date_folder = folder / date_folder
        for f in os.listdir(date_folder):
            if 'srgan' not in f and f.endswith('hdf5'):
                weights += [str(date_folder / f)]
    return weights

def convert_weight_files_to_model_files():
    output = pathlib.Path('/code/output')
    root = output / 'weights'
    target = output / 'models'

    if len(os.listdir(root)) == 0:
        raise Exception('weights directory is empty. have you bound a local volume of weights to the docker container?')

    weights = []
    errs = []

    for folder in os.listdir(root):
        arch, arch_params, data = get_params(folder)
        weights += [(w, arch, arch_params, data) for w in get_weights(root / folder)]
        
    weights = weights[0:]
        
    i = 0
    processed = [] 
    skipped = [] 
    for weight, arch, arch_params, data in tqdm(weights):
        try:
            i += 1
            weight_name = weight.split('/')[-3:]
            weight_name = '/'.join(weight_name).split('.')[0] + '.h5'
            target_path = target / weight_name
            if os.path.exists(target_path):
                skipped.append(weight_name)
            else:
                tf.keras.backend.clear_session() # needed for https://github.com/tensorflow/tfjs/issues/755#issuecomment-489665951
                model = get_model(arch, arch_params)
                model.model.load_weights(weight)
                os.makedirs('/'.join(str(target_path).split('/')[0:-1]), exist_ok=True)
                model.model.save(target_path)       
                processed.append(weight_name)
        except Exception as e:
            errs += [(weight, e)]
            
    print(f'Successfully processed {len(processed)} files, skipped {len(skipped)} files.')
    if len(processed):
        print('***********')
        print('The files processed were:')
        for weight in processed:
            print(f'* {weight}')
        print('***********')
    if len(skipped):
        print('***********')
        print('The files skipped were:')
        for weight in skipped:
            print(f'* {weight}')
        print('***********')
    if len(errs) > 0:
        print(f'The following {len(errs)} weights could not be processed\n-------------------------------')
        for err, e in errs:
            print(err, e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert ESRGAN weights to models')
    parser.add_argument(f'--root', type=str)
    args = parser.parse_args()

    convert_weight_files_to_model_files()

