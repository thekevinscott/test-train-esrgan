import pathlib
import argparse
from utils import prepare_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--scale', '-s', type=int)
    parser.add_argument('directories', nargs='*')

    args = parser.parse_args()
    print('preprocessing data', args)

    directory_arguments = []
    for d in args.directories:
        parts = d.split(':')
        src = parts[0]
        dest = ':'.join(parts[1:])
        if src is None or dest is None:
            raise Exception(f'Invalid directory argument provided, must be of format <src>:<dest>. You provided: {d}')

        directory_arguments.append((
            src,
            dest,
        ))

    result = prepare_data(directory_arguments, scale=args.scale)
    print(result)
