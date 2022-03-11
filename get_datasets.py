from datasets.load_datasets import dataset_loaders
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=str, nargs='*',
                    help='The datasets that should be downloaded.')
args = parser.parse_args()

if args.datasets is None:
    args.datasets = list(dataset_loaders.keys())

print('Datasets for downloading:', args.datasets, sep='\n')

datasets_dir = 'data'
for dataset in args.datasets:
    dataset_loaders[dataset](datasets_dir)
    for subset in ['train', 'test', 'valid']:
        try:
            x = np.load(f'{datasets_dir}/{dataset}_x_{subset}.npy', allow_pickle=True)
            y = np.load(f'{datasets_dir}/{dataset}_y_{subset}.npy', allow_pickle=True)
            data = pd.DataFrame(x, columns=[f'f{i}' for i in range(x.shape[1])])
            if 'int' not in str(y.dtype) and len(np.unique(y)) < 32:
                y = y.astype('int')
            data['target'] = y
            data.to_csv(f'{datasets_dir}/{dataset}_{subset}.csv', index=False)
        except:
            pass
