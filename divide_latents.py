import numpy as np
from glob import glob
import os
import argparse
import shutil

def main(args):

    files = glob(f'{args.path}/fold0/*.npy')
    for i, f in enumerate(files):
        model = f.split('_')[0]
        print('Processing file', i+1, 'of', len(files))
        if 'latent' in f or 'input' in f:
            latent = np.load(f)
            if 'latent' in f:
                layer = f.split('.')[0][-1]
                path = f'{args.output_path}/humanactivity_{model}finetuning/{layer}'
            else:
                path = f'{args.output_path}/humanactivity_{model}finetuning/input'
            for idx, l in enumerate(latent):
                l = l.reshape(1, -1)
                np.save(f'{path}/{idx}.npy', l)
        elif f.split('/')[-1] == 'det_y.npy':
            labels = np.load(f)
            for i, lab in enumerate(labels):
                print('Processing file', i+1, 'of', len(labels))
                label = np.load(lab)
                path = f'{args.output_path}/humanactivity_{model}finetuning'
                for idx, l in enumerate(label.T):
                    np.save(f'{path}/detailed_y_{idx}.npy', l)
        else:
            # simply copy file to output folder
            new_name = '_'.join(f.split('/')[-1].split('_')[1:])
            shutil.copy(f, f'{args.output_path}/humanactivity_{model}finetuning/{new_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Divide latents')
    parser.add_argument('--path', type=str, default='/Users/theb/Documents/PhD/code/alignement_of_representations/data/features/', help='Path to the folder containing the latents')
    parser.add_argument('--output_path', type=str, default='/Users/theb/Documents/PhD/code/alignement_of_representations/data/features/', help='Path to the folder containing the latents')
    args = parser.parse_args()

    main(args)