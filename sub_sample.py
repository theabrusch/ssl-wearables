import numpy as np
from glob import glob
import os
import shutil

remake = False
move_subset = True

if remake:
    files = glob('/Users/theb/Documents/PhD/code/alignement_of_representations/data/features/*/*/*.npy')
    for i, f in enumerate(files):
        print('Processing file', i+1, 'of', len(files))
        try:
            latents = np.load(f, allow_pickle=True)
            latents = latents.reshape(-1)
            np.save(f, latents)
        except:
            print('Error with file', f)
            os.remove(f)

nsamps = 1000
if move_subset:
    models = ['humanactivity_postfinetuning', 'humanactivity_prefinetuning']
    for m in models:
        src_directory = f'/Users/theb/Documents/PhD/code/alignement_of_representations/data/features/{m}/'
        dst_directory = f'/Users/theb/Documents/PhD/code/alignement_of_representations/data/features/{m}_subset/'
        # create dst_directory
        if not os.path.exists(dst_directory):
            os.mkdir(dst_directory)
        # load detailed y
        det_y = np.load('/Users/theb/Documents/PhD/code/alignement_of_representations/data/features/humanactivity_postfinetuning/detailed_y_1.npy')
        
        files = ['person_id.npy', 'y_labels.npy', 'detailed_y_0.npy', 'detailed_y_1.npy', 'detailed_y_2.npy']
        loaded_files = dict()
        for f in files:
            loaded_files[f] = np.load(src_directory + f)

        full_idx = []
        classes = np.unique(det_y)
        for c in classes:
            det_y_c_idx = np.where(det_y == c)[0]
            # select 1000 random samples
            if len(det_y_c_idx) > nsamps:
                idx = np.random.choice(det_y_c_idx, nsamps, replace=False)
            else:
                idx = det_y_c_idx

            full_idx.append(idx)
        full_idx = np.sort(np.concatenate(full_idx))
        det_y_subset = det_y[full_idx]

        # subsample loaded files and save in dtst_directory
        for f in files:
            loaded_files[f] = loaded_files[f][full_idx]
            np.save(dst_directory + f, loaded_files[f])
        
        # copy the files corresponding to the index from all layers to dst_directory
        layers = ['layer_0', 'layer_1', 'layer_2', 'layer_3', 'layer_4', 'layer_5']
        for l in layers:
            if not os.path.exists(dst_directory + l):
                os.mkdir(dst_directory + l)
            else:
                # delete folder and content and create it again
                shutil.rmtree(dst_directory + l)
                os.mkdir(dst_directory + l)

            for i, idx in enumerate(full_idx):
                file_name = str(idx) + '.npy'
                new_file_name = str(i) + '.npy'
                shutil.copy(src_directory + l + '/' + file_name, dst_directory + l + '/' + new_file_name)
            