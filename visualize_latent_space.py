from sklearn.manifold import TSNE
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tracemalloc


pre_post = ['pre', 'post']
layer = np.arange(5,6)

for pp in pre_post:
    for l in layer:
        latent = np.load(f'/Users/theb/Desktop/fold0/{pp}_latent{l}.npy')
        true_y = np.load(f'/Users/theb/Desktop/fold0/{pp}_y_test.npy')

        unique_y = np.unique(true_y)
        down_sample = True
        if down_sample:
            n = 1000
            down_sampled_version = np.zeros((4*n, *latent.shape[1:]))
            down_sampled_y = np.zeros(4*n)

            for y in unique_y:
                idx = true_y == y
                idx = np.random.choice(np.where(idx)[0], n, replace = False)
                down_sampled_version[y*n:(y+1)*n] = latent[idx]
                down_sampled_y[y*n:(y+1)*n] = true_y[idx]
            down_sampled_version = np.reshape(down_sampled_version, (down_sampled_version.shape[0], -1))
        else:
            down_sampled_version = np.reshape(latent, (latent.shape[0], -1))
            down_sampled_y = true_y


        shuffleidx = np.arange(4*n)
        np.random.shuffle(shuffleidx)
        down_sampled_version = down_sampled_version[shuffleidx]
        down_sampled_y = down_sampled_y[shuffleidx]

        perplexity = [5, 10, 20, 30, 40, 50]

        for p in perplexity:
            transformer = TSNE(perplexity=p)
            transform = transformer.fit_transform(down_sampled_version)

            labels = ['light', 'moderate-vigorous', 'sedentary', 'sleep']

            fig, ax = plt.subplots()
            scatter = ax.scatter(transform[:,0], transform[:,1], c = down_sampled_y, cmap = 'tab10')
            ax.legend(handles=scatter.legend_elements()[0], labels=labels)
            plt.title(f'Latent Space Visualization with t-SNE, perplexity = {p}')
            plt.savefig(f'latent_plots/{pp}latent_space_{l}_{p}.png')

pre_post = 'post'
layer = 5

post_latent_4 = np.load(f'/Users/theb/Desktop/fold0/{pre_post}_latent{layer}.npy')
det_y = np.load(f'/Users/theb/Desktop/fold0/{pre_post}_det_y.npy')
det_y_labels = np.unique(det_y[:,2])
det_y = LabelEncoder().fit_transform(det_y[:,2])
true_y = np.load(f'/Users/theb/Desktop/fold0/{pre_post}_y_test.npy')

unique_y = np.unique(true_y)
down_sample = True
if down_sample:
    n = 1000
    down_sampled_version = np.zeros((4*n, *post_latent_4.shape[1:]))
    down_sampled_y = np.zeros(4*n)
    down_sampled_det_y = np.zeros(4*n)

    for y in unique_y:
        idx = true_y == y
        idx = np.random.choice(np.where(idx)[0], n, replace = False)
        down_sampled_version[y*n:(y+1)*n] = post_latent_4[idx]
        down_sampled_y[y*n:(y+1)*n] = true_y[idx]
        down_sampled_det_y[y*n:(y+1)*n] = det_y[idx]
    down_sampled_version = np.reshape(down_sampled_version, (down_sampled_version.shape[0], -1))
else:
    down_sampled_version = np.reshape(post_latent_4, (post_latent_4.shape[0], -1))
    down_sampled_y = true_y
    down_sampled_det_y = det_y


shuffleidx = np.arange(4*n)
np.random.shuffle(shuffleidx)
down_sampled_version = down_sampled_version[shuffleidx]
down_sampled_y = down_sampled_y[shuffleidx]
down_sampled_det_y = down_sampled_det_y[shuffleidx]

p = 5.

transformer = TSNE(perplexity=p)
transform = transformer.fit_transform(down_sampled_version)

labels = list(det_y_labels)

fig, ax = plt.subplots()
scatter = ax.scatter(transform[:,0], transform[:,1], c = down_sampled_det_y, cmap = 'tab10')
ax.legend(*scatter.legend_elements())
plt.title(f'Latent Space Visualization with t-SNE, perplexity = {p}')
plt.savefig(f'{pre_post}latent_space_{p}.png')


