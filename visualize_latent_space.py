from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tracemalloc


pre_post = ['pre', 'post']
layer = np.arange(5)
det_y_labels = [['bicycling', 'household-chores', 'manual-work', 'mixed-activity', 'sitting', 'sleep', 'sports', 'standing', 'vehicle', 'walking'], ['bicycling', 'gym', 'sitstand+activity', 'sitstand+lowactivity',
                'sitting', 'sleep', 'sports', 'standing', 'vehicle', 'walking', 'walking+activity'], ['bicycling', 'sedentary-non-screen', 'sedentary-screen', 'sleep','sport-interrupted', 'sports-continuous', 'tasks-light',
                'tasks-moderate', 'vehicle', 'walking']]

for pp in pre_post:
    for l in layer:
        latent = np.load(f'/Users/theb/Desktop/fold0/{pp}_latent{l}.npy')
        true_y = np.load(f'/Users/theb/Desktop/fold0/{pp}_y_test.npy')
        det_y = np.load(f'/Users/theb/Desktop/fold0/{pp}_det_y.npy')

        unique_y = np.unique(true_y)
        down_sample = True
        if down_sample:
            n = 1000
            down_sampled_version = np.zeros((4*n, *latent.shape[1:]))
            down_sampled_y = np.zeros(4*n)
            down_sampled_det_y = np.zeros((4*n, 3))

            for y in unique_y:
                idx = true_y == y
                idx = np.random.choice(np.where(idx)[0], n, replace = False)
                down_sampled_version[y*n:(y+1)*n] = latent[idx]
                down_sampled_y[y*n:(y+1)*n] = true_y[idx]
                down_sampled_det_y[y*n:(y+1)*n,:] = det_y[idx,:]
            down_sampled_version = np.reshape(down_sampled_version, (down_sampled_version.shape[0], -1))
        else:
            down_sampled_version = np.reshape(latent, (latent.shape[0], -1))
            down_sampled_y = true_y
            down_sampled_det_y = det_y


        shuffleidx = np.arange(4*n)
        np.random.shuffle(shuffleidx)
        down_sampled_version = down_sampled_version[shuffleidx]
        down_sampled_y = down_sampled_y[shuffleidx]
        down_sampled_det_y = down_sampled_det_y[shuffleidx]

        perplexity = [5, 10, 20, 30, 40, 50]

        for p in perplexity:
            transformer = TSNE(perplexity=p)
            transform = transformer.fit_transform(down_sampled_version)

            labels = ['light', 'moderate-vigorous', 'sedentary', 'sleep']

            fig, ax = plt.subplots()
            scatter = ax.scatter(transform[:,0], transform[:,1], c = down_sampled_y, cmap = 'Paired')
            ax.legend(handles=scatter.legend_elements()[0], labels=labels)
            plt.title(f'Latent Space Visualization with t-SNE, perplexity = {p}')
            plt.savefig(f'latent_plots/{pp}latent_space_{l}_{p}.png')

            for i in range(3):
                fig, ax = plt.subplots()
                scatter = ax.scatter(transform[:,0], transform[:,1], c = down_sampled_det_y[:,i], cmap = 'Paired')
                y_lim = ax.get_ylim()
                ax.set_ylim(y_lim[0], y_lim[1] + 50)
                ax.legend(handles=scatter.legend_elements()[0], labels=det_y_labels[i], ncol = 3, loc = 'upper center')
                plt.title(f'Latent Space Visualization with t-SNE, detailed labels {i+1}, perplexity = {p}')
                plt.savefig(f'latent_plots/{pp}latent_space_{l}_{p}_det_y_{i}.png')

pre_post = 'post'
layer = 5

det_y_idx = 0

post_latent_4 = np.load(f'/Users/theb/Desktop/fold0/{pre_post}_latent{layer}.npy')

det_y = np.load(f'/Users/theb/Desktop/fold0/{pre_post}_det_y.npy')
det_y_spec = det_y[:,det_y_idx]
det_y_labels_spec = det_y_labels[det_y_idx]
true_y = np.load(f'/Users/theb/Desktop/fold0/{pre_post}_y_test.npy')

unique_y = np.unique(true_y)
down_sample = False
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
        down_sampled_det_y[y*n:(y+1)*n] = det_y_spec[idx]
    down_sampled_version = np.reshape(down_sampled_version, (down_sampled_version.shape[0], -1))
else:
    down_sampled_version = np.reshape(post_latent_4, (post_latent_4.shape[0], -1))
    down_sampled_y = true_y
    down_sampled_det_y = det_y_spec

nn = NearestNeighbors(n_neighbors=4).fit(down_sampled_version)
nearest_neighbors = nn.kneighbors(down_sampled_version, return_distance=False)[:,1:]
nn_matrix = np.zeros((4*n, 4*n))
for i in range(4*n):
    nn_matrix[i, nearest_neighbors[i]] = 1

dist = nn_matrix.sum(axis = 0)
plt.hist(dist, bins = 100)
plt.savefig('hubness_hist.png')



shuffleidx = np.arange(4*n)
np.random.shuffle(shuffleidx)
down_sampled_version = down_sampled_version[shuffleidx]
down_sampled_y = down_sampled_y[shuffleidx]
down_sampled_det_y = down_sampled_det_y[shuffleidx]

p = 10.

transformer = TSNE(perplexity=p)
transform = transformer.fit_transform(down_sampled_version)

labels = list(det_y_labels)

fig, ax = plt.subplots()
scatter = ax.scatter(transform[:,0], transform[:,1], c = down_sampled_det_y, cmap = 'Paired')
y_lim = ax.get_ylim()
ax.set_ylim(y_lim[0], y_lim[1] + 50)
ax.legend(handles=scatter.legend_elements()[0], labels=det_y_labels_spec, ncol = 3, loc = 'upper center')
plt.title(f'Latent Space Visualization with t-SNE, perplexity = {p}')
plt.savefig(f'{pre_post}latent_space_{p}.png')


