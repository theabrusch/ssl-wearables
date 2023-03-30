from sklearn.manifold import TSNE
import umap
import numpy as np
import matplotlib.pyplot as plt

post_latent_4 = np.load('/Users/theb/Desktop/fold0/post_latent4.npy')
true_y = np.load('/Users/theb/Desktop/fold0/pre_y_test.npy')

unique_y = np.unique(true_y)
down_sample = True
if down_sample:
    n = 10000
    down_sampled_version = np.zeros((4*n, post_latent_4.shape[1], post_latent_4.shape[2]))
    down_sampled_y = np.zeros(4*n)

    for y in unique_y:
        idx = true_y == y
        idx = np.random.choice(np.where(idx)[0], n, replace = False)
        down_sampled_version[y*n:(y+1)*n] = post_latent_4[idx]
        down_sampled_y[y*n:(y+1)*n] = true_y[idx]
    down_sampled_version = np.reshape(down_sampled_version, (down_sampled_version.shape[0], -1))
else:
    down_sampled_version = np.reshape(post_latent_4, (post_latent_4.shape[0], -1))
    down_sampled_y = true_y

shuffleidx = np.arange(4*n)
np.random.shuffle(shuffleidx)
down_sampled_version = down_sampled_version[shuffleidx]
down_sampled_y = down_sampled_y[shuffleidx]

transformer = umap.UMAP()
transform = transformer.fit_transform(down_sampled_version)

colors = ['red', 'green', 'blue', 'orange']
c = [colors[int(y)] for y in down_sampled_y]

plt.scatter(transform[:,0], transform[:,1], c = c)
plt.show()