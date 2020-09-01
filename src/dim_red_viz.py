import numpy as np
import scipy
import matplotlib.pyplot as plt 
import seaborn as sns
import random
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
import seaborn as sns
from matplotlib.colors import ListedColormap
import os


def visualize_manifold_method(X, emb_method, hyperparams_to_test, colors, name_prefix, filename = None, load_path = None, save_path = None, alpha = None, subselect = slice(None), cbar = None, ax = None, **kwargs):
    """
    Vizualizes grid of 2D embeddings varying hyperparams_to_test, and 3D projection of one of them
    method: sklearn method
    hyperparams_to_test: dictionary with keys hyperparams and values ranges
    kwargs: all other hyperparams of the method
    X: data
    For 3D projection, initial parameters are used
    subselect: only subselects for later 2D and 3D projections
    returns: X_2D_emb, X_3D_emb
    """
    IMG_PATH = "/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/reports/figures/exploratory/covid/dim_red_viz/"+name_prefix
    # FIGURES_PATH = '/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/reports/figures/'
    sns.set_style("whitegrid", {'axes.grid' : False})
    sns.set_palette('colorblind')
    if cbar == 'clustering':
        friendly_cmap = ListedColormap(sns.color_palette('colorblind', len(np.unique(colors))).as_hex())
    if cbar == 'colormap':
        friendly_cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    if cbar is None:
        friendly_cmap = None
    #TODO add scaling
    #Default params:
    default_args = copy(kwargs)
    embeddings_results = {} #dict with resulting embeddings with keys -- varied hyperparams, values: different embs
    for range_key in hyperparams_to_test.keys():
        #Set to default kwargs:
        kwargs = copy(default_args)
        X_embs = []
        for hyperparam in hyperparams_to_test[range_key]:
            kwargs[range_key] = hyperparam
            model = emb_method(**kwargs)
            # model.fit(X)
            X_embs.append(model.fit_transform(X))
        embeddings_results[range_key] = X_embs
        
    for key in embeddings_results.keys():
        X_embs = embeddings_results[key]
        fig, axs = plt.subplots(2, len(X_embs)//2, figsize=(30,15))#, #sharex=True, sharey=True, figsize=(20,20))
        fig.suptitle(emb_method.__name__)
        plt.subplots_adjust(top=0.93)
    #     colors = covid_clusters['x'].copy()
    #     colors[colors != 11] = 0
        for num, ax_ in enumerate(axs.flatten()):
            ax_.scatter(X_embs[num][:, 0], X_embs[num][:, 1],  c = colors, cmap = friendly_cmap)
            ax_.title.set_text('{} {}'.format(key, hyperparams_to_test[key][num]))
        plt.savefig(IMG_PATH+'Viz_2D_{}_{}.png'.format(key, emb_method.__name__), bbox_inches = 'tight')
    
    kwargs = copy(default_args) #set to default
    kwargs['n_components'] = 2 #set to two to plot with default
    model = emb_method(**kwargs)
    # model.fit(X)
    if ax is None:
        if load_path is None:
            X_2D_emb = model.fit_transform(X)
            if save_path is not None:
                np.save(os.path.join(save_path, 'X_2D_emb.npy'), X_2D_emb)
        else:
            X_2D_emb = np.load(os.path.join(load_path, 'X_2D_emb.npy'))
        # current_palette = sns.color_palette()
        # sns.palplot(current_palette)
        
        fig = plt.figure(figsize = (15,10))
        ax_ = fig.add_subplot(111)#, projection='3d')

        scatter_2d = ax_.scatter(X_2D_emb[subselect,0], X_2D_emb[subselect, 1], c=colors, alpha = alpha, cmap = friendly_cmap)
        if cbar == 'clustering':
            ticks = np.sort(np.unique(colors))[::-1]
            cbar_clust = fig.colorbar(scatter_2d, ticks = ticks)
            # print('CHECK CBAR:', np.sort(np.unique(colors))[::-1])
            cbar_clust.ax.set_yticklabels(['Cluster ' + str(i) for i in ticks])
        if cbar == 'colormap':
            fig.colorbar(scatter_2d)
        ax_.set_xlabel('{} 0'.format(emb_method.__name__))
        ax_.set_ylabel('{} 1'.format(emb_method.__name__))
        ax_.set_title('{} 2D projection'.format(emb_method.__name__))
        print('2D Args:', kwargs)
        
        # plt.savefig(IMG_PATH+'Viz_2D_{}.png'.format(emb_method.__name__), bbox_inches = 'tight')
        if filename is not None:
            fig.savefig(filename+'_2D.png', bbox_inches = 'tight')

    ########3D:##########
    kwargs = copy(default_args) #set to default
    kwargs['n_components'] = 3
    model = emb_method(**kwargs)
    # model.fit(X)
    if load_path is None:
        X_3D_emb = model.fit_transform(X)
        if save_path is not None:
            np.save(os.path.join(save_path, 'X_3D_emb.npy'), X_3D_emb)
    else:
        X_3D_emb = np.load(os.path.join(load_path, 'X_3D_emb.npy'))
    
    if ax is None:
        fig = plt.figure(figsize = (15,10))
        ax_ = fig.add_subplot(111, projection='3d')

        ax_.scatter(X_3D_emb[subselect,0], X_3D_emb[subselect, 1], X_3D_emb[subselect, 2], c=colors, alpha = alpha, cmap = friendly_cmap)

        ax_.set_xlabel('{} 0'.format(emb_method.__name__))
        ax_.set_ylabel('{} 1'.format(emb_method.__name__))
        ax_.set_zlabel('{} 2'.format(emb_method.__name__))
        ax_.set_title('{} 3D projection'.format(emb_method.__name__))
        ax_.view_init(30, 200)
        print('3D Args:', kwargs)
        
        # plt.savefig(IMG_PATH+'Viz_3D_{}.png'.format(emb_method.__name__), bbox_inches = 'tight')
        if filename is not None:
            fig.savefig(filename+'_3D.png', bbox_inches = 'tight')
    
    if ax is not None:
        ax.scatter(X_3D_emb[subselect,0], X_3D_emb[subselect, 1], X_3D_emb[subselect, 2], c=colors, alpha = alpha, cmap = friendly_cmap)
        ax.set_xlabel('{} 0'.format(emb_method.__name__))
        ax.set_ylabel('{} 1'.format(emb_method.__name__))
        ax.set_zlabel('{} 2'.format(emb_method.__name__))
        ax.set_title('{} 3D projection'.format(emb_method.__name__))
        # ax.view_init(30, 200)
        print('3D Args:', kwargs)
        return None, X_3D_emb


    return X_2D_emb, X_3D_emb

    
if __name__ == "__main__":
    pass