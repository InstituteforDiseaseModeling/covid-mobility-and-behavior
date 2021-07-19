  
"""Module for applying dimensionality reduction and manifold learning methods to 
the SafeGraph stay-at-home time series data during COVID-19 pandemic.
"""
import seaborn as sns
from sklearn.manifold import trustworthiness
import numpy as np
import scipy
import matplotlib.pyplot as plt 
import random
from mpl_toolkits.mplot3d import Axes3D
from copy import copy
from matplotlib.colors import ListedColormap
import os
import pandas as pd
from src.config import configurations
import scipy
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
import geopandas as gpd
from src.utils import define_paths, load_data, convert_to_python_dict, save_obj, load_obj, find_closest_time_series
from src.utils import plot_time_series, find_cos_similarity, intersection, relabel_clusters

#for PCA
def compute_covariance(X):
    """
    Returns the mean and covariance matrix of the demeaned dataset X (e.g. for PCA)
    
    Parameters
    ----------
    X : np.array
        data

    Returns
    -------
    mu : np.array
        mean
    C : np.array : 
        Covariance Matrix
    
    """
    mu = np.mean(X, axis = 0)
    C = (X-mu).T.dot(X-mu)/len(X)
    return mu, C

def explained_variance(X):
    """
    Returns the fraction of explained variance as a function of the PCA component number
    
    Parameters
    ----------
    X : np.array
        data

    Returns
    -------
    fraction_exp_var : np.array
        fraction of explained variance as a function of the PCA component number  
    """
    mu, Sigma = compute_covariance(X)
    l, V = np.linalg.eig(Sigma)
    sorted_indices = np.argsort(l)[::-1]
    l = l[sorted_indices].astype('float')
    fraction_exp_var = np.cumsum(l)/np.sum(l)
    return fraction_exp_var

def return_metric(method, model):
    """
    Returns reconstruction error metric for a dim-reduction method
    
    Parameters
    ----------
    method : sklearn class
        embedding method with well-defined .fit_transform method, usually from sklearn.manifold
    model : sklearn class
        fitted model

    Returns
    -------
    float or None
        Reconstruction error value for a fitted model or None if the metric is not defined

    Notes
    -------
    So far, the error metric is only defined for Isomap
    """
    if method.__name__ == 'Isomap':
        return model.reconstruction_error()
    if method.__name__ == 'TSNE':
        return None # TODO
    return None

#Choose optimal dimension for a particular dim reduction method
#Also works for choosing arbitrary hyperparameters, choose_dimension is a bit of an unfortunate name for the function
def choose_dimension(X, emb_method, hyperparams_to_test, **kwargs):
    """
    Vizualizes grid of 2D embeddings varying hyperparams_to_test, and 3D projection of the embedding with default parameters
    
    Parameters
    ----------
    X : np.array
        data
    emb_method : sklearn class
        embedding method with well-defined .fit_transform method, usually from sklearn.manifold
    hyperparams_to_test : dict
        dictionary with hyperparams to test as keys and their respective ranges as values
    kwargs : dict
        hyperparams of the method (including the default values for the ones to vary)

    Returns
    -------
    errors_results : dict
        dictionary with reconstruction errors as values, keys -- varied hyperparams
    embeddings_results : dict
        dictionary with embeddings as values, keys -- varied hyperparams
    trustws_results : dict
        dictionary with trustworthiness metrics as values, keys -- varied hyperparams

    Notes
    -------
    For 3D projection, default parameters are used
    """
    sns.set_style("whitegrid", {'axes.grid' : False})
    #Default params:
    default_args = copy(kwargs)
    embeddings_results = {} #dict with resulting embeddings with keys -- varied hyperparams, values: different embs
    errors_results = {} #dict with lists of errors as values, keys -- varied hyperparams
    trustws_results = {} #dict with lists of trustworthiness as values, keys -- varied hyperparams
    for range_key in hyperparams_to_test.keys():
        #Set to default kwargs:
        kwargs = copy(default_args)
        X_embs = []
        errors = []
        trustws = [] 
        for hyperparam in hyperparams_to_test[range_key]:
            print('Trying {}={}'.format(range_key, hyperparam))
            kwargs[range_key] = hyperparam
            model = emb_method(**kwargs)
            # model.fit(X)
            X_embedded = model.fit_transform(X)
            X_embs.append(X_embedded)
            errors.append(return_metric(emb_method, model))
            #the idea behind this is to test how good local structure is preserved, thus n_neighbors=5 is used, methods had more neighbors
            trustws.append(trustworthiness(X, X_embedded, n_neighbors=5, metric='euclidean'))
        embeddings_results[range_key] = X_embs
        errors_results[range_key] = errors
        trustws_results[range_key] = trustws
    return errors_results, embeddings_results, trustws_results


def visualize_manifold_method(X, emb_method, hyperparams_to_test, colors, filename = None, load_path = None, save_path = None, alpha = None, subselect = slice(None), cbar = None, ax = None, **kwargs):
    """
    Vizualizes grid of 2D embeddings varying hyperparams_to_test. Visualizes 2D and 3D embeddings with default parameters separately.
    
    Parameters
    ----------
    X : np.array
        data
    emb_method : sklearn class
        embedding method with well-defined .fit_transform method, usually from sklearn.manifold
    hyperparams_to_test : dict
        dictionary with hyperparams to test as keys and their respective ranges as values
    colors : np.array
        Array of colors,ust be of the same shape with X. Usually this would be clustering colors
    filename : str
        Path to save figures, None by default. If None, figures are not saved
    load_path : str
        For the subsequent 2D and 3D visualization, path to load preprocessed embeddings from to avoid lengthy computations
    save_path : str
        For the subsequent 2D and 3D visualization, path to save embeddings to to avoid lengthy computations in the future
    alpha : float 
        Number between 0.0 and 1.0, transparency for the 2D and 3D plots
    subselect : slice
        Slice to subselect samples to visualize. Only subselects for subsequent 2D and 3D projections. Helpful to throw away clusters.
    cbar : str
        Type of colorscheme. Could be 'clust' for discrete colorscheme or 'colormap' for continuous colorscheme. cbar is an unfortunate name.
    ax : matplotlib ax object
         ax artist to use for plotting within external figures. If None, figures are created here and both 2D and 3D are visualized    
    kwargs : dict
        hyperparams of the method (including the default values for the ones to vary)
    

    Returns
    -------
    X_2D_emb : np.array
        2D embedding, could be None if axis is specified
    X_3D_emb : np.array
        3D embedding

    Notes
    -------
    For 3D projection, default parameters are used
    hyperparams_to_test could be empty, the code would just visualize 3D and possibly 2D projections
    if ax is None, both 2D and 3D default projections are visualized. If ax is not None, then only 3D is visualized.
  
    """
    # IMG_PATH = "/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/reports/figures/exploratory/covid/dim_red_viz/"+name_prefix
    # FIGURES_PATH = '/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/reports/figures/'
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

    sns.set_style("whitegrid", {'axes.grid' : False})
    # sns.set_palette('colorblind')
    if cbar == 'clustering':
#         friendly_cmap = ListedColormap(sns.color_palette('colorblind', len(np.unique(colors))).as_hex())
        almost_sequential_pal = configurations['clustering_palette']#['#1f78b4','#a6cee3','#fdbf6f','#ff7f00', '#cc78bc']
        friendly_cmap = ListedColormap(sns.color_palette(almost_sequential_pal, len(np.unique(colors))).as_hex())
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
        if filename is not None:
            fig.savefig(filename+'Viz_2D_{}_{}.png'.format(key, emb_method.__name__), bbox_inches = 'tight')
    
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
        # ax.scatter(X_3D_emb[subselect,0], X_3D_emb[subselect, 1], X_3D_emb[subselect, 2], c=colors, alpha = alpha, cmap = friendly_cmap)
        print('Len of colors {}'.format(len(colors)))
        ax.scatter(X_3D_emb[subselect,0], X_3D_emb[subselect, 1], X_3D_emb[subselect, 2], c=colors, alpha = alpha, cmap = friendly_cmap)
        ax.set_xlabel('{} 0'.format(emb_method.__name__))
        ax.set_ylabel('{} 1'.format(emb_method.__name__))
        ax.set_zlabel('{} 2'.format(emb_method.__name__))
        ax.set_title('{} 3D projection'.format(emb_method.__name__))
        # ax.view_init(30, 200)
        print('3D Args:', kwargs)
        return None, X_3D_emb


    return X_2D_emb, X_3D_emb

def viz_SE(X, colors, filename = None, alpha = None, cbar = None, subselect = slice(None), ax = None, load_path = None, save_path = None): 
    #Try Spectral Embedding

    emb_method = SpectralEmbedding

    hyperparams_to_test = {}#{'n_neighbors': [50, 50]}
#     colors = colors#labels_multirun['Labels_3']#SE_clusters['x'].values
    SE_X_2D_emb, SE_X_3D_emb = visualize_manifold_method(X, emb_method, hyperparams_to_test, colors, filename = filename,
                                                         alpha = alpha, cbar = cbar, subselect = subselect, ax = ax, 
                                                         load_path = load_path, save_path = save_path, **{'n_neighbors': 50,
     'n_components': 2,
     'n_jobs': -1})
    
#     SE_X_2D_emb, SE_X_3D_emb = visualize_manifold_method(X, emb_method, hyperparams_to_test, colors, filename = None, load_path = None, save_path = None, alpha = None, subselect = slice(None), cbar = None, ax = None, **kwargs)
    
    return SE_X_2D_emb, SE_X_3D_emb

from pydiffmap import diffusion_map as dm

# def dm_wrapper(n_neighbors, n_components, bandwidth_type, alpha_dm, epsilon):
#     return dm.DiffusionMap.from_sklearn(n_evecs=n_components, epsilon=epsilon, alpha=alpha_dm, k=n_neighbors, bandwidth_type=bandwidth_type)
def dm_wrapper(n_neighbors, n_components, bandwidth_type, alpha_dm, epsilon):
    if alpha_dm is None:
        return dm.DiffusionMap.from_sklearn(n_evecs=n_components, epsilon=epsilon, alpha=(1/2-n_components/4), k=n_neighbors, bandwidth_type=bandwidth_type)
    else:
        return dm.DiffusionMap.from_sklearn(n_evecs=n_components, epsilon=epsilon, alpha=alpha_dm,
                                            k=n_neighbors, bandwidth_type=bandwidth_type)


def viz_DM(X, colors, filename=None, alpha=None, cbar=None, subselect=slice(None), ax=None, load_path=None,
           save_path=None, dm_bandwidth = 'fixed'):
    # Try Spectral Embedding

    emb_method = dm_wrapper

    hyperparams_to_test = {}  # {'n_neighbors': [50, 50]}

    #Fixed bandwidth
    if dm_bandwidth == 'fixed':
        SE_X_2D_emb, SE_X_3D_emb = visualize_manifold_method(X, emb_method, hyperparams_to_test, colors, filename=filename,
                                                             alpha=alpha, cbar=cbar, subselect=subselect, ax=ax,
                                                             load_path=load_path, save_path=save_path, **{'n_neighbors': 50,
                                                                                                          'n_components': 2,
                                                                                                          'bandwidth_type': None,
                                                                                                          'alpha_dm':1.0,
                                                                                                          'epsilon': 0.5})
    #Variable bandwidth
    if dm_bandwidth == 'variable':
        SE_X_2D_emb, SE_X_3D_emb = visualize_manifold_method(X, emb_method, hyperparams_to_test, colors, filename=filename,
                                                             alpha=alpha, cbar=cbar, subselect=subselect, ax=ax,
                                                             load_path=load_path, save_path=save_path, **{'n_neighbors': 100,
                                                                                                          'n_components': 2,
                                                                                                          'bandwidth_type': -1/2,
                                                                                                          'alpha_dm': 1/2 - 15/4,
                                                                                                          'epsilon': 0.0002})
    return SE_X_2D_emb, SE_X_3D_emb
        
def viz_Isomap(X, colors, filename = None, alpha = None, cbar = None, subselect = slice(None), ax = None, load_path = None, save_path = None):
    emb_method = Isomap
    hyperparams_to_test = {}#{'n_neighbors': [50, 50]}
#     colors = colors#SE_clusters['x'].values#lusters[:,1]

    Isomap_X_2D_emb, Isomap_X_3D_emb = visualize_manifold_method(X, emb_method, hyperparams_to_test, colors,
                              filename = filename, 
                              alpha = alpha, cbar = cbar, subselect = subselect, ax = ax, 
                                                                 load_path = load_path, save_path = save_path, **{'n_neighbors': 50, 
                                                                    'n_components': 2, 
                                                                    'max_iter': 1000, 
                                                                    'n_jobs': -1})
    return Isomap_X_2D_emb, Isomap_X_3D_emb
    
def viz_LLE(X, colors, filename = None, alpha = None, cbar = None, subselect = slice(None), ax = None, load_path = None, save_path = None):
    #Try LLE, standard, tweaked parameters a little

    emb_method = LocallyLinearEmbedding
    hyperparams_to_test = {}#'n_neighbors': [200, 200],
    # colors = clusters[:,1]


    LLE_X_2D_emb, LLE_X_3D_emb = visualize_manifold_method(X, emb_method, hyperparams_to_test, colors,
                              filename = filename, 
                              alpha = alpha, cbar = cbar, subselect = subselect, ax = ax, load_path = load_path, 
                                                           save_path = save_path, **{'n_neighbors': 200,
                                                                       'n_components': 2,
                                                                       'reg': 0.1,
                                                                       'eigen_solver': 'auto',
                                                                       'tol': 1e-06,
                                                                       'max_iter': 1000,
                                                                       'method': 'standard',
                                                                       'hessian_tol': 0.0001,
                                                                       'modified_tol': 1e-12,
                                                                       'neighbors_algorithm': 'auto',  
                                                                       'random_state': None,
                                                                       'n_jobs': -1})
    return LLE_X_2D_emb, LLE_X_3D_emb
    
    
def viz_cluster_map(colors, index_X, filename = None, title = None, cbar_label = None, cmap = None, ax = None, state = None, edgecolor = 'lightgrey', linewidth=0.1):#, state_shape_df_ = state_shape_df_):
#     FIGURES_PATH = '/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/reports/figures/'
    #FIX THIS FOR OTHER STATES!
#     sns.set_palette("tab10")
    SHAPE_PATH, FIGURE_PATH, RAW_DATA_PATH, INCOME_POPULATION_PATH = define_paths(state)

    if cmap is not None:
        almost_sequential_pal = configurations['clustering_palette']#['#1f78b4','#a6cee3','#fdbf6f','#ff7f00', '#cc78bc']
        friendly_cmap = ListedColormap(sns.color_palette(almost_sequential_pal, len(np.unique(colors))).as_hex())#'colorblind'
    else:
        friendly_cmap = None
    state_shape_df = gpd.read_file(SHAPE_PATH)
    state_shape_df_ = state_shape_df.copy()
    state_shape_df_.GEOID = state_shape_df_.GEOID.astype('int')
    state_shape_df_ = state_shape_df_.set_index('GEOID')
    
    colors = pd.DataFrame(colors, columns = ['c'])
    colors['block_ind'] = index_X
    colors = colors.set_index('block_ind')
    state_shape_df_['colormap'] = colors['c']
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('off')
        state_shape_df_.plot(column = 'colormap', ax = ax, legend = True, edgecolor="face", cmap = friendly_cmap, linewidth=0.1, legend_kwds={'label': cbar_label,'orientation': "horizontal"})
        ax.set(title= title)
    else:
        ax.axis('off')
        if np.any(state_shape_df_.isnull()):#if there is missing data
            missing_kwds={'color': 'lightgrey'}
        else:
            missing_kwds = None
        state_shape_df_.plot(column = 'colormap', ax = ax, legend = False, edgecolor=edgecolor, cmap = friendly_cmap, linewidth=linewidth, missing_kwds=missing_kwds)
        ax.set(title = title)

    if filename is not None:
        fig.savefig(filename + '.png', bbox_inches = 'tight')

if __name__ == "__main__":
    pass