#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Imports
from src.PCA import ComputeCovariance
from src.dim_red_viz import visualize_manifold_method
#REDO this to just import a function
from src.data.dimensionality_reduction.choose_dimension import choose_dimension
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding
from copy import copy 
from sklearn.manifold import trustworthiness
from kneed import DataGenerator, KneeLocator
from rpy2.robjects.conversion import localconverter



STATE = 'tx'


#Load the data
# covid_ga = pd.read_csv('/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/data/raw/covid/stayathome-texas.csv')
covid_state = pd.read_csv('sample.tar.gz', compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False)
covid_ga_ = covid_ga.set_index('Unnamed: 0')
covid_ga_ = covid_ga_[covid_ga_.columns[:117]] #to consider the same time period with our WA analysis
#Prepare the data: drop nans
X = covid_ga_.dropna().values
index_X = covid_ga_.dropna().index
columns_X = covid_ga_.dropna().columns
X.shape, covid_ga_.shape


# In[4]:


#Define functions
import scipy
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import Isomap



def find_closest_time_series(SE_behavior, SE_means, covid_, n = 5):
    """Finds n closest time series to the cluster means (or the vectors in SE_means)
    returns dict with keys V1...V6 cluster names and values dataframes with the n closest samples
    same for block_ids    
    """
    closest_to_mean_samples = {}
    closest_to_mean_block_ids = {}
    for cluster in SE_means.columns:
        sorted_indices = np.squeeze(np.argsort(scipy.spatial.distance.cdist(SE_behavior, SE_means[cluster][None,:], metric='euclidean'), axis = 0))
        closest_to_mean_samples[cluster] = covid_.loc[index_X[sorted_indices[:n]]] #Take the closest n points
        closest_to_mean_block_ids[cluster] = index_X[sorted_indices[:n]]
    return closest_to_mean_samples, closest_to_mean_block_ids


def plot_time_series(time_series_df, name = 'time_series_of_interest', filename = None, only_melt = False):
    FIGURES_PATH = '/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/reports/figures/'
    time_series_df_ = time_series_df.copy().T
    time_series_df_['dates'] = pd.to_datetime(time_series_df.columns)
    #closest_to_mean_sample_ = closest_to_mean_sample.T
    #closest_to_mean_sample_['dates'] = pd.to_datetime(columns_X)
#     closest_to_mean_sample_
    #Cluster means figure
    sns.set(style="darkgrid")
    melted_df = pd.melt(time_series_df_,
                        id_vars=['dates'], # Variables to keep
                        var_name="Block")
    if only_melt:
        return melted_df
    g = sns.relplot(x = 'dates', y="value", row = 'Block', 
                    kind="line", row_order=time_series_df.index, data=melted_df, sort = False, aspect = 5, marker='o')
    g.fig.autofmt_xdate(0,10)
    plt.subplots_adjust(top=0.93)
    g.fig.suptitle(name, size = 'xx-large')
    g.set(ylim=(0, 1))


    #Add the important dates
    # ax = g.axes[1]

    for axs in g.axes:
        axs[0].axvline(datetime(2020, 2, 29), color = 'green', ls='--', label='emergency_WA')
        axs[0].axvline(datetime(2020, 3, 3), color = 'blue', ls='--', label='emergency_Seattle')
        axs[0].axvline(datetime(2020, 3, 11), color = 'red', ls='--', label='events_ban')
        axs[0].axvline(datetime(2020, 3, 12), color = 'brown', ls='--', label='schools')
        axs[0].axvline(datetime(2020, 3, 15), color = 'green', ls='-', label='restrautrants')
        axs[0].axvline(datetime(2020, 3, 21), color = 'blue', ls='-', label='Everett_stay_home')
        axs[0].axvline(datetime(2020, 3, 23), color = 'red', ls='-', label='stay-at-home')
        axs[0].axvline(datetime(2020, 3, 25), color = 'brown', ls='-', label='WA_parks_closed')
        axs[0].axvline(datetime(2020, 3, 31), color = 'green', ls='-.', label='National_guard_WA')
        axs[0].axvline(datetime(2020, 4, 2), color = 'blue', ls='-.', label='stay-at-home_extended')

        axs[0].legend()

    # ax[0].text(0.5,25, "Some text")
    if filename is not None:
        g.fig.savefig(FIGURES_PATH+filename, bbox_inches = 'tight')
    
    return melted_df
        
def viz_blocks_on_the_map(block_ids, name = 'blocks_of_interest', filename = None):#, washington_ = washington_):
    FIGURES_PATH = '/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/reports/figures/'
    washington = gpd.read_file('/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/data/external/washington_blockgroup_shape/tl_2015_53_bg/tl_2015_53_bg.shp')
#     washington_ = washington.copy()
#     washington_.GEOID = washington_.GEOID.astype('int')
#     washington_ = washington_.set_index('GEOID')
    washington['blocks_of_interest'] = False
    washington['blocks_of_interest'][washington.GEOID.astype('int').isin(block_ids.astype('int'))] = True
#     colors = pd.DataFrame(colors, columns = ['c'])
#     colors['block_ind'] = index_X
#     colors = colors.set_index('block_ind')
#     washington_['colormap'] = colors['c']
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    washington.plot(column = 'blocks_of_interest', ax = ax, legend = False, edgecolor="face", cmap = 'OrRd',
                    linewidth=0.4, legend_kwds={'label': 'Cluster','orientation': "horizontal"})
    ax.set(title=name)
    if filename is not None:
        fig.savefig(FIGURES_PATH+filename, bbox_inches = 'tight')
        
def viz_SE(X, colors, filename = None, alpha = None, subselect = slice(None)): 
    #Try Spectral Embedding
    from sklearn.manifold import SpectralEmbedding

    emb_method = SpectralEmbedding

    hyperparams_to_test = {}#{'n_neighbors': [50, 50]}
#     colors = colors#labels_multirun['Labels_3']#SE_clusters['x'].values
    SE_X_2D_emb, SE_X_3D_emb = visualize_manifold_method(X, emb_method, hyperparams_to_test, colors, '', filename = filename, alpha = alpha, subselect = subselect, **{'n_neighbors': 50,
     'n_components': 2,
     'n_jobs': -1})
    
    return SE_X_2D_emb, SE_X_3D_emb


        
def viz_Isomap(X, colors, filename = None, alpha = None, subselect = slice(None)):
    emb_method = Isomap
    hyperparams_to_test = {}#{'n_neighbors': [50, 50]}
#     colors = colors#SE_clusters['x'].values#lusters[:,1]

    Isomap_X_2D_emb, Isomap_X_3D_emb = visualize_manifold_method(X, emb_method, hyperparams_to_test, colors,
                              '', filename = filename, 
                              alpha = alpha, subselect = subselect, **{'n_neighbors': 50, 
                                                                    'n_components': 2, 
                                                                    'max_iter': 1000, 
                                                                    'n_jobs': -1})
    return Isomap_X_2D_emb, Isomap_X_3D_emb
    
def viz_LLE(X, colors, filename = None, alpha = None, subselect = slice(None)):
    #Try LLE, standard, tweaked parameters a little
    from sklearn.manifold import LocallyLinearEmbedding

    emb_method = LocallyLinearEmbedding
    hyperparams_to_test = {}#'n_neighbors': [200, 200],
    # colors = clusters[:,1]


    LLE_X_2D_emb, LLE_X_3D_emb = visualize_manifold_method(X, emb_method, hyperparams_to_test, colors,
                              '', filename = filename, 
                              alpha = alpha, subselect = subselect, **{'n_neighbors': 200,
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
    
    
def viz_cluster_map(colors, index_X, filename = None):#, washington_ = washington_):
    FIGURES_PATH = '/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/reports/figures/'
    #FIX THIS FOR OTHER STATES!
    washington = gpd.read_file('/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/data/external/texas_blockgroup_shape/tl_2016_48_bg/tl_2016_48_bg.shp')
    washington_ = washington.copy()
    washington_.GEOID = washington_.GEOID.astype('int')
    washington_ = washington_.set_index('GEOID')
    
    colors = pd.DataFrame(colors, columns = ['c'])
    colors['block_ind'] = index_X
    colors = colors.set_index('block_ind')
    washington_['colormap'] = colors['c']
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    washington_.plot(column = 'colormap', ax = ax, legend = True, edgecolor="face", linewidth=0.4, legend_kwds={'label': 'Cluster','orientation': "horizontal"})
    ax.set(title='GMM Clustering')
    if filename is not None:
        fig.savefig(FIGURES_PATH + filename, bbox_inches = 'tight')

        
def create_walk_df(X_emb, similarity, index):
    """X_emb -- embedding,
    similarity -- to sort by in the walk, colormap, the same dimension with X_emb
    index -- non-artifact index e.g., to subselect
    """
    similarities_non_artifact = similarity[index]
    X_walk = X_emb[index]
    X_walk_df = pd.DataFrame(X_walk, columns = ['Mode_{}'.format(i) for i in range(X_walk.shape[1])])
    X_walk_df['Similarity'] = similarities_non_artifact
    return X_walk_df.sort_values(by = 'Similarity')#.reset_index(drop = True)

def find_cos_similarity(X_2D_emb):
    right_end = X_2D_emb[np.argmax(X_2D_emb, axis = 0)[0]]
    left_end = X_2D_emb[np.argmin(X_2D_emb, axis = 0)[0]]
    left_end, right_end, np.argmin(X_2D_emb, axis = 0)[0]
    center = (left_end + right_end)/2
    cosine_colors = 1-cosine_similarity([right_end - center], X_2D_emb-center, dense_output=True).flatten()
    return cosine_colors


# In[5]:


def return_metric(method, model):
    if method.__name__ == 'Isomap':
        return model.reconstruction_error()
    if method.__name__ == 'TSNE':
        return None # TODO
    return None
    #TODO other methods too


# ### Choose optimal dimensionality using Isomap, SE and LLE

# In[31]:


get_ipython().run_cell_magic('time', '', "#Let's try isomap first\n#The best parameters from the vizualization are taken from previous analysis: 50 neighbors\n#let's try several n_components\nemb_method = Isomap\nhyperparams_to_test = {'n_components': np.arange(2, 117, 2)} #CHANGE THIS TO 117!!!\n\nerrors_results, embeddings_results, trustws_results = choose_dimension(X, emb_method, hyperparams_to_test, '', **{'n_neighbors': 50, \n                                                                             'n_components': 2, \n                                                                             'max_iter': 1000,\n                                                                             'n_jobs': -1})\n# hyperparams_to_test = {'n_components': [16]}\n\n# errors_results, embeddings_results, trustws_results = choose_dimension(X, emb_method, hyperparams_to_test, '', **{'n_neighbors': 50, \n#                                                                              'n_components': 2, \n#                                                                              'max_iter': 10,\n#                                                                              'n_jobs': -1})\n\n#UNCOMMENT THE ABOVE")


# In[ ]:


get_ipython().run_cell_magic('time', '', "#Let's try isomap first\n#The best parameters from the vizualization are taken from previous analysis: 50 neighbors\n#let's try several n_components\nemb_method = Isomap\nhyperparams_to_test = {'n_components': np.arange(30, 117, 2)} #CHANGE THIS TO 117!!!\n\nerrors_results_Isomap, embeddings_results_Isomap, trustws_results_Isomap = choose_dimension(X, emb_method, hyperparams_to_test, '', **{'n_neighbors': 50, \n                                                                             'n_components': 2, \n                                                                             'max_iter': 1000,\n                                                                             'n_jobs': -1})\n# hyperparams_to_test \n= {'n_components': [16]}\n\n# errors_results, embeddings_results, trustws_results = choose_dimension(X, emb_method, hyperparams_to_test, '', **{'n_neighbors': 50, \n#                                                                              'n_components': 2, \n#                                                                              'max_iter': 10,\n#                                                                              'n_jobs': -1})\n\n#UNCOMMENT THE ABOVE")


# In[32]:


embeddings_results['n_components'][0].shape


# In[33]:


get_ipython().run_line_magic('matplotlib', 'notebook')
# max_components = 30
#Look at the errors Isomap gives
plt.plot(np.arange(2, max_components, 2), errors_results['n_components'])


# In[34]:


# x, y = enumerate(errors['n_components'])
get_ipython().run_line_magic('matplotlib', 'notebook')
kneedle = KneeLocator(np.arange(2, max_components, 2), errors_results['n_components'], S=1.0, curve='convex', direction='decreasing')#, interp_method='polynomial')\
kneedle.plot_knee()
plt.title(emb_method.__name__ + ' Errors')
plt.xlabel('n_components')
plt.ylabel('Isomap reconstruction error')
kneedle.elbow


# In[36]:


kneedle = KneeLocator(np.arange(2, max_components, 2), np.array(trustws_results['n_components']), S=1, curve='concave', direction='increasing', interp_method='polynomial', online = False)
kneedle.plot_knee()
plt.title(emb_method.__name__  + ' trustworthiness')
plt.xlabel('n_components')
plt.ylabel('trustworhiness')
kneedle.knee, kneedle.knee_y
# online=True


# In[67]:


#UNCOMMENT THIS!
# #Save the dataframe with optimal dim
# good_dim = int(np.squeeze(np.where(hyperparams_to_test['n_components'] == kneedle.knee)))
# X_isomap = embeddings_results['n_components'][good_dim] #pick the best (knee point) n_components
# X_isomap_df = pd.DataFrame(X_isomap, columns = ['Mode {}'.format(i) for i in range(X_isomap.shape[1])])#, index = index_X)
# X_isomap_df.to_csv('/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/data/interim/Isomap_'+ str(X_isomap.shape[1])+ 'D_' + STATE + '.csv')
# X_isomap_df.shape


# In[295]:


#Save the dataframe with optimal dim
# good_dim = int(np.squeeze(np.where(hyperparams_to_test['n_components'] == kneedle.knee)))
X_isomap = embeddings_results['n_components'][0] #pick the best (knee point) n_components
X_isomap_df = pd.DataFrame(X_isomap, columns = ['Mode {}'.format(i) for i in range(X_isomap.shape[1])])#, index = index_X)
X_isomap_df.to_csv('/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/data/interim/Isomap_'+ str(X_isomap.shape[1])+ 'D_' + STATE + '.csv')
X_isomap_df.shape


# In[23]:


get_ipython().run_cell_magic('time', '', "#Let's try Spectral Emdedding, will use trustworhiness to see how much local structure is retained\n#The best parameters from the vizualization seem to be maybe 50 neighbors, let's try several n_components\nemb_method = SpectralEmbedding\nmax_components = 117\nhyperparams_to_test = {'n_components': np.arange(2, max_components, 2)}\n\nerrors_results_SE, embeddings_results_SE, trustws_results_SE = choose_dimension(X, emb_method, hyperparams_to_test, '', **{'n_neighbors': 50,\n 'n_components': 2,\n 'n_jobs': -1})\n\n#UNCOMMENT THE ABOVE!!!\n# hyperparams_to_test = {'n_components': [14]}\n# # colors = covid_clusters['x'].copy()\n\n# errors_results_SE, embeddings_results_SE, trustws_results_SE = choose_dimension(X, emb_method, hyperparams_to_test, '', **{'n_neighbors': 50,\n#  'n_components': 2,\n#  'n_jobs': -1})")


# In[24]:


# max_components = 30
kneedle = KneeLocator(np.arange(2, max_components, 2), np.array(trustws_results_SE['n_components']), S=1, curve='concave', direction='increasing', interp_method='polynomial')
kneedle.plot_knee()
plt.title(emb_method.__name__  + ' trustworthiness')
plt.xlabel('n_components')
plt.ylabel('trustworhiness')
kneedle.knee, kneedle.knee_y


# In[25]:


# #Save the dataframe with optimal dim
# good_dim_SE = int(np.squeeze(np.where(hyperparams_to_test['n_components'] == kneedle.knee)))
# X_SE = embeddings_results['n_components'][good_dim_SE] #pick the best (knee point) n_components
# X_SE_df = pd.DataFrame(X_SE, columns = ['Mode {}'.format(i) for i in range(X_SE.shape[1])])#, index = index_X)
# X_SE_df.to_csv('/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/data/interim/SE_' + str(X_SE.shape[1]) + 'D_' + STATE + '.csv')
# X_SE_df.shape


# In[26]:


#Save the dataframe with optimal dim
good_dim_SE = int(np.squeeze(np.where(hyperparams_to_test['n_components'] == kneedle.knee)))
X_SE = embeddings_results_SE['n_components'][good_dim_SE] #pick the best (knee point) n_components
X_SE_df = pd.DataFrame(X_SE, columns = ['Mode {}'.format(i) for i in range(X_SE.shape[1])])#, index = index_X)
X_SE_df.to_csv('/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/data/interim/SE_' + str(X_SE.shape[1]) + 'D_' + STATE + '.csv')
X_SE_df.shape


# In[27]:


get_ipython().run_cell_magic('time', '', "#LLE\nemb_method = LocallyLinearEmbedding\nhyperparams_to_test = {'n_components': np.arange(2, 117, 2)}\n\n\nerrors_results_LLE, embeddings_results_LLE, trustws_results_LLE  = choose_dimension(X, emb_method, hyperparams_to_test, '',**{'n_neighbors': 200,\n 'n_components': 2,\n 'reg': 0.1, #Note here that we later ended up using reg 0.1, so setting it\n 'eigen_solver': 'auto',\n 'tol': 1e-06,\n 'max_iter': 1000,\n 'method': 'standard',\n 'hessian_tol': 0.0001,\n 'modified_tol': 1e-12,\n 'neighbors_algorithm': 'auto',\n 'random_state': None,\n 'n_jobs': -1})\n\n# #Uncomment the above!!!\n# hyperparams_to_test = {'n_components': [14]}\n# # colors = covid_clusters['x'].copy()\n\n# errors_results_LLE, embeddings_results_LLE, trustws_results_LLE  = choose_dimension(X, emb_method, hyperparams_to_test, '',**{'n_neighbors': 200,\n#  'n_components': 2,\n#  'reg': 0.1, #Note here that we later ended up using reg 0.1, so setting it\n#  'eigen_solver': 'auto',\n#  'tol': 1e-06,\n#  'max_iter': 1000,\n#  'method': 'standard',\n#  'hessian_tol': 0.0001,\n#  'modified_tol': 1e-12,\n#  'neighbors_algorithm': 'auto',\n#  'random_state': None,\n#  'n_jobs': -1})")


# In[28]:


kneedle = KneeLocator(np.arange(2, 117, 2), np.array(trustws_results_LLE['n_components']), S=1, curve='concave', direction='increasing', interp_method='polynomial')
kneedle.plot_knee()
plt.title(emb_method.__name__  + ' trustworthiness')
plt.xlabel('n_components')
plt.ylabel('trustworhiness')
kneedle.knee, kneedle.knee_y


# In[29]:


# #Save the dataframe with optimal dim
# good_dim_LLE = int(np.squeeze(np.where(hyperparams_to_test['n_components'] == kneedle.knee)))
# X_LLE = embeddings_results['n_components'][good_dim_LLE] #pick the best (knee point) n_components
# X_LLE_df = pd.DataFrame(X_LLE, columns = ['Mode {}'.format(i) for i in range(X_LLE.shape[1])])#, index = index_X)
# X_LLE_df.to_csv('/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/data/interim/LLE_' + str(X_LLE.shape[1]) + 'D_' + STATE + '.csv')
# X_LLE_df.shape


# In[30]:


good_dim_LLE = int(np.squeeze(np.where(hyperparams_to_test['n_components'] == kneedle.knee)))
X_LLE = embeddings_results_LLE['n_components'][good_dim_LLE] #pick the best (knee point) n_components
X_LLE_df = pd.DataFrame(X_LLE, columns = ['Mode {}'.format(i) for i in range(X_LLE.shape[1])])#, index = index_X)
X_LLE_df.to_csv('/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/data/interim/LLE_' + str(X_LLE.shape[1]) + 'D_' + STATE + '.csv')
X_LLE_df.shape


# ## Visualize SE, Isomap, LLE

# In[10]:


#Viz clustering on the emb
# filename_SE_clustering = FIGURES_ADDITIONAL_PATH + 'clustering'
# %%matplotlib notebook
_ = viz_SE(X, 'red', filename = None, alpha = 0.5)


# In[11]:


_ = viz_Isomap(X, 'red', filename = None, alpha = 0.5)


# In[12]:


_ = viz_LLE(X, 'red', filename = None, alpha = 0.5)


# ## GMM clustering in R

# In[13]:


# ro.r('install.packages("mclust")')
base = importr('base')
mclust = importr('mclust')
ro.r('set.seed(0)')


# In[14]:


with localconverter(ro.default_converter + pandas2ri.converter):
#     BIC = mclust.mclustBIC(X_SE_df, G = pd.Series(np.arange(1,12)))
    mc = mclust.Mclust(X_SE_df, G = pd.Series(np.arange(1,6)))
    print(base.summary(mc))
    print(np.quantile(mc[15], [0, 0.25, 0.5, 0.75, 1]))
    SE_clusters = np.array(mc[14])
    SE_means = np.array(mc[12][1])


# In[312]:


with localconverter(ro.default_converter + pandas2ri.converter):
#     BIC = mclust.mclustBIC(X_SE_df, G = pd.Series(np.arange(1,12)))
    mc = mclust.Mclust(X_isomap_df, G = pd.Series(np.arange(1,6)))
    print(base.summary(mc))
    print(np.quantile(mc[15], [0, 0.25, 0.5, 0.75, 1]))
    isomap_clusters = np.array(mc[14])
    isomap_means = np.array(mc[12][1])


# In[313]:


with localconverter(ro.default_converter + pandas2ri.converter):
#     BIC = mclust.mclustBIC(X_SE_df, G = pd.Series(np.arange(1,12)))
    mc = mclust.Mclust(X_LLE_df, G = pd.Series(np.arange(1,6)))
    print(base.summary(mc))
    print(np.quantile(mc[15], [0, 0.25, 0.5, 0.75, 1]))
    LLE_clusters = np.array(mc[14])
    LLE_means = np.array(mc[12][1])


# ### Interpretability: Centroids, Map, Embedding

# In[314]:


#Start with LLE
LLE_means = pd.DataFrame(LLE_means, columns = ['V'+str(i+1) for i in range(LLE_means.shape[1])])
LLE_means


# In[315]:


N_TIMESERIES = 5
FIGURES_ADDITIONAL_PATH = 'exploratory/covid/interpretability/TX/'
closest_to_mean_samples, closest_to_mean_block_ids = find_closest_time_series(X_LLE_df, LLE_means, covid_ga_, n = N_TIMESERIES)
closest_to_mean_samples['V4'].shape


# In[316]:


#Look at the closest time series
for cluster in LLE_means.columns:   
    filename_ts_plot = FIGURES_ADDITIONAL_PATH + 'LLE_closest_time_series_'+cluster+'.png'
    plot_time_series(closest_to_mean_samples[cluster], 'Cluster_'+cluster, filename = filename_ts_plot)


# In[317]:


#Viz on the emb
#CHECK DISCREPANCY BETWEEN X AND X_DF_LLE
filename_LLE_clustering = FIGURES_ADDITIONAL_PATH + 'LLE_clustering'
_ = viz_LLE(X, LLE_clusters, filename = filename_LLE_clustering, alpha = 0.5)


# In[320]:


# #Viz clustering on the map
filename_LLE_clustering_map = FIGURES_ADDITIONAL_PATH + 'LLE_clustering_map'
viz_cluster_map(LLE_clusters, index_X, filename = filename_LLE_clustering_map)


# In[322]:


#Now Isomap
isomap_means = pd.DataFrame(isomap_means, columns = ['V'+str(i+1) for i in range(isomap_means.shape[1])])
# isomap_means


# In[323]:


N_TIMESERIES = 5
FIGURES_ADDITIONAL_PATH = 'exploratory/covid/interpretability/TX/'
closest_to_mean_samples, closest_to_mean_block_ids = find_closest_time_series(X_isomap_df, isomap_means, covid_ga_, n = N_TIMESERIES)
closest_to_mean_samples['V4'].shape


# In[324]:


#Look at the closest time series
for cluster in isomap_means.columns:   
    filename_ts_plot = FIGURES_ADDITIONAL_PATH + 'isomap_closest_time_series_'+cluster+'.png'
    plot_time_series(closest_to_mean_samples[cluster], 'Cluster_'+cluster, filename = filename_ts_plot)


# In[330]:


#Viz on the emb
#CHECK DISCREPANCY BETWEEN X AND X_DF_LLE
filename_isomap_clustering = FIGURES_ADDITIONAL_PATH + 'isomap_clustering'
_ = viz_Isomap(X, isomap_clusters, filename = filename_isomap_clustering, alpha = 0.5)


# In[326]:


# #Viz clustering on the map
filename_isomap_clustering_map = FIGURES_ADDITIONAL_PATH + 'isomap_clustering_map'
viz_cluster_map(isomap_clusters, index_X, filename = filename_isomap_clustering_map)


# In[328]:


# #Viz on the emb
# #CHECK DISCREPANCY BETWEEN X AND X_DF_LLE
# filename_isomap_clustering = FIGURES_ADDITIONAL_PATH + 'isomap_clustering_SE_viz'
# _ = viz_SE(X_isomap_df.values, isomap_clusters, filename = filename_isomap_clustering, alpha = 0.5)


# In[15]:


#last, SE

SE_means = pd.DataFrame(SE_means, columns = ['V'+str(i+1) for i in range(SE_means.shape[1])])
# SE_means


# In[16]:


N_TIMESERIES = 5
FIGURES_ADDITIONAL_PATH = 'exploratory/covid/interpretability/TX/'
closest_to_mean_samples, closest_to_mean_block_ids = find_closest_time_series(X_SE_df, SE_means, covid_ga_, n = N_TIMESERIES)
closest_to_mean_samples['V4'].shape


# In[17]:


#Look at the closest time series
for cluster in SE_means.columns:   
    filename_ts_plot = FIGURES_ADDITIONAL_PATH + 'SE_closest_time_series_'+cluster+'.png'
    plot_time_series(closest_to_mean_samples[cluster], 'Cluster_'+cluster, filename = filename_ts_plot)


# In[18]:


#Viz on the emb
#CHECK DISCREPANCY BETWEEN X AND X_DF_LLE
filename_SE_clustering = FIGURES_ADDITIONAL_PATH + 'SE_clustering'
_ = viz_SE(X, SE_clusters, filename = filename_SE_clustering, alpha = 0.5)


# In[19]:


# #Viz clustering on the map
filename_SE_clustering_map = FIGURES_ADDITIONAL_PATH + 'SE_clustering_map'
viz_cluster_map(SE_clusters, index_X, filename = filename_SE_clustering_map)


# In[336]:


ga_shape = gpd.read_file('/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/data/external/georgia_blockgroup_shape/tl_2017_13_bg.shp')


# In[337]:


ga_shape.head()


# In[339]:


# ga_shape.TRACTCE.value_counts()


# In[ ]:




