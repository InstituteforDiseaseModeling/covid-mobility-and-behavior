#Imports
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
import geopandas as gpd
import datetime
import scipy
from src.config import configurations
from src.dimensionality_reduction import compute_covariance, visualize_manifold_method, choose_dimension
from src.dimensionality_reduction import viz_cluster_map, viz_LLE, viz_Isomap, viz_SE
from src.utils import define_paths, load_data, convert_to_python_dict, save_obj, load_obj, find_closest_time_series
from src.utils import plot_time_series, find_cos_similarity, intersection, relabel_clusters
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.conversion import localconverter
from sklearn.manifold import Isomap, SpectralEmbedding, LocallyLinearEmbedding, trustworthiness
from sklearn.manifold import trustworthiness
from copy import copy 
from kneed import DataGenerator, KneeLocator
import pickle
from matplotlib import gridspec
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import scipy
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

plt.rcParams['figure.dpi'] = 600

def GMM_clustering_R(X_SE_df, method, default_cluster_num = None):
    """Function to check BIC and perform GMM clustering on embedded dataset"""
    #First, import r packages and fix random seed:
    base = importr('base')
    mclust = importr('mclust')
    ro.r('set.seed(0)')
    
    #Now, check BIC and make a plot
    num_components_to_try = pd.Series(np.arange(1,12)) #try up to 12 components
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.r('set.seed(0)')
        BIC_SE = mclust.mclustBIC(X_SE_df, G = num_components_to_try)
    
    model_names = ['EII', 'VII', 'EEI', 'VEI', 'EVI', 'VVI', 'EEE', 'EVE', 'VEE', 'VVE', 'EEV', 'VEV', 'EVV', 'VVV']
    sns.set(style="darkgrid")
#     sns.set_palette("tab10")
    BIC_SE_df = pd.DataFrame(BIC_SE, columns = model_names)
#     plt.figure()
    BIC_SE_df.plot(marker = 'o')
    plt.title('GMM BIC on ' + method.__name__)
    
    #Now, find the knee point of the optimal BIC plot (the best GMM parametrization)
    best_parametrization = BIC_SE_df.columns[BIC_SE_df.max().argmax()]
    kneedle = KneeLocator(num_components_to_try, BIC_SE_df[best_parametrization], S=1, curve='concave', direction='increasing', interp_method='interp1d')
#     plt.figure()
    kneedle.plot_knee()
    plt.title('GMM BIC on ' + method.__name__ + ': Knee Point')
    plt.xlabel('num_GMM_components')
    plt.ylabel('')
    print('Elbow point: {} components with BIC {}'.format(kneedle.knee, kneedle.knee_y))
    
    #Pick the best number of GMM components:
    best_num_components = kneedle.knee-1
    if default_cluster_num is not None:
        best_num_components = default_cluster_num-1
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.r('set.seed(0)')
        mc = mclust.Mclust(X_SE_df, G = pd.Series([num_components_to_try[best_num_components]]))
        print(base.summary(mc))
        print('Uncertainty quantiles:', np.quantile(mc[15], [0, 0.25, 0.5, 0.75, 1]))
        mc_dict = convert_to_python_dict(mc)
        SE_model_name = mc_dict['modelName']
        print(SE_model_name)
        param = mc_dict['parameters']
        SE_means = np.array(convert_to_python_dict(param)['mean'])
        SE_uncertainty = np.array(mc_dict['uncertainty'])
        SE_z = np.array(convert_to_python_dict(mc)['z'])
        SE_clusters = np.array(convert_to_python_dict(mc)['classification'])
        SE_means = pd.DataFrame(SE_means, columns = ['V'+str(i+1) for i in range(SE_means.shape[1])])
#         np.array(mc[14])
#         SE_means = np.array(mc[12][1])
    return SE_clusters, SE_means, SE_z, SE_uncertainty

def create_avg_df(SE_clusters, index_X, covid_):
    SE_clusters_block_indexed = pd.Series(SE_clusters, index = index_X)
    df_CI = covid_.dropna()#[SE_clusters_block_indexed['x']==4]
    df_CI['cluster'] = SE_clusters_block_indexed.astype('int')
    df_CI['block'] = df_CI.index
    dates = df_CI.columns.drop(['block', 'cluster'])
    avg_per_clust = df_CI.groupby('cluster').mean()[dates]#[dates].plot()
    return avg_per_clust

def index_with_blocks_and_save(STATE, emb_df_optimal_D, emb_2D, emb_3D, clusters, z, uncertainty, index_X, emb_method):
    df_optimal_D = emb_df_optimal_D.set_index(index_X)
    df_2D = pd.DataFrame(emb_2D, columns = ['Mode ' + str(i) for i in range(emb_2D.shape[1])], index = index_X)
    df_3D = pd.DataFrame(emb_3D, columns = ['Mode ' + str(i) for i in range(emb_3D.shape[1])], index = index_X)
    df_clusters = pd.Series(clusters, index = index_X)
    df_z = z.set_index(index_X)
    df_uncertainty = pd.Series(uncertainty, index = index_X)
    
    #Save
    path_to_processed = os.path.join(configurations['DATA_PATH'], 'processed')
    df_optimal_D.to_csv(os.path.join(path_to_processed, emb_method.__name__ + str(df_optimal_D.shape[1]) + 'D_' + STATE + '.csv'))
    df_2D.to_csv(os.path.join(path_to_processed, emb_method.__name__ + str(df_2D.shape[1]) + 'D_' + STATE + '.csv'))
    df_3D.to_csv(os.path.join(path_to_processed, emb_method.__name__ + str(df_3D.shape[1]) + 'D_' + STATE + '.csv'))
    df_clusters.to_csv(os.path.join(path_to_processed, 'labels_' + STATE + '.csv'))
    df_z.to_csv(os.path.join(path_to_processed, 'z_' + STATE + '.csv'))
    df_uncertainty.to_csv(os.path.join(path_to_processed, 'uncertainty_' + STATE + '.csv'))
    return df_optimal_D, df_2D, df_3D, df_clusters, df_z, df_uncertainty

def add_state_to_fig(state, fig, spec, row, NUM_STATES, X, reordered_SE_clusters, index_X, reordered_avg_per_clust, load_path = None, save_path = None, separate = False, two_cols = False, configurations = None):
    SHAPE_PATH, FIGURE_PATH, RAW_DATA_PATH, INCOME_POPULATION_PATH = define_paths(state)
    almost_sequential_pal = configurations['clustering_palette']#['#1f78b4','#a6cee3','#fdbf6f','#ff7f00', '#cc78bc']
    sns.set_palette(almost_sequential_pal)
    NUM_COLS = 3
    if two_cols:
        NUM_COLS = 2
    SEPARATE_PATH = os.path.abspath(os.path.join(os.getcwd(), '..', 'reports', 'figures', 'Separate_Summary'))
    # First column
    if separate:
        fig = plt.figure(figsize = (5*7/3, 5*(6.25 - 0.77)/4))
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(spec[NUM_COLS*row], projection = '3d')#NUM_STATES, NUM_COLS, NUM_COLS*row + 1, projection = '3d')
    ax.set_anchor('E')
    if two_cols:
        cbar = None
    else:
        cbar = 'clustering'
    _, X_3D_SE = viz_SE(X, reordered_SE_clusters, filename = None, alpha = 0.5, cbar = cbar, ax = ax, load_path = load_path, save_path = save_path)
    if state == 'wa':
        ax.view_init(30, 200)
    if state == 'ga':
        ax.view_init(30, 80)
    if state == 'tx':
        ax.view_init(30, 245)
    if state == 'ca':
        ax.view_init(30, 80)
    
    ax.set_xlim(np.array([np.min(X_3D_SE[:,0]), np.max(X_3D_SE[:,0])]))
    ax.set_ylim(np.array([np.min(X_3D_SE[:,1]), np.max(X_3D_SE[:,1])]))
    ax.set_zlim(np.array([np.min(X_3D_SE[:,2]), np.max(X_3D_SE[:,2])]))
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    plt.axis('tight')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])#,rotation=-15, va='center', ha='right')
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])
    ax.xaxis._axinfo['juggled'] = (0,0,1)
#     ax.dist = 5.5
    # ax.yaxis._axinfo['juggled'] = (1,1,1)
    # ax.zaxis._axinfo['juggled'] = (1,0,0)
    ax.grid(False)
    ax.set_title('')
#     ax.set_ylabel("Median Population", fontname="Arial", fontsize=12)

    if separate:
        fig.savefig(os.path.join(FIGURE_PATH, state + '_embedding_col_1.png'), bbox_inches = 'tight',  pad_inches=0, dpi = 900)
        
    
    # Second column
    if separate:
        fig = plt.figure(figsize = (5*7/3, 5*(6.25 - 0.77)/4))
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(spec[NUM_COLS*row + 1])#NUM_STATES, NUM_COLS, NUM_COLS*row + 2)
        ax.set_aspect('equal')
#     ax.set_anchor('C')
    if state == 'wa':
        ax.margins(0.2, 0.2) 
    
    if two_cols:
        cmap = None
    else:
        cmap = 'clust'
        
    viz_cluster_map(reordered_SE_clusters, index_X, filename = None, cmap = cmap, ax = ax, title = state, state = state)
    ax.set_title('')
    plt.axis('tight')
    
    if separate:
        fig.savefig(os.path.join(FIGURE_PATH, state + '_map_col_2.png'), bbox_inches = 'tight',  pad_inches=0, dpi = 900)
    
    
    #Third column
    if not two_cols:
        if separate:
            fig = plt.figure(figsize = (5*7/3, 5*(6.25 - 0.77)/4))
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(spec[NUM_COLS*row + 2])#NUM_STATES, NUM_COLS, NUM_COLS*row + 3)
        plot_df = reordered_avg_per_clust.T
        plot_df.index = [pd.to_datetime(date, format='%Y-%m-%d').date() for date in plot_df.index]
        
        almost_sequential_pal = ['#1f78b4','#a6cee3','#fdbf6f','#ff7f00', '#cc78bc']
        friendly_cmap = ListedColormap(sns.color_palette(almost_sequential_pal, len(np.unique(reordered_SE_clusters))).as_hex())

        plot_df.plot(ax = ax, ylim = [0.1, 0.7], legend = False, cmap = friendly_cmap) 
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=MO, interval=1))
        ax.tick_params(which='minor', length=0, color='r')
        ax.tick_params(axis = 'both', length=10, width=3, which='major')
        ax.axvline(datetime(2020, 6, 1), color = 'k', alpha = .3, ls='--', lw =.5)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
#         ax.xticks(fontsize=11*5)
#         ax.yticks(fontsize=11*5)
        ax.tick_params(axis='both', which='major', labelsize=6*5)
        ax.grid(True, 'minor', 'x', ls='--', lw=.5, c='k', alpha=.3)
        ax.set_xlim([datetime(2020, 2, 19), datetime(2020, 6, 21)])    
        if separate:
            fig.savefig(os.path.join(FIGURE_PATH, state + '_time_series_col_3.png'), bbox_inches = 'tight',  pad_inches=0, dpi = 900)

def analysis(STATE, method, method_kwargs, hyperparams_to_test, fig, spec, row, precomputed = False, separate = False, two_cols = False, NUM_STATES = 1, configurations = None):
    #First, define appropriate paths
    SHAPE_PATH, FIGURE_PATH, RAW_DATA_PATH, INCOME_POPULATION_PATH = define_paths(STATE)
    
    #Load the data
    covid_, X, index_X, columns_X = load_data(RAW_DATA_PATH)
    
    #Do dim red
    print('##################D-RED#################')
    emb_method = method
    if not precomputed:
        errors_results, embeddings_results, trustws_results = choose_dimension(X, emb_method, hyperparams_to_test, **method_kwargs)

        save_obj(embeddings_results, STATE + '_embeddings_results' + method.__name__)
        save_obj(errors_results, STATE + '_errors_results' + method.__name__)
        save_obj(trustws_results, STATE + '_trustws_result' + method.__name__)
    if precomputed:
        embeddings_results = load_obj(STATE + '_embeddings_results' + method.__name__)
        errors_results = load_obj(STATE + '_errors_results' + method.__name__)
        trustws_results = load_obj(STATE + '_trustws_result' + method.__name__)
    
    if (len(hyperparams_to_test['n_components']) > 1) and (errors_results['n_components'][0] is not None):
        plt.plot(hyperparams_to_test['n_components'], errors_results['n_components'])
    
    if (len(hyperparams_to_test['n_components']) > 1):
        kneedle = KneeLocator(hyperparams_to_test['n_components'], np.array(trustws_results['n_components']), S=1, curve='concave', direction='increasing', interp_method='polynomial', online = False)
        kneedle.plot_knee()
        plt.title(emb_method.__name__  + ' trustworthiness')
        plt.xlabel('n_components')
        plt.ylabel('trustworhiness')
        kneedle.knee, kneedle.knee_y

    #Save the dataframe with optimal dim
    if (len(hyperparams_to_test['n_components']) > 1):
        good_dim = int(np.squeeze(np.where(hyperparams_to_test['n_components'] == kneedle.knee)))
    else:
        good_dim = 0
    X_method = embeddings_results['n_components'][good_dim] #pick the best (knee point) n_components
    X_method_df = pd.DataFrame(X_method, columns = ['Mode {}'.format(i) for i in range(X_method.shape[1])])#, index = index_X)
    X_method_df.to_csv(os.path.join(configurations['DATA_PATH'], 'interim', method.__name__ + str(X_method.shape[1]) + 'D_' + STATE + '.csv'))
    print('Saving optimal embedding. Method: ', method.__name__, 'shape: ', X_method_df.shape)
    
    print('##################INITIAL VIZ#################')
    #Find the 2D and 3D embeddings and continuous colors based on that
    filename_initial = os.path.join(FIGURE_PATH, 'initial_' + method.__name__)
    if method.__name__ == 'Isomap':
        viz = viz_Isomap
    if method.__name__ == 'SpectralEmbedding':
        viz = viz_SE
    if method.__name__ == 'LocallyLinearEmbedding':
        viz = viz_LLE
        
    if precomputed:
        load_path = os.path.join('obj', STATE)
        save_path = None
    else:
        load_path = None
        save_path = os.path.join('obj', STATE)
    X_2D_emb, X_3D_emb = viz(X, colors = None, filename = filename_initial, alpha = 0.5, load_path = load_path, save_path = save_path)
    cos_colors = find_cos_similarity(X_2D_emb)
    #Color the manifold continuously 
    filename_initial_colored = os.path.join(FIGURE_PATH, 'initial_'+ method.__name__ + '_colored')
    X_2D_emb, X_3D_emb = viz(X, colors = cos_colors, filename = filename_initial_colored, cbar = None, alpha = 0.5, load_path = load_path, save_path = save_path)
    
    print('##################GMM CLUSTERING#################')
    #Import R for clustering
    base = importr('base')
    mclust = importr('mclust')
    ro.r('set.seed(1)')
    
    dontprecomputeclusters = True
#     if not precomputed:
    if dontprecomputeclusters:
        clusters, means, z, uncertainty = GMM_clustering_R(X_method_df, method, default_cluster_num=5) #could change this to 5 to be consistent across states to auto-id clust #
        clusters_block_indexed = pd.Series(clusters, index = index_X)

        avg_per_clust = create_avg_df(clusters, index_X, covid_)

        reordered_clusters, reordered_means, reordered_z, reordered_uncertainty = relabel_clusters(clusters.astype('int'), avg_per_clust, means, z, uncertainty)
        reordered_avg_per_clust = create_avg_df(reordered_clusters, index_X, covid_)
        #Save
        np.save(os.path.join('obj', STATE + '_reordered_clusters.npy'), reordered_clusters,)
        reordered_means.to_csv(os.path.join('obj', STATE + '_reordered_means.csv'))
        reordered_z.to_csv(os.path.join('obj', STATE + '_reordered_z.csv'))
        np.save(os.path.join('obj', STATE + '_reordered_uncertainty.npy'),reordered_uncertainty)
        
        reordered_avg_per_clust.to_csv(os.path.join('obj', STATE + '_reordered_avg_per_clust.csv'))
    
#     if precomputed:
    if not dontprecomputeclusters:
        reordered_clusters = np.load(os.path.join('obj', STATE + '_reordered_clusters.npy'))
        reordered_means = pd.read_csv(os.path.join('obj', STATE + '_reordered_means.csv'), index_col = 0)
        reordered_z = pd.read_csv(os.path.join('obj', STATE + '_reordered_z.csv'), index_col = 0)
        reordered_uncertainty = np.load(os.path.join('obj', STATE + '_reordered_uncertainty.npy'))
        reordered_avg_per_clust = pd.read_csv(os.path.join('obj', STATE + '_reordered_avg_per_clust.csv'), index_col = 0)        
    
    #Save the data for Dennis (for only this method)
    index_with_blocks_and_save(STATE, X_method_df, X_2D_emb, X_3D_emb, reordered_clusters, reordered_z, reordered_uncertainty, index_X, emb_method)
    
    N_TIMESERIES = 5
    closest_to_mean_samples, closest_to_mean_block_ids = find_closest_time_series(X_method_df, reordered_means, covid_, index_X, n = N_TIMESERIES)
    
    print('##################FINAL VIZ#################')
    sns.set(style="whitegrid")
    if two_cols:
        reordered_clusters = cos_colors #Change colors
    add_state_to_fig(STATE, fig, spec, row, NUM_STATES, X, reordered_clusters, index_X, reordered_avg_per_clust, load_path = load_path, save_path = save_path, separate = separate, two_cols = two_cols, configurations = configurations)
    
if __name__ == "__main__":
    pass