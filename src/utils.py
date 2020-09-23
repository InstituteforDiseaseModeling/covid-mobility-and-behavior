"""Module with utility functions
"""
from src.config import configurations
import os

def plot_time_series(time_series_df, name = 'time_series_of_interest', filename = None, only_melt = False):
    """TODO"""
#     FIGURES_PATH = '/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/reports/figures/'
    time_series_df_ = time_series_df.copy().T
    time_series_df_['dates'] = pd.to_datetime(time_series_df.columns)
    #closest_to_mean_sample_ = closest_to_mean_sample.T
    #closest_to_mean_sample_['dates'] = pd.to_datetime(columns_X)
#     closest_to_mean_sample_
    #Cluster means figure
    sns.set(style="darkgrid")
#     sns.set_palette("tab10")
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
        g.fig.savefig(filename + '.png', bbox_inches = 'tight')
    
    return melted_df

def find_closest_time_series(embedding, means, covid_block_indexed, index_X, n = 5):
    """Finds n closest time series to the cluster means (the vectors in means), 
    usually in the lower-dimensional space
    returns dict with keys V1...V6 cluster names and values dataframes with the n closest samples
    same for block_ids

    Parameters
    ----------
    embedding : np.array
        low-dimensional embedding
    means : str
        means of the GMM clusters in the low-dimensional space
    covid_block_indexed : pd.DataFrame
        stay-at-home time series
    index_X : pandas index
        index corresponding to rows of embedding
    n : int
        number of closest time series to return 
                
    Returns
    -------
    closest_to_mean_samples : dict
        dict with keys cluster names (from means.columns) and values dataframes with the n closest samples
    closest_to_mean_block_ids : dict
        dict with keys cluster names (from means.columns) and values ids of the correponding CBGs
    """
    closest_to_mean_samples = {}
    closest_to_mean_block_ids = {}
    for cluster in means.columns:
        sorted_indices = np.squeeze(np.argsort(scipy.spatial.distance.cdist(embedding, means[cluster][None,:], metric='euclidean'), axis = 0))
        closest_to_mean_samples[cluster] = covid_block_indexed.loc[index_X[sorted_indices[:n]]] #Take the closest n points
        closest_to_mean_block_ids[cluster] = index_X[sorted_indices[:n]]
    return closest_to_mean_samples, closest_to_mean_block_ids

def save_obj(obj, name):
    """
    Pickle objects, should be used from /notebooks folder

    Parameters
    ----------
    obj : Python object
        object to pickle
    name : str
        name of the object
        
    Returns
    -------
    """
    with open(os.path.join('obj', name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    Load pickled objects, should be used from /notebooks folder

    Parameters
    ----------
    name : str
        name of the object
        
    Returns
    -------
    Python object
        Loaded object
    """
    with open(os.path.join('obj', name + '.pkl'), 'rb') as f:
        return pickle.load(f)

def convert_to_python_dict(mc):
    """
    Convert R dataframe to Python dict

    Parameters
    ----------
    mc : rpy2 object
        R dataframe
        
    Returns
    -------
    dict
        Python dict with the information from R dataframe
    """
    return dict(zip(mc.names, list(mc)))

def define_paths(STATE):
    """
    Returns paths for raw data, shape files, figures and income-population files
    
    Parameters
    ----------
    STATE : str
        State to get the paths for

    Returns
    -------
    SHAPE_PATH : str
        shape files path
    FIGURE_PATH : str
        figures path
    RAW_DATA_PATH : str
        raw data path
    INCOME_POPULATION_PATH : str
        income-population files path
    """
    #Shape files:
    WASHINGTON_SHAPE_PATH = configurations['WASHINGTON_SHAPE_PATH']
    TEXAS_SHAPE_PATH = configurations['TEXAS_SHAPE_PATH']
    GEORGIA_SHAPE_PATH = configurations['GEORGIA_SHAPE_PATH']
    CALIFORNIA_SHAPE_PATH = configurations['CALIFORNIA_SHAPE_PATH']

    #Figures:
    WASHINGTON_FIGURE_PATH = configurations['WASHINGTON_FIGURE_PATH']
    TEXAS_FIGURE_PATH = configurations['TEXAS_FIGURE_PATH']
    GEORGIA_FIGURE_PATH = configurations['GEORGIA_FIGURE_PATH']
    CALIFORNIA_FIGURE_PATH = configurations['CALIFORNIA_FIGURE_PATH']
    FIGURE_PATH = configurations['FIGURE_PATH']

    #Data:
    DATA_PATH = configurations['DATA_PATH']
    WASHINGTON_RAW_DATA_PATH = configurations['WASHINGTON_RAW_DATA_PATH']
    GEORGIA_RAW_DATA_PATH = configurations['GEORGIA_RAW_DATA_PATH']
    TEXAS_RAW_DATA_PATH = configurations['TEXAS_RAW_DATA_PATH']
    CALIFORNIA_RAW_DATA_PATH = configurations['CALIFORNIA_RAW_DATA_PATH']
    
    #Income-population path:
    INCOME_POPULATION_PATH = os.path.join(DATA_PATH, 'external', 'income_population')
    
    if STATE == 'wa':
        SHAPE_PATH = WASHINGTON_SHAPE_PATH
        FIGURE_PATH = WASHINGTON_FIGURE_PATH
        RAW_DATA_PATH = WASHINGTON_RAW_DATA_PATH
    if STATE == 'tx':
        SHAPE_PATH = TEXAS_SHAPE_PATH
        FIGURE_PATH = TEXAS_FIGURE_PATH
        RAW_DATA_PATH = TEXAS_RAW_DATA_PATH
#         DATA_PATH  = 
    if STATE == 'ga':
        SHAPE_PATH = GEORGIA_SHAPE_PATH
        FIGURE_PATH = GEORGIA_FIGURE_PATH
        RAW_DATA_PATH = GEORGIA_RAW_DATA_PATH
    if STATE == 'ca':
        SHAPE_PATH = CALIFORNIA_SHAPE_PATH
        FIGURE_PATH = CALIFORNIA_FIGURE_PATH
        RAW_DATA_PATH = CALIFORNIA_RAW_DATA_PATH
    return SHAPE_PATH, FIGURE_PATH, RAW_DATA_PATH, INCOME_POPULATION_PATH

#Load the data
def load_data(RAW_DATA_PATH):
    """
    Loads raw data

    Parameters
    ----------
    RAW_DATA_PATH : str

    Returns
    -------
    covid_block_indexed : pd.DataFrame
        Dataframe with stay-at-home time series data, indexed by CBGs
    X : np.array
        Numpy array with stay-at-home time series data, dropped nans
    index_X : pandas index
        CBG indices correponding to the rows of X
    columns_X : list 
        List of the column names of X
    """
    covid = pd.read_csv(RAW_DATA_PATH)
    covid_block_indexed = covid.set_index('Unnamed: 0')
    covid_block_indexed = covid_block_indexed[covid_block_indexed.columns[:117]] #to consider the same time period with our WA analysis
    #Prepare the data: drop nans
    X = covid_block_indexed.dropna().values
    index_X = covid_block_indexed.dropna().index
    columns_X = covid_block_indexed.dropna().columns
    print('Data loaded. Dropped NaNs shape:{}. Initial shape:{}'.format(X.shape, covid_block_indexed.shape))
    return covid_block_indexed, X, index_X, columns_X

if __name__ == "__main__":
    pass