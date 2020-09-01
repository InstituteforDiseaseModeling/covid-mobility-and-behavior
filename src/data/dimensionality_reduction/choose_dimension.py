import copy 
import seaborn as sns
from sklearn.manifold import trustworthiness

def return_metric(method, model):
    if method.__name__ == 'Isomap':
        return model.reconstruction_error()
    if method.__name__ == 'TSNE':
        return None # TODO
    return None

def choose_dimension(X, emb_method, hyperparams_to_test, name_prefix, **kwargs):
    """
    Vizualizes grid of 2D embeddings varying hyperparams_to_test, and 3D projection of one of them
    method: sklearn method
    hyperparams_to_test: dictionary with keys hyperparams and values ranges
    kwargs: all other hyperparams of the method
    X: data
    For 3D projection, initial parameters are used
    """
    IMG_PATH = "/home/rlevin/notebooks/notebooks/datadrivenmethodsforgemspoliocovid/reports/figures/exploratory/covid/dim_red_viz/"+name_prefix
    sns.set_style("whitegrid", {'axes.grid' : False})
    #TODO add scaling
    #Default params:
    default_args = copy.copy(kwargs)
    print('here')
    embeddings_results = {} #dict with resulting embeddings with keys -- varied hyperparams, values: different embs
    errors_results = {} #dict with lists of errors as values, keys -- varied hyperparams
    trustws_results = {} #dict with lists of trustworthiness as values, keys -- varied hyperparams
    for range_key in hyperparams_to_test.keys():
        #Set to default kwargs:
        kwargs = copy.copy(default_args)
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
            trustws.append(trustworthiness(X, X_embedded, n_neighbors=5, metric='euclidean'))#the idea behind this is to test how good local structure is preserved, methods had more neighbors
        embeddings_results[range_key] = X_embs
        errors_results[range_key] = errors
        trustws_results[range_key] = trustws
    return errors_results, embeddings_results, trustws_results

    if __name__ == "__main__":
        pass