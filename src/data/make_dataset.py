import os
from dotenv import load_dotenv, find_dotenv
import urllib.request
from pathlib import Path
import zipfile



# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

#Get raw data download url from .env (should be defined there)
# raw_data_url = os.environ.get("RAW_DATA_URL")
# income_population_data_url = os.environ.get("INCOME_POPULATION_URL")
demo_data_url = 'https://www.dropbox.com/sh/w4tjp849lnchb9d/AACv0jyNFI2V3mq9kZDPVWePa?dl=1'
#Define project directory to easier define paths
# project_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
project_dir = Path(__file__).resolve().parents[2]
#Places urls (for cities):
wa_places = 'https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_53_place_500k.zip' #'https://www2.census.gov/geo/tiger/TIGER2019/PLACE/tl_2019_53_place.zip'
tx_places = 'https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_48_place_500k.zip'#'https://www2.census.gov/geo/tiger/TIGER2019/PLACE/tl_2019_48_place.zip'
ca_places = 'https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_06_place_500k.zip'#'https://www2.census.gov/geo/tiger/TIGER2019/PLACE/tl_2019_06_place.zip' 
ga_places = 'https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_13_place_500k.zip'#'https://www2.census.gov/geo/tiger/TIGER2019/PLACE/tl_2019_13_place.zip'
#Shape files:
wa_shape = 'https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_53_bg_500k.zip'#'https://www2.census.gov/geo/tiger/TIGER2019/BG/tl_2019_53_bg.zip'
tx_shape = 'https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_48_bg_500k.zip'#'https://www2.census.gov/geo/tiger/TIGER2019/BG/tl_2019_48_bg.zip'
ca_shape = 'https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_06_bg_500k.zip'#'https://www2.census.gov/geo/tiger/TIGER2019/BG/tl_2019_06_bg.zip'
ga_shape = 'https://www2.census.gov/geo/tiger/GENZ2019/shp/cb_2019_13_bg_500k.zip'#'https://www2.census.gov/geo/tiger/TIGER2019/BG/tl_2019_13_bg.zip'

def create_data_directories():
    """Creates data directories
    Creates project_dir/data
            project_dir/data/raw
            project_dir/data/interim
            project_dir/data/processed
            project_dir/data/external

    Parameters
    ----------

    Returns
    -------
    None
    """
    Path(os.path.join(project_dir, 'data')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, 'data', 'raw')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, 'data', 'interim')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, 'data', 'processed')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(project_dir, 'data', 'external')).mkdir(parents=True, exist_ok=True)
    return 


def download_and_unzip_data(data_url, data_path):
    """Download and unzip function

    Downloads data from the provided url, unzips it to data_path

    Parameters
    ----------
    data_url : str
        URL to download data
    data_path : str
        Path to save the unzipped data

    Returns
    -------
    None
    """
    data_zip_path = data_path + '.zip'
    # data_path = os.path.join(project_dir, 'data', 'raw')
    urllib.request.urlretrieve(data_url, data_zip_path)
    with zipfile.ZipFile(data_zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    os.remove(data_zip_path)

    return None
    
# def download_external_data():
#     external_data_zip_path = os.path.join(project_dir, 'data', 'external.zip')
#     external_data_path = os.path.join(project_dir, 'data', 'external')
#     urllib.request.urlretrieve(external_data_url, external_data_zip_path)
#     with zipfile.ZipFile(raw_data_zip_path, 'r') as zip_ref:
#         zip_ref.extractall(raw_data_path)



def main():
    """Main function for downloading the data

    Downloads raw data from the provided url, unzips it to project_dir/data/raw
    Downloads external data and unzips it to project_dir/data/external

    Parameters
    ----------

    Returns
    -------
    None
    """
    #Create directories
    print('Creating data directories')
    create_data_directories()
    #Download and save raw data
    #We cannot provide a url to SafeGraph data, the raw data has to be downloaded separately and stored in the respective directories, see README
    # print('Downloading and saving raw data')
    # raw_data_path = os.path.join(project_dir, 'data', 'raw')
    # download_and_unzip_data(raw_data_url, raw_data_path)

    #Download and save demo data
    #Do demonstrate analysis we provide demo data to run our analysis on
    print('Downloading and saving raw data')
    demo_data_path = os.path.join(project_dir, 'data', 'demo')
    download_and_unzip_data(demo_data_url, demo_data_path)

    #Download and save shapefiles:
    print('Downloading and saving shapefiles')
    external_data_path = os.path.join(project_dir, 'data', 'external')
    #WA
    wa_shape_path = os.path.join(external_data_path, 'wa_shape')
    download_and_unzip_data(wa_shape, wa_shape_path)
    #GA
    ga_shape_path = os.path.join(external_data_path, 'ga_shape')
    download_and_unzip_data(ga_shape, ga_shape_path)
    #TX
    tx_shape_path = os.path.join(external_data_path, 'tx_shape')
    download_and_unzip_data(tx_shape, tx_shape_path)
    #CA
    ca_shape_path = os.path.join(external_data_path, 'ca_shape')
    download_and_unzip_data(ca_shape, ca_shape_path)


    #Download and save places
    print('Downloading and saving places')
    #WA
    wa_places_path = os.path.join(external_data_path, 'wa_places')
    download_and_unzip_data(wa_places, wa_places_path)
    #GA
    ga_places_path = os.path.join(external_data_path, 'ga_places')
    download_and_unzip_data(ga_places, ga_places_path)
    #TX
    tx_places_path = os.path.join(external_data_path, 'tx_places')
    download_and_unzip_data(tx_places, tx_places_path)
    #CA
    ca_places_path = os.path.join(external_data_path, 'ca_places')
    download_and_unzip_data(ca_places, ca_places_path)

    #Download and save income_population census data
    #We cannot provide a url to census data, it has to be downloaded separately and stored in the respective directories, see README
    # print('Downloading and saving income-population')
    # income_population_path = os.path.join(external_data_path, 'income_population')
    # download_and_unzip_data(income_population_data_url, income_population_path)

    return



# #@click.command()
# #@click.argument('input_filepath', type=click.Path(exists=True))
# #@click.argument('output_filepath', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')


if __name__ == '__main__':
    main()
