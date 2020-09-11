import os
from src.data.make_dataset import download_and_unzip_data
from dotenv import load_dotenv, find_dotenv
import urllib.request
from pathlib import Path
import zipfile



# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

#Define project directory to easier define paths
# project_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
project_dir = Path(__file__).resolve().parents[2]
#Get processed data download url from .env (should be defined there)
processed_data_url = os.environ.get("PROCESSED_DATA_URL")


def main():
    """Main function for downloading the processed data

    Downloads processed data from the provided url, unzips it to project_dir/data/processed

    Parameters
    ----------

    Returns
    -------
    None
    """
    print('Project dir:', project_dir)
    #Download and save processed data
    print('Downloading and saving processed data')
    processed_data_path = os.path.join(project_dir, 'data', 'processed')
    print('Saved to', processed_data_path)
    download_and_unzip_data(processed_data_url, processed_data_path)


if __name__ == '__main__':
    main()
