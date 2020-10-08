import os
from pathlib import Path

project_dir = Path(__file__).resolve().parents[1]

#Shape files:
# WASHINGTON_SHAPE_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'external', 'wa_shape', 'tl_2019_53_bg.shp'))
# TEXAS_SHAPE_PATH = os.path.abspath(os.path.join(project_dir,'data','external', 'tx_shape','tl_2019_48_bg.shp'))
# GEORGIA_SHAPE_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'external', 'ga_shape','tl_2019_13_bg.shp'))
# CALIFORNIA_SHAPE_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'external', 'ca_shape','tl_2019_06_bg.shp'))
WASHINGTON_SHAPE_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'external', 'wa_shape', 'cb_2019_53_bg_500k.shp'))
TEXAS_SHAPE_PATH = os.path.abspath(os.path.join(project_dir,'data','external', 'tx_shape','cb_2019_48_bg_500k.shp'))
GEORGIA_SHAPE_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'external', 'ga_shape','cb_2019_13_bg_500k.shp'))
CALIFORNIA_SHAPE_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'external', 'ca_shape','cb_2019_06_bg_500k.shp'))


WASHINGTON_PLACES_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'external', 'wa_places', 'cb_2019_53_place_500k.shp'))
TEXAS_PLACES_PATH = os.path.abspath(os.path.join(project_dir,'data','external', 'tx_places','cb_2019_48_place_500k.shp'))
GEORGIA_PLACES_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'external', 'ga_places', 'cb_2019_13_place_500k.shp'))
CALIFORNIA_PLACES_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'external', 'ca_places','cb_2019_06_place_500k.shp'))

#Figures:
WASHINGTON_FIGURE_PATH = os.path.abspath(os.path.join(project_dir, 'reports', 'figures', 'WA'))
TEXAS_FIGURE_PATH = os.path.abspath(os.path.join(project_dir, 'reports', 'figures', 'TX'))
GEORGIA_FIGURE_PATH = os.path.abspath(os.path.join(project_dir, 'reports', 'figures', 'GA'))
CALIFORNIA_FIGURE_PATH = os.path.abspath(os.path.join(project_dir, 'reports', 'figures', 'CA'))
FIGURE_PATH = os.path.abspath(os.path.join(project_dir, 'reports', 'figures'))

#Data:
DATA_PATH = os.path.abspath(os.path.join(project_dir, 'data'))
WASHINGTON_RAW_DATA_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'raw', 'stayathome-washington.csv'))
GEORGIA_RAW_DATA_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'raw', 'stayathome-georgia.csv'))
TEXAS_RAW_DATA_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'raw', 'stayathome-texas.csv'))
CALIFORNIA_RAW_DATA_PATH = os.path.abspath(os.path.join(project_dir, 'data', 'raw', 'stayathome-california.csv'))

#Income-population path:
INCOME_POPULATION_PATH = os.path.join(DATA_PATH, 'external', 'income_population')

print(project_dir)

configurations = {
'clustering_palette': ['#1f78b4','#a6cee3','#fdbf6f','#ff7f00', '#cc78bc'],
'WASHINGTON_SHAPE_PATH': WASHINGTON_SHAPE_PATH,
'TEXAS_SHAPE_PATH': TEXAS_SHAPE_PATH,
'GEORGIA_SHAPE_PATH': GEORGIA_SHAPE_PATH,
'CALIFORNIA_SHAPE_PATH': CALIFORNIA_SHAPE_PATH,

'wa_PLACES_PATH': WASHINGTON_PLACES_PATH,
'tx_PLACES_PATH': TEXAS_PLACES_PATH,
'ga_PLACES_PATH': GEORGIA_PLACES_PATH,
'ca_PLACES_PATH': CALIFORNIA_PLACES_PATH,

'WASHINGTON_FIGURE_PATH': WASHINGTON_FIGURE_PATH,
'TEXAS_FIGURE_PATH': TEXAS_FIGURE_PATH,
'GEORGIA_FIGURE_PATH': GEORGIA_FIGURE_PATH,
'CALIFORNIA_FIGURE_PATH': CALIFORNIA_FIGURE_PATH,
'FIGURE_PATH': FIGURE_PATH,
'DATA_PATH': DATA_PATH,
'WASHINGTON_RAW_DATA_PATH': WASHINGTON_RAW_DATA_PATH,
'GEORGIA_RAW_DATA_PATH': GEORGIA_RAW_DATA_PATH,
'TEXAS_RAW_DATA_PATH': TEXAS_RAW_DATA_PATH,
'CALIFORNIA_RAW_DATA_PATH': CALIFORNIA_RAW_DATA_PATH,
'INCOME_POPULATION_PATH': INCOME_POPULATION_PATH,
'PROJECT_DIR' : project_dir
}


if __name__ == '__main__':
    pass
