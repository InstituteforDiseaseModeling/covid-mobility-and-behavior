COVID Mobility and Behavior
==============================

This is a code and analysis repository for the paper "Cell phone tracking data reveals heterogeneity in stay at
home behavior during the SARS-CoV-2 pandemic". We analyze Safegraph COVID stay-at-home behavior data using nonlinear 
dimensionality reduction techniques and clustering. We produce lower-dimensional embeddings that capture the stay-at-home
behavioral patterns that correlate with socioeconomic information (e.g. income). Using the lower-dimensional embeddings, 
we obtain geographically connected clusters which reveal differences in stay-at-home behavior between rural and urban areas. 
The results are consistent across multiple states.

Getting Started
---------------
To use the code from this repository, we need a Linux machine or Windows with <a href = "https://docs.microsoft.com/en-us/windows/wsl/install-win10">WSL</a>  or <a href = "https://cygwin.com/cygwin-ug-net/cygwin-ug-net.pdf">Cygwin</a> configured to run Linux commands. A <a href = "https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent">conda</a> installation of <a href = "https://www.python.org/downloads/">Python 3</a> and an installation of <a href = "https://www.r-project.org/">R</a> are also required. An installation of a <a href = "https://jupyter.org/install">Jupyter Notebook</a> is needed for the correct execution of all the make commands. 

To clone this repository:
    
    git clone https://github.com/InstituteforDiseaseModeling/covid-mobility-and-behavior.git
    
<a href = "https://www.gnu.org/software/make/">Make</a> utility automates downloading the data and creating the environment. Tip: typing `make` in the terminal will show description of available commands.

First, we need to create a virtual environment and install the requirements for our project (the below commands should be executed from the root directory of the project).
The following commands will create a conda virtual environment for the project and activate it:

    make create_environment
    source activate covid-mobility-and-behavior
    
After that, we install required packages and create a jupyter kernel for the project (make sure R is installed on the system):

    make requirements
    
Now, we can download the data. This will download raw data and necessary shapefiles (the command could take up to 20 min to run depending on the Internet speed):

    make data
   
Optionally, we can download precomputed low-dimensional embeddings and other analysis files to allow for creating figures without waiting for the code to run.

    make precomputed_data
    
Finally, we should be able to run notebooks from the [/notebooks](/notebooks) folder. Dennis, please use the <a href = "https://github.com/InstituteforDiseaseModeling/covid-mobility-and-behavior/blob/master/notebooks/Supplement_PCA_explained_variance.ipynb">PCA Explained Variance notebook</a> for now to test the code. If it works, that would mean that the local package was installed correctly and that the data was downloaded and saved correctly.
Other notebooks are in the process of being cleaned up and are coming very soon!


Project Organization (draft)
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data`, `make environment`, `make requirements` 
    ├── README.md   
    ├── requirements.txt   <- Requirements file for reproducing the Python analysis environment
    ├── setup.py           <- Installation script for the local package
    │
    ├── demo
    |   ├──obj             <- Directory to save computed objects
    |   |
    │   ├── Demo-Main-Analysis.ipynb    <- Main demo notebook with the dimesionality reduction + clustering pipeline applied to synthetic demo data
    │   └── make-synthetic-wa.R         <- Script to generate demo data: synthetic mobility dynamics in Washington state
    │
    ├── data
    │   ├── external       <- Data from external sources, e.g. shapefiles for plotting maps (from census.gov)  
    │   ├── interim        <- Intermediate data files
    │   ├── processed      <- Final data sets -- final clustering labels and final low-dimensional coordinates for every state
    │   └── raw            <- Raw data -- this is where <a href = "https://www.safegraph.com/covid-19-data-consortium">SafeGraph mobility data</a> should be placed 
    │
    ├── notebooks          <- Jupyter notebooks with the analysis code and the code to generate figures
    |   ├──obj             <- Directory to save computed objects
    |   |
    │   ├── Main-Analysis-Figure2.ipynb    <- Main notebook with the dimesionality reduction + clustering pipeline applied to all 4 states, produces Figure 2
    │   ├── Schematic-Figure1.ipynb        <- Generates panels for the pipeline description in Figure 1 of the paper
    │   ├── Zoomed-Maps-Figure3.ipynb      <- Generates zoomed-in maps for Figure 3 of the paper
    │   └── income-population-KS.ipynb     <- Analysis of income and population density in identified clusters, Kolmogorov-Smirnov test for response speed distributions
    │
    ├── censuscode         <- Source code for interpretation analysis
    │   ├── get-acs2018data.R    <- Script to download ACS data (requires inserting API key to access ACS data)  
    │   └── make-census-plots.R  <- Script to interpret the clusters by correlating them with socieconomic data, produces Figures 4,5, and 6 of the paper
    |
    ├── reports            <- Final figures
    │   └── figures        
    │
    └── src                <- Source code 
        |
        ├── data           <- Scripts to download data (only downloads demo data and publicly available data like shapefiles, the SafeGraph data access should be requested from SafeGraph)
        │ 
        ├── config.py                      <- Configurations defining data paths and color palettes
        ├── core_pipeline.py               <- Source code for applying the pipeline of nonlinear dimensionality reduction + GMM clustering
        ├── dimensionality_reduction.py    <- Functions for dimensionality reduction methods and their visualization
        └── utils.py                       <- Helper functions
        
        

--------

<p><small>Project template based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
