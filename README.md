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
To use the code from this repository, we need a Linux machine or Windows with <a href = "https://docs.microsoft.com/en-us/windows/wsl/install-win10">WSL</a>  or <a href = "https://cygwin.com/cygwin-ug-net/cygwin-ug-net.pdf">Cygwin</a> configured to run Linux commands. A <a href = "https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent">conda</a> installation of <a href = "https://www.python.org/downloads/">Python 3</a> and an installation of <a href = "https://www.r-project.org/">R</a> are also required.

To clone this repository:
    
    git clone https://github.com/InstituteforDiseaseModeling/covid-mobility-and-behavior.git
    
<a href = "https://www.gnu.org/software/make/">Make</a> utility automates downloading the data and creating the environment. Tip: typing `make` in the terminal will show description of available commands.

First, we need to create a virtual environment and install the requirements for our project (the below commands should be executed from the root directory of the project).
The following commands will create a conda virtual environment for the project, install required packages and create a jupyter kernel for the project.

    make create_environment
    make requirements
    
Now, we can download the data. This will download raw data and necessary shapefiles (the command could take up to 20 min to run depending on the Internet speed):

    make data
   
Optionally, we can download precomputed low-dimensional embeddings and other analysis files to allow for creating figures without waiting for the code to run.

    make precomputed_data
    
Finally, we should be able to run notebooks from the [/notebooks](/notebooks) folder. Dennis, please use the [/notebooks/Supplement_PCA_explained_variance.ipynb](PCA explained variance notebook) for now to test the code. If it works, that would mean that the local package was installed correctly and that the data was downloaded and saved correctly.
Other notebooks are in the process of being cleaned up and are coming very soon!


Project Organization (draft)
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` 
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources. (Not on Github, in IDM dropbox, downloaded automatically)
    │   ├── interim        <- Intermediate data that has been transformed.(Not on Github, in IDM dropbox, downloaded automatically)
    │   ├── processed      <- The final, canonical data sets for modeling.(Not on Github, in IDM dropbox, downloaded automatically)
    │   └── raw            <- The original, immutable data dump.(Not on Github, in IDM dropbox, downloaded automatically)
    │
    ├── docs               <- documentation, see html files in the build directory. Will be published using github pages.
    │
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download data
        │   └── make_dataset.py
        │ 
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
           └── visualize.py
    

--------

<p><small>Project template based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
