COVID Mobility and Behavior
==============================

This is a code and analysis repository for the paper <a href = https://www.medrxiv.org/content/10.1101/2020.10.31.20223776v2.full>Cell phone mobility data and manifold learning: Insights into population behavior during the COVID-19 pandemic</a>. Cell-phone mobility data offers a modern measurement instrument to investigate human mobility and behavior at an unprecedented scale. We investigate aggregated and anonymized mobility data (<a href = "https://www.safegraph.com/covid-19-data-consortium">SafeGraph COVID mobility data</a>) which measures how populations at the census-block-group geographic scale stayed at home in California, Georgia, Texas, and Washington in the beginning of the COVID-19 pandemic. Using manifold learning techniques, we find patterns of mobility behavior that align with stay-at-home orders, correlate with socioeconomic factors, cluster geographically, reveal subpopulations that likely migrated outof urban areas, and, importantly, link to COVID-19 case counts. The analysis and approach provides policy makers a framework for interpreting mobility data and behavior to inform actions aimed at curbing the spread of COVID-19.

Getting Started: System Requirements
-------------------------------------
To use the code from this repository, we need a Linux machine or Windows with <a href = "https://docs.microsoft.com/en-us/windows/wsl/install-win10">WSL</a>  or <a href = "https://cygwin.com/cygwin-ug-net/cygwin-ug-net.pdf">Cygwin</a> configured to run Linux commands. A <a href = "https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html#install-linux-silent">conda</a> installation of <a href = "https://www.python.org/downloads/">Python 3</a> and an installation of <a href = "https://www.r-project.org/">R</a> are also required. An installation of a <a href = "https://jupyter.org/install">Jupyter Notebook</a> is needed for the correct execution of the make commands (see below). The Python dependencies are specified in [requirements.txt](requirements.txt). The code was developed and tested on Ubuntu 18.04 computer with 16 GB RAM and Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz processor.

Installation Instructions
---------------------------
First, we clone the repository:
    
    git clone https://github.com/InstituteforDiseaseModeling/covid-mobility-and-behavior.git
    
<a href = "https://www.gnu.org/software/make/">Make</a> utility automates downloading the data and creating the environment. Typing `make` in the terminal will show description of available commands.

First, we need to create a virtual environment and install the requirements for our project (the below commands should be executed from the root directory of the project).
The following commands will create a conda virtual environment for the project and activate it:

    make create_environment
    source activate covid-mobility-and-behavior
    
After that, we install required packages and create a jupyter kernel for the project (make sure R is installed on the system):

    make requirements
    
Note that the above command installs a new jupyter kernel for the created virtual environment. This could be avoided by commenting out the respective lines in [Makefile](Makefile).

Now, we can download the data. The following will download necessary external data, e.g. shapefiles (the command could take up to 20 min to run depending on the Internet speed):

    make data
   
Now we should be able to run the demo notebook from the [/demo](/demo) folder. The installation process is expected to take up to 10 minutes (20 minutes for slow connections).

Note that the raw SafeGraph data is not publicly accessible and cannot be downloaded automatically. Access has to be requested through <a href = "https://www.safegraph.com/covid-19-data-consortium">SafeGraph COVID data consortium</a>. The CBG-level mobility data should be placed in `data/raw`. While the results of our analysis could be viewed by accessing the [/notebooks](/notebooks) directory, the code would not run correctly without the raw SafeGraph data.
    


Demo
-----
Since we cannot share the SafeGraph data directly, we provide a demo dataset to showcase our method. 

Synthetic time series are generated for each CBG in Washington State. The time series are based on 4 basis functions: two different sine waves and two exponentials (one rising and one falling).

One county was selected to be based on each of the 4 basis functions. Time series are generated for each CBG within these counties by multiplying the appropriate basis function by a random number. Thus, all CBGs within a single county are the same function multiplied by a scalar. Noise is added to the synthetic time series.

Synthetic time series are generated for the remaining CBGs using a combination of two of the basis functions. Each county is assigned a pair of basis functions, and the time series for each CBG is the product of one basis function + a random weight and the other function + another random weight. These time series are essentially products of two basis functions.

![basis functions](assets/synthmap-counties.png =250x250)


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data`, `make create_environment`, `make requirements` 
    ├── README.md   
    ├── requirements.txt   <- Requirements file for reproducing the Python analysis environment
    ├── setup.py           <- Installation script for the local package
    │
    ├── demo
    |   ├──obj             <- Directory to save computed helper objects
    |   |
    │   ├── Demo-Main-Analysis.ipynb    <- Main demo notebook with the dimesionality reduction + clustering pipeline applied to synthetic demo data
    │   └── make-synthetic-wa.R         <- Script to generate demo data: synthetic mobility dynamics in Washington state
    │
    ├── data
    │   ├── external       <- Data from external sources, e.g. shapefiles for plotting maps (from census.gov)  
    │   ├── interim        <- Intermediate data files
    │   ├── processed      <- Final data sets -- final clustering labels and final low-dimensional coordinates for every state
    │   └── raw            <- Raw data -- this is where SafeGraph mobility data should be placed 
    │
    ├── notebooks          <- Jupyter notebooks with the analysis code and the code to generate figures
    |   ├──obj             <- Directory to save computed helper objects
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
