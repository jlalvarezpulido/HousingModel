# README

This project is a Terminal based application that could be run in the terminal or by double clicking the file icon so long as python is in the OS's PATH.

## Files

- housingmodel.py
- Housing.csv
- log.csv

## Python Packages used

_not all packages need to be installed directly as some are depencies and will install along with the main 4_

Package         Version
--------------- -----------
- contourpy       1.2.0
- cycler          0.12.1
- fonttools       4.50.0
- joblib          1.3.2
- kiwisolver      1.4.5
- matplotlib      3.8.3
- numpy           1.26.4
- packaging       24.0
- pandas          2.2.1
- pillow          10.2.0
- pip             22.0.2
- pyparsing       3.1.2
- python-dateutil 2.9.0.post0
- pytz            2024.1
- scikit-learn    1.4.1.post1
- scipy           1.13.0
- seaborn         0.13.2
- setuptools      59.6.0
- six             1.16.0
- threadpoolctl   3.4.0
- tzdata          2024.1

## Installation
### Install latest version of python

use link to download python from offical site https://www.python.org/downloads/

__All commands should be run with adminstrator privelages__

_Be sure to Download the python version that matches your OS and Processor type_

All installation will take place in the Terminal, Command Prompt, or PowerShell; Depending on the OS

### upgrade pip

use command below to upgrade pip

`pip3 install --upgrade pip`

_pip is the standard package manager for Python_

### Create and activate a virtual environment

#### Crete a virtual environment
Create a virtual environment
#### On Linux or MacOS terminal

`python3 -m venv .venv`

#### On Windows

`python -m venv .venv`

_Be sure to create environemt on the same directory that the python files, and csv's located_

#### On Linux or MacOS terminal

On Unix like terminals use command below to activate environment

`source .venv/bin/activate`

#### On Windows

`cd .venv`

On Command Prompt run the following command

`venv\Scripts\activate.bat`

On PowerShell run the following command

`venv\Scripts\Activate.ps1`

### Install packages and Libraries

`pip install pandas`

`pip install matplotlib`

`pip install seaborn`

`pip install -U scikit-learn`

### Run the application

_All python files and csv's should be in the same directory for this to work_

run the application with the python command

`python housingmodel.py`






## Data
Data was sourced from Kaggle

https://www.kaggle.com/datasets/sukhmandeepsinghbrar/housing-price-dataset

The raw data set will contain the following columns:
-	price (Housing price of recent listing or previously sold at price)
-	bathrooms (Number of bathrooms)
-	grade (Overall grade rating 1 to 13)
-	waterfront (Indicates if property has waterfront view (0 for no 1 for yest))
-	sqft_lot (Total size in square feet)
-	sqft_living (Living area size in square feet)
-	long (Longitude coordinate of property location)
-	lat (Latitude coordinate of property location)
-	bedrooms (Number of bedrooms)

## Commands

List of commands used in the application

`predict` : Will begin a prediction query

`city_list` : display a list of popular cities in Washington with latitude and logitude values

`commands` : display a list of valid commands

`heatmap` : displays descriptvie heatmap of the data

`distribution` : displays descriptvie price distribution of the data

`scatterplot` : displays descriptvie scatter plot of the price vs sqft_living

`r2`: displays the R-Squared value

`mae` : displays the mean absolute error

## Good Latitude and Longitude Test Values
While The values displayed are from Washington\nAny valid latitude and longitude could be used to extrapolate the prediction.
- Seattle: Latitude 47.60, Longitude -122.33
- Spokane: Latitude 47.65, Longitude -117.42
- Tacoma: Latitude 47.25, Longitude -122.44
- Bellevue: Latitude 47.61, Longitude -122.20
- Everett: Latitude 47.97, Longitude -122.20
- Kent: Latitude 47.38, Longitude -122.23
- Renton: Latitude 47.48, Longitude -122.21
- Federal Way: Latitude 47.32, Longitude -122.31

## Algorithm

Scikit's Random Forest Regression

This alogrithm was chosen as it works great with continues numerical data.

For categorical data a Decision Tree Regressor would be a better fit.
