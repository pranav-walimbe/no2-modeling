# define path in which to create venv
VENV_PATH=<PATH> # example: /global/home/users/pranavwalimbe/conus_co2/setup/venv

# instantiate virtual environment named venv
python -m venv $VENV_PATH

# activate venv
source $VENV_PATH/bin/activate

# install necessary libraries 
pip install -r <PATH TO REQUIREMENTS> # example: /global/home/users/pranavwalimbe/conus_co2/setup/requirements.txt