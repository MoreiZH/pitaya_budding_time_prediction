# pitaya_budding_time_prediction
Budding time of pitaya is related to temperature and sunshine duration. This project uses temperature and sunshine duration to find the budding feature, and use that feature to predict the budding time.
# Installation
This tool is written with python3.8, the packages used are listed in requirements.txt, install them by command line: *pip install -r requirements.txt*
# Usage
1st step <br>
missing_value_completion.py will produce 24 hours' hourly temperature based a hourly level record with some of the data missing.<br>
2nd step <br>
data_merger_weather.py will merge the temperature and sunshine duration data.<br>
3rd step <br>
feature_engineering.py will engeneer the feature, predict the budding time and calculate the loss.<br>
# Output
The output file will be created in current work directory.

