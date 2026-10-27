# Weather Report Prediction and Data Analysis

This project predicts rainfall using the historical Indian rainfall and weather dataset from Kaggle. The dataset includes daily weather observations such as temperature, wind speed, air pressure, elevation, and rainfall.

## Project Structure

- data/india_weather_rainfall_data.xlsx
- outputs/graphs/
- outputs/results/
- models/rainfall_model.pkl
- src/data_preprocessing.py
- src/eda.py
- src/train_model.py
- src/evaluate_model.py
- src/predict_rainfall.py
- app.py
- README.md
- requirements.txt

## Dataset

The project uses the actual Kaggle dataset:

- ameydilipmorye/indian-rainfall-and-weather-data
- file: india_weather_rainfall_data.xlsx

The real features used for rainfall prediction are:

- avg_temp
- min_temp
- max_temp
- wind_speed
- air_pressure
- elevation
- latitude
- longitude

Target:

- rainfall

## Setup

1. Create a virtual environment (optional but recommended):
   python -m venv .venv
   .venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Train the model:
   python src/train_model.py

4. Run the dashboard:
   streamlit run app.py

## Model Workflow

1. Load the Excel dataset.
2. Remove duplicates and invalid rainfall values.
3. Convert the date column to datetime.
4. Sort data chronologically.
5. Split the data using an 80/20 chronological train-test split.
6. Train multiple regression models.
7. Compare MAE, RMSE, and R² score.
8. Save the best pipeline and results.

## Notes

- The project uses a chronological split to avoid future data leakage.
- The rainfall target is a regression value and is not treated as a classification problem.
- The dataset may contain many zero or low rainfall values; this is common and affects model behavior.
- The model predicts rainfall quantity in the same unit as the dataset.

## Commands

Train model:

python src/train_model.py

Run dashboard:

streamlit run app.py
