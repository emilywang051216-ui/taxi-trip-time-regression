# Taxi Trip Time Regression

A large-scale machine learning project that predicts taxi trip duration using geographic, temporal, and route-level features.

The final pipeline uses LightGBM, leakage-safe out-of-fold target encoding, early stopping, and multi-seed ensembling.

## Project Overview

The objective is to predict `trip_time`, measured in minutes, from trip and pickup information.

The original competition data contains approximately:

- 2.14 million training rows
- 152,000 test rows

The evaluation metric is **Mean Absolute Error (MAE)**, where a lower score is better.

## Key Features

The model uses:

- recorded trip distance
- Haversine pickup-to-drop-off distance
- log-transformed distance
- pickup hour and weekday
- rush-hour indicator
- weekend indicator
- pickup and drop-off coordinates
- pickup-zone target encoding
- drop-off-zone target encoding
- route target encoding

## Modelling Approach

The project workflow includes:

1. loading and validating the data
2. exploratory data analysis
3. geographic and temporal feature engineering
4. rounded coordinate zones and route construction
5. five-fold out-of-fold target encoding
6. LightGBM training with early stopping
7. validation using MAE
8. error analysis and feature importance
9. multi-seed final-model ensembling
10. submission generation

## Project Structure

```text
taxi-trip-time-regression/
├── README.md
├── taxi_trip_time_regression.ipynb
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── results/
```

## Dataset

The competition dataset is not included in this repository.

Place these files in the `data/` directory:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

The expected training columns include:

```text
pickup_lat
pickup_long
dropoff_lat
dropoff_long
distance_km
hour
weekday
trip_time
```

The test data must also include an `id` column for submission generation.

## Feature Engineering

### Geographic features

The notebook calculates Haversine distance from the pickup and drop-off coordinates and creates a log-transformed distance feature.

### Temporal features

Rush-hour and weekend indicators are created from `hour` and `weekday`.

Rush-hour values used in the model are:

```text
7, 8, 9, 16, 17, 18, 19
```

### Zone and route features

Pickup and drop-off coordinates are rounded to four decimal places to form location identifiers:

- `puk`: pickup zone
- `dok`: drop-off zone
- `route`: pickup-zone and drop-off-zone pair

### Target encoding

Pickup, drop-off, and route identifiers are converted into smoothed mean-trip-time features.

Training encodings are generated out-of-fold so that a row is never encoded using its own target value.

## Model

The final model is LightGBM with the L1 regression objective.

Main parameters:

```text
objective: regression_l1
metric: mae
learning_rate: 0.06
num_leaves: 127
min_child_samples: 100
feature_fraction: 0.8
bagging_fraction: 0.8
bagging_freq: 1
lambda_l2: 2.0
```

Early stopping is used to select the number of boosting rounds. Final predictions are averaged across models trained with three random seeds:

```text
42, 7, 2024
```

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/6hliu/taxi-trip-time-regression.git
cd taxi-trip-time-regression
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the data

Place `train.csv` and `test.csv` inside the `data/` directory.

### 4. Run the notebook

```bash
jupyter notebook taxi_trip_time_regression.ipynb
```

The notebook includes:

```python
DEBUG_MODE = True
```

Debug mode runs the pipeline on a smaller sample. Change it to:

```python
DEBUG_MODE = False
```

to train on the full dataset.

## Outputs

Running the notebook can generate:

```text
results/trip_time_distribution.png
results/distance_vs_trip_time.png
results/median_trip_time_by_hour.png
results/actual_vs_predicted.png
results/absolute_error_distribution.png
results/feature_importance.png
submission.csv
```

## Previous Competition Results

The development process improved the leaderboard MAE from:

| Version | Leaderboard MAE |
|---|---:|
| CatBoost baseline | 5.06874 |
| Tuned CatBoost | 4.99548 |

The final LightGBM score should be added after the completed submission result is confirmed.

## Limitations

- Full training is computationally expensive on CPU.
- The validation split assumes row ordering has useful meaning.
- Rare and unseen routes receive target encodings close to the global mean.
- The data does not directly include traffic, weather, or road-network conditions.
- Rounded coordinate zones are an approximation of real geographic regions.

## Future Improvements

- compare time-aware and random validation
- add cyclical hour and weekday features
- test multiple geographic grid resolutions
- add route direction and coordinate-difference features
- tune LightGBM using cross-validation
- blend LightGBM and CatBoost predictions

## Technologies

- Python
- pandas
- NumPy
- matplotlib
- scikit-learn
- LightGBM
- Jupyter Notebook

## Author

Haoran Liu
