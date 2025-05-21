# House Price Prediction

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
[![Kaggle](https://img.shields.io/badge/Kaggle-House%20Prices-blue)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Overview
This project uses machine learning to predict house prices based on the [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). It includes a detailed preprocessing pipeline, explores various regression models, and generates predictions for Kaggle submission. The main models are Linear Regression (with variants) and Random Forest Regression, implemented both from scratch and with scikit-learn. The final model submitted to Kaggle is scikit-learn's `RandomForestRegressor`.

## Dataset
The Ames Housing Dataset includes 79 features about homes in Ames, Iowa, with `SalePrice` as the target. Key features used include:

- `OverallQual`: Overall material and finish quality
- `GrLivArea`: Above-ground living area (sq. ft.)
- `TotalBsmtSF`: Total basement area (sq. ft.)
- `1stFlrSF`: First floor area (sq. ft.)
- `YearBuilt`: Year of construction
- `YearRemodAdd`: Year of remodeling
- `MasVnrArea`: Masonry veneer area (sq. ft.)
- `BsmtFinSF1`: Type 1 finished basement area
- `WoodDeckSF`: Wood deck area (sq. ft.)
- `2ndFlrSF`: Second floor area (sq. ft.)
- `OpenPorchSF`: Open porch area (sq. ft.)
- `LotArea`: Lot size (sq. ft.)

The data is sourced from `train.csv` and `test.csv` on Kaggle.

## Preprocessing
Data preparation involves these steps:

1. **Load Data**: Import and summarize `train.csv`.
2. **Check Invalid Values**: Ensure no negative values in numeric columns where inappropriate.
3. **Handle Missing Data**: Drop columns with >80% missing values (e.g., `Alley`, `PoolQC`); impute others with median (numeric) or mode (categorical).
4. **Correct Skewness**: Apply log transformation to skewed features (e.g., `LotArea`, `GrLivArea`) and `SalePrice` if skewness > 0.5.
5. **Handle Outliers**: Clip values using Z-score (>3) and IQR bounds.
6. **Explore Features**: Visualize distributions and relationships with `SalePrice`.
7. **Analyze Correlations**: Remove collinear features (e.g., `TotRmsAbvGrd`).
8. **Encode Categorical Features**: Use ordinal encoding (e.g., `ExterQual`) and one-hot encoding for others.
9. **Build Pipeline**: Combine `SimpleImputer`, `StandardScaler`, and `OneHotEncoder`.

The data is split into 80% training and 20% validation sets.

## Models
The project implements:

1. **Linear Regression** (from scratch):
   - Variants: Standard, Ridge (L2), LASSO (L1), Elastic Net (L1+L2).
   - Optimized with gradient descent and Adam optimizer.
2. **Random Forest Regression**:
   - From scratch: Custom trees with bootstrap sampling.
   - Scikit-learn: `RandomForestRegressor` (200 trees, no max depth).

The final Kaggle model is scikit-learn’s `RandomForestRegressor` trained on the original `SalePrice`.

## Evaluation
Performance metrics on the validation set:

| Model                          | MAPE       | RMSE      | R²      |
|--------------------------------|------------|-----------|---------|
| Linear Regression (scratch)    | ~0.0077    | ~0.1319   | -       |
| Random Forest (scratch)        | ~0.0087    | ~0.1598   | ~0.8631 |
| RandomForestRegressor (sklearn)| ~0.0091 (log scale) | - | - |
| RandomForestRegressor (submission) | ~0.099 (original scale) | - | - |

Kaggle uses Root Mean Squared Logarithmic Error (RMSLE) for scoring. Visualizations include target distribution, correlation heatmaps, feature importance, and actual vs. predicted plots.

## Requirements
Install the required Python libraries with:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
Installation
Clone the repository:
bash

git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
Install dependencies (see above).
Download train.csv and test.csv from Kaggle.
Usage
Open house_price_prediction.ipynb in Jupyter or Google Colab.
Place train.csv and test.csv in the project directory or upload them.
Run all notebook cells to preprocess data, train models, and create submission.csv.
Submit submission.csv to Kaggle.
Results
The notebook includes:

Target Distribution: Histograms/KDE of SalePrice.
Correlation Heatmap: Highlights key correlations (e.g., OverallQual: 0.79).
Feature Importance: Ranks predictors like OverallQual.
Actual vs. Predicted: Shows model accuracy.
Residuals: Displays error patterns.
The final RandomForestRegressor achieves ~9.9% MAPE on the validation set.

Future Work
Test models like XGBoost or Gradient Boosting.
Add interaction terms in feature engineering.
Perform hyperparameter tuning (e.g., grid search).
Improve imputation methods.
Optimize directly for RMSLE.
License
This project is licensed under the MIT License.

Acknowledgments
Kaggle House Prices Competition
Inspired by machine learning tutorials and Kaggle community kernels.
