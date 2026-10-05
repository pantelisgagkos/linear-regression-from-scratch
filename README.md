# Linear Regression from Scratch

Linear Regression implemented from scratch using NumPy and validated against scikit-learn.

The project was developed as part of the Artificial Intelligence course at Harokopio University.

## Project Overview

The goal of the project was to implement Ordinary Least Squares (OLS) linear regression without using a machine learning library for the model itself.

The implementation includes:

- Model training using the normal equation
- Weight and bias estimation
- Prediction on unseen data
- Mean Squared Error (MSE) evaluation
- Testing on synthetic data
- Validation on the California Housing dataset
- Comparison with scikit-learn's `LinearRegression`

## Implementation

The model estimates the parameters of the linear regression equation:

`y = Xw + b`

The bias term is incorporated by extending the feature matrix with a column of ones.

The model parameters are computed using the normal equation:

`θ = (XᵀX)⁻¹Xᵀy`

## Files

### `linear_regression.py`

Custom implementation of linear regression using NumPy.

Includes:

- `fit()`
- `predict()`
- `evaluate()`
- `mse()`

### `test_lr.py`

Tests the implementation on a synthetic dataset with known model parameters.

The synthetic target is generated approximately as:

`y = 3x₁ - 2x₂ + 5 + noise`

### `test_california.py`

Evaluates the custom model on the California Housing dataset and compares its performance with scikit-learn.

The experiment uses:

- 20 independent train/test splits
- 70% training data
- 30% test data
- RMSE as the evaluation metric

## Results

The average results over 20 runs were:

| Model | Mean RMSE | Standard Deviation |
|---|---:|---:|
| Custom Linear Regression | 0.7280 | 0.0095 |
| scikit-learn LinearRegression | 0.7280 | 0.0095 |

The two implementations produced matching results to four decimal places.

This validates the correctness of the custom Ordinary Least Squares implementation.

## Technologies

- Python
- NumPy
- scikit-learn
- Linear Regression
- Ordinary Least Squares
- Machine Learning
- Model Evaluation
- RMSE

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the synthetic test:

```bash
python3 test_lr.py
```

Run the California Housing comparison:

```bash
python3 test_california.py
```

## Academic Context

This project was completed for the Artificial Intelligence course at the Department of Informatics and Telematics, Harokopio University.
