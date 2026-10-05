import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from linear_regression import LinearRegression
from sklearn.linear_model import LinearRegression as SkLR

# ------------- settings -------------
N_RUNS = 20
TEST_SIZE = 0.30
SEED = 42
# ------------------------------------

X, y = fetch_california_housing(return_X_y=True, as_frame=False)

rmse_custom  = []
rmse_sklearn = []

for i in range(N_RUNS):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED + i
    )

    # δική σου υλοποίηση
    my_model = LinearRegression()
    my_model.fit(X_train, y_train)
    _, mse_my = my_model.evaluate(X_test, y_test)
    rmse_custom.append(np.sqrt(mse_my))

    # scikit-learn
    sk_model = SkLR().fit(X_train, y_train)
    y_pred_sk = sk_model.predict(X_test)
    rmse_sklearn.append(np.sqrt(mean_squared_error(y_test, y_pred_sk)))

# --------- αποτελέσματα ----------
print(f"Custom  RMSE: mean={np.mean(rmse_custom):.4f} ± {np.std(rmse_custom):.4f}")
print(f"sklearn RMSE: mean={np.mean(rmse_sklearn):.4f} ± {np.std(rmse_sklearn):.4f}")

