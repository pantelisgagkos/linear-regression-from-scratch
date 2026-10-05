import numpy as np
from linear_regression import LinearRegression

# 1. dummy dataset
rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2))
true_w = np.array([3.0, -2.0])
y = X @ true_w + 5 + rng.normal(scale=0.5, size=100)

# 2. split 80/20
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# 3. train
model = LinearRegression()
model.fit(X_train, y_train)

# 4. evaluate
print("weights:", model.w, "bias:", model.b)
print("train MSE:", model.mse(X_train, y_train))
print("test  MSE:", model.mse(X_test, y_test))
