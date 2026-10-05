import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None   # weights
        self.b = None   # bias





    # ---------- training ----------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        # προσθέτουμε στήλη 1 για το bias
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])

        # κανονικές εξισώσεις
        theta = np.linalg.inv(X_ext.T @ X_ext) @ X_ext.T @ y
        self.w = theta[:-1, :].ravel()
        self.b = float(theta[-1])

    # ---------- inference ----------
    def predict(self, X):
        X = np.asarray(X)
        return X @ self.w + self.b

    # ---------- αξιολόγηση ----------
    def evaluate(self, X, y):
        """Επιστρέφει (ŷ, MSE) στο σύνολο (X, y)."""
        if self.w is None:
            raise ValueError("Model not fitted yet.")

        X = np.asarray(X)
        y = np.asarray(y)
        y_hat = self.predict(X)
        mse = np.mean((y_hat - y) ** 2)
        return y_hat, mse


    def mse(self, X, y):
        """Mean-squared-error στο (X, y)."""
        _, mse = self.evaluate(X, y)
        return mse
