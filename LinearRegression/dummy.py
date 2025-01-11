#Dummy Regressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

def get_dummy_regressor(X_train, y_train, X_test, y_test):
    # DummyRegressor for y_train mean
    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(X_train, y_train)

    # Predicting
    y_dummy_pred = dummy_model.predict(X_test)

    # Calculating metrics
    dummy_rmse = mean_squared_error(y_test, y_dummy_pred)**0.5
    dummy_r2 = r2_score(y_test, y_dummy_pred)

    return (dummy_rmse, dummy_r2)