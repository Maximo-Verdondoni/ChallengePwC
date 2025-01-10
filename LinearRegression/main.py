#Our Model
import preprocessing
import my_model
import bootstrap

#Dummy Regressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the data
X_train, X_test, y_train, y_test = preprocessing.load_and_preprocess_data('data/DataTransformed.csv')

# Training the lineal model and getting the metrics
linear_model, mse, rmse, r2 = my_model.train_and_evaluate_model(X_train, X_test, y_train, y_test)

# Lets calculate the CIs of the metrics
mse_interval, r2_interval = bootstrap.bootstrap_metrics(linear_model, X_train, y_train, X_test, y_test)
print(f"RMSE 95% confidence interval: {mse_interval}")
print(f"R2 95% confidence interval: {r2_interval}")


# DummyRegressor for y_train mean
dummy_model = DummyRegressor(strategy="mean")
dummy_model.fit(X_train, y_train)

# Predicting
y_dummy_pred = dummy_model.predict(X_test)

# Calculating metrics
dummy_rmse = mean_squared_error(y_test, y_dummy_pred)**0.5
dummy_r2 = r2_score(y_test, y_dummy_pred)

print("\n--- DUMMY REGRESSOR ---")
print(f"Dummy Regressor MSE: {dummy_rmse}")
print(f"Dummy Regressor R2: {dummy_r2}")