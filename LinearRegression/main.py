#Our Model
import preprocessing
import my_model
import bootstrap
from cross_validation import rmse_scorer, cross_validate_model

# Load and preprocess the data
X_train, X_test, y_train, y_test, scaler = preprocessing.load_and_preprocess_data('data/DataTransformed.csv')

# Training the lineal model and getting the metrics
linear_model, mse, rmse, r2_adjusted = my_model.train_and_evaluate_model(X_train, X_test, y_train, y_test)

# Lets calculate the CIs of the metrics
mse_interval, r2_adjusted_interval = bootstrap.bootstrap_metrics(linear_model, X_train, y_train, X_test, y_test)
print(f"RMSE 95% confidence interval: {mse_interval}")
print(f"R²-adjusted 95% confidence interval: {r2_adjusted_interval}")

#Cross-Validation
rmse_cv, r2_cv = cross_validate_model(linear_model, X_train, y_train, cv=5)
print("-- CROSS VALIDATION --")
print(f"RMSE 95% confidence interval: {rmse_cv}")
print(f"R²-adjusted 95% confidence interval: {r2_cv}")

