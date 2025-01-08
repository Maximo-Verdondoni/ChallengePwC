import preprocessing
import model
import bootstrap

# Load and preprocess the data
X_train, X_test, y_train, y_test = preprocessing.load_and_preprocess_data('data/DataTransformed.csv')

# Training the lineal model and getting the metrics
linear_model, mse, rmse, r2 = model.train_and_evaluate_model(X_train, X_test, y_train, y_test)

#Lets see the sample metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# Lets calculate the CIs of the metrics
mse_interval, r2_interval = bootstrap.bootstrap_metrics(linear_model, X_train, y_train, X_test, y_test)
print(f"MSE 95% confidence interval: {mse_interval}")
print(f"R2 95% confidence interval: {r2_interval}")
