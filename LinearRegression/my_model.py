from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    
    # Fitting the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculating metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Adjusted R^2
    n = X_test.shape[0]  
    p = X_test.shape[1] 
    r2_adjusted = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    
    return model, mse, rmse, r2_adjusted

