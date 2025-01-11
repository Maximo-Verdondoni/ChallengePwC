import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def cross_validate_model(model, X_train, y_train, cv=5, n_iterations=200):
    #Creating the scorer for RMSE
    rmse = make_scorer(rmse_scorer, greater_is_better=False)  # mayor valor no es mejor en RMSE

    #Cross-validation for RMSE
    cv_rmse_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=rmse)
    
    # Cross-validation for R-squared
    cv_r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    
    # RMSE to positive
    cv_rmse_scores = -cv_rmse_scores  

    # Calculating confidence intervals (95%) for RMSE and R²
    rmse_lower, rmse_upper = np.percentile(cv_rmse_scores, 2.5), np.percentile(cv_rmse_scores, 97.5)
    r2_lower, r2_upper = np.percentile(cv_r2_scores, 2.5), np.percentile(cv_r2_scores, 97.5)
    
    return (rmse_lower, rmse_upper), (r2_lower, r2_upper)