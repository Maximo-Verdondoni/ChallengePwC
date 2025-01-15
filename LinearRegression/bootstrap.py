import numpy as np
import scipy.stats as stats
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score

def bootstrap_metrics(model, X_train, y_train, X_test, y_test, n_iterations=200):
    mse_scores = []
    r2_scores = []
    r2_adjusted_scores = []
    
    n = X_test.shape[0]  # Number of observations in the test set
    p = X_test.shape[1]  # Number of predictors
    
    for _ in range(n_iterations):
        #Generating bootstrap samples
        X_resampled, y_resampled = resample(X_train, y_train)
        
        # fitting the modeal for each generated sample
        model.fit(X_resampled, y_resampled)
        
        # predicting
        y_pred_resampled = model.predict(X_test)
        
        # calculating metrics
        mse = mean_squared_error(y_test, y_pred_resampled)
        r2 = r2_score(y_test, y_pred_resampled)
        r2_adjusted = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        r2_adjusted_scores.append(r2_adjusted)
    
    #Then we calculate the 95% CI
    rmse_lower = np.percentile(mse_scores, 2.5)**0.5
    rmse_upper = np.percentile(mse_scores, 97.5)**0.5
    
    r2_adjusted_lower = np.percentile(r2_adjusted_scores, 2.5)
    r2_adjusted_upper = np.percentile(r2_adjusted_scores, 97.5)
    
    return (rmse_lower, rmse_upper), (r2_adjusted_lower, r2_adjusted_upper)
