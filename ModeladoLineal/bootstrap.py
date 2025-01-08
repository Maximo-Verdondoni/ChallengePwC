import numpy as np
import scipy.stats as stats
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score

def bootstrap_metrics(model, X_train, y_train, X_test, y_test, n_iterations=200):
    mse_scores = []
    r2_scores = []
    
    for _ in range(n_iterations):
        #Generating bootstrap samples
        X_resampled, y_resampled = resample(X_train, y_train, random_state=42)
        
        # fitting the modeal for each generated sample
        model.fit(X_resampled, y_resampled)
        
        # predicting
        y_pred_resampled = model.predict(X_test)
        
        # calculating metrics
        mse_scores.append(mean_squared_error(y_test, y_pred_resampled))
        r2_scores.append(r2_score(y_test, y_pred_resampled))
    
    #Then we calculate the 95% CI
    mse_lower = np.percentile(mse_scores, 2.5)
    mse_upper = np.percentile(mse_scores, 97.5)
    
    r2_lower = np.percentile(r2_scores, 2.5)
    r2_upper = np.percentile(r2_scores, 97.5)
    
    return (mse_lower, mse_upper), (r2_lower, r2_upper)
