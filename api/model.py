import pickle
import numpy as np
import pandas as pd
from fastapi import HTTPException

linear_model = pickle.load(open("api/linear_model.pkl", "rb"))
scaler = pickle.load(open("api/scaler.pkl", "rb"))

def predict(features: dict):

    #Mapping Gender
    gender_mapping = {"Male": 0, "Female": 1}
    gender = gender_mapping.get(features["gender"], -1) #-1 for invalid cases

    # Mapping Education Level
    education_mapping = {
        "Bachelor's": 1,
        "Master's": 2,
        "PhD": 3
    }
    education_level = education_mapping.get(features["education_level"], -1)  #-1 for invalid cases

    input_data = pd.DataFrame([[
        gender,
        education_level,
        features["years_of_experience"]
    ]], columns=['Gender', 'Education Level', 'Years of Experience'])

    # Checking input data
    if gender == -1 or education_level == -1:
        raise HTTPException(status_code=400, detail="Invalid input for 'gender' or 'education_level'. Please check the values.")
    
    # Make the prediction
    input_data_scaled = scaler.transform(input_data)
    prediction = linear_model.predict(input_data_scaled)
    return int(prediction[0])