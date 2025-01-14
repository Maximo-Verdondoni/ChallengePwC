from fastapi import FastAPI
from api.model import predict
from api.schemas import PredictionData

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Salary Prediction API is running"}

@app.post("/predict/")
def predict_salary(data: PredictionData):
    features = data.dict()
    salary = predict(features)
    return {"predicted_salary": salary}