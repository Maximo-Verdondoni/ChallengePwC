from pydantic import BaseModel

class PredictionData(BaseModel):
    gender: str
    education_level: str
    years_of_experience: int