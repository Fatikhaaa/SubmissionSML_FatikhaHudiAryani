from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

# Path model saat di dalam Docker
MODEL_PATH = os.path.join("best_model", "model.pkl")

model = joblib.load(MODEL_PATH)

class InputData(BaseModel):
    age: float
    bmi: float
    children: int
    sex_male: int
    sex_female: int
    smoker_yes: int
    smoker_no: int
    region_northwest: int
    region_southeast: int
    region_southwest: int
    region_northeast: int

@app.get("/")
def home():
    return {"message": "API berjalan melalui Docker!"}

@app.post("/predict")
def predict(data: InputData):
    row = [
        data.age, data.bmi, data.children,
        data.sex_male, data.sex_female,
        data.smoker_yes, data.smoker_no,
        data.region_northwest, data.region_southeast,
        data.region_southwest, data.region_northeast
    ]

    prediction = model.predict([row])[0]
    return {"prediction": prediction}
