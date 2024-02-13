from fastapi import FastAPI
from joblib import load
import numpy as np
import pandas as pd
from pydantic import BaseModel
model = load(open("model11.pkl", "rb"))

class Apple(BaseModel):
    Size: float
    Weight: float
    Sweetness: float
    Crunchiness: float
    Juiciness: float
    Ripeness: float
    Acidity: float


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predictArray(data: Apple):
    d = {"Size":[data.Size], "Weight":[data.Weight], "Sweetness":[data.Sweetness], "Crunchiness":[data.Crunchiness], "Juiciness":[data.Juiciness], "Ripeness":[data.Ripeness], "Acidity":[data.Acidity]}
    df = pd.DataFrame(d)
    prediction = model.predict(df)
    print(prediction)
    return {"prediction": prediction.tolist()}