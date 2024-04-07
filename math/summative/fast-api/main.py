import uvicorn
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load the pickled model
with open("best_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

# Define a class to represent the request body
class PredictionRequest(BaseModel):
    X: List[float]

# Define endpoint for linear regression prediction
@app.post("/predict")
def predict_sales(request: PredictionRequest):
    # Extract input data from request
    X = request.X

    # Convert input data to numpy array and reshape it
    X_array = np.array(X).reshape(-1, 1)

    # Make predictions using the loaded model
    Y_pred = lr_model.predict(X_array)

    return {"predictions": Y_pred.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
