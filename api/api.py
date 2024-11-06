from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
from typing import List
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    data: List


class UpdateModelRequest(BaseModel):
    version: int


app = FastAPI()


host = os.getenv("MLFLOW_HOST", "localhost")
port = os.getenv("MLFLOW_PORT", "8080")
model_name = os.getenv("MODEL_NAME", "tracking-iris")
model_version = os.getenv("MODEL_VERSION", "1")

uri = f"http://{host}:{port}"
model_uri = f"models:/{model_name}/1"
logger.info(f"MLflow tracking URI: {uri}")
mlflow.set_tracking_uri(uri=uri)
model = mlflow.pyfunc.load_model(model_uri)


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        X = np.array(request.data).astype(np.float64)
        predictions = model.predict(X)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update-model")
def update_model(request: UpdateModelRequest):
    global model
    global model_uri
    try:
        model_uri = f"models:/tracking-iris/{request.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        return {"status": "Model updated", "new_version": request.version}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
