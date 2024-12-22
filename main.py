from fastapi import FastAPI, HTTPException

from models.inference import InferenceRequest

from routers.train import router as train_router
from routers.inference import router as inference_router

app = FastAPI()
app.include_router(train_router)
app.include_router(inference_router)

@app.get("/")
def hello_world():
    return {"message": "Hello World!"}