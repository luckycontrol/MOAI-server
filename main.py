from fastapi import FastAPI

from routers.train import router as train_router
from routers.inference import router as inference_router
from routers.tensorboard import router as tensorboard_router
from routers.stop import router as stop_router
from routers.export import router as export_router

app = FastAPI()
app.include_router(train_router)
app.include_router(inference_router)
app.include_router(tensorboard_router)
app.include_router(stop_router)
app.include_router(export_router)

# get 테스트
@app.get("/")
def hello_world():
    return {"message": "Hello World!"}

# post 테스트
@app.post("/")
def hello_world():
    return {"message": "Hello World!"}