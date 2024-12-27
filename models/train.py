from pydantic import BaseModel

class TrainParams(BaseModel):
    imgsz: int
    batch_size: int
    model_type: str
    weight_type: str
    epoch: int
    resume: bool

class TrainRequest(BaseModel):
    project: str
    subproject: str
    task: str
    version: str
    train_params: TrainParams

