from pydantic import BaseModel

class TrainParams(BaseModel):
    imgsz: int
    batch_size: int
    model_type: str
    weight_type: str
    epoch: int
    patience: int
    resume: bool

class Hyps(BaseModel):
    lr: float
    hsv: bool
    degrees: float
    translate: float
    scale: float
    flipud: bool
    fliplr: bool
    mosaic: bool

class TrainRequest(BaseModel):
    project: str
    subproject: str
    task: str
    version: str
    train_params: TrainParams

