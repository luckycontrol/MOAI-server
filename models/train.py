from pydantic import BaseModel

class TrainRequest(BaseModel):
    volume_path: str
    project: str
    subproject: str
    task: str
    version: str
    model_type: str

