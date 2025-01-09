from pydantic import BaseModel

class TrainRequest(BaseModel):
    project: str
    subproject: str
    task: str
    version: str
    model_type: str

