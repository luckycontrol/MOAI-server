from pydantic import BaseModel

class TensorboardParams(BaseModel):
    project: str
    subproject: str
    task: str
    version: str