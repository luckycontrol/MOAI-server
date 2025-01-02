from pydantic import BaseModel

class TensorboardParams(BaseModel):
    project: string
    subproject: string
    task: string
    version: string