from pydantic import BaseModel

class InferenceRequest(BaseModel):
    project: str
    subproject: str
    task: str
    version: str