from pydantic import BaseModel

class StopParams(BaseModel):
    project: str
    subproject: str
    task: str
    version: str
    