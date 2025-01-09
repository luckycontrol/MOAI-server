from pydantic import BaseModel

class ExportRequest(BaseModel):
    project: str
    subproject: str
    task: str
    version: str