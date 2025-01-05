from pydantic import BaseModel

class InferenceRequest(BaseModel):
    volume_path: str
    project: str
    subproject: str
    task: str
    version: str
    inference_name: str
    imgsz: int