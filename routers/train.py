from fastapi import APIRouter
from typing import Dict
from models.train import TrainRequest

from containers.yolo_container import train_yolo, inference_yolo

router = APIRouter()

@router.post("/train")
async def train(request: TrainRequest) -> Dict:
    """
    학습 시작 엔드포인트.

    Args:
        request (TrainRequest): 학습에 필요한 정보

    Returns:
        Dict: 요청 처리 결과
    """

    try:
        if request.train_params.model_type == "yolo":
            return await train_yolo(request)

        return {
            "status": "success",
            "message": "학습 시작",
            "data": request.model_dump()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )