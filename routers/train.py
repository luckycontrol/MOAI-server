from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict
from models.train import TrainRequest

from containers.yolo_container import train_yolo, inference_yolo

router = APIRouter()

@router.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks) -> Dict:
    """
    학습 시작 엔드포인트.

    Args:
        request (TrainRequest): 학습에 필요한 정보
        background_tasks (BackgroundTasks): 백그라운드 작업 관리자

    Returns:
        Dict: 요청 처리 결과
    """

    try:
        if request.train_params.model_type == "yolo":
            background_tasks.add_task(train_yolo, request)

        return {
            "status": "success",
            "message": "학습 진행 중",
            "data": request.model_dump()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )