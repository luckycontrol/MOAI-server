from fastapi import APIRouter, HTTPException
from typing import Dict
from models.train import TrainRequest

from containers.yolo_container import train_yolo

router = APIRouter()

@router.post("/train")
async def train(request: TrainRequest) -> Dict:
    """
    학습 시작 엔드포인트.

    Args:
        request (TrainRequest): 학습에 필요한 정보
        background_tasks (BackgroundTasks): 백그라운드 작업 관리자

    Returns:
        Dict: 요청 처리 결과
    """

    try:
        train_yolo(request)
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )