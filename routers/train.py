from fastapi import APIRouter, HTTPException
from typing import Dict
from models.train import TrainRequest
import docker

from containers.yolo_container import train_yolo
from utils.is_container_running import is_container_running

router = APIRouter()
client = docker.from_env()

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
        # 현재 training 또는 inference 중인 컨테이너가 있는지 확인
        is_running = is_container_running()
        # 활성화된 container 가 있다면 GPU 사용중이므로 작업을 진행하지 않음
        if is_running:
            return {
                "status": "error",
                "message": "현재 GPU 사용중"
            }

        if request.train_params.model_type == "yolo":
            train_yolo(request)

        return {
            "status": "in_progress",
            "message": "학습 진행 중",
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )