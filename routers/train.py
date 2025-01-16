from fastapi import APIRouter, HTTPException
from typing import Dict
import subprocess

from models.train import TrainRequest
from containers.model_container import train_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info(f"[Train] 학습 요청 수신: {request}")

        # 현재 학습중이거나 예측중인 컨테이너가 있으면 X
        running_container_ids = subprocess.check_output(["docker", "ps", "-q"]).decode().strip().split()  # Split into list of IDs
        for container_id in running_container_ids:
            # Get the container name
            container_name = subprocess.check_output([
                "docker", "inspect", "--format={{.Name}}", container_id
            ]).decode().strip().lstrip('/')  # Remove leading slash from name
            if container_name.endswith('server') or container_name.endswith('_export') or container_name.endswith('_tensorboard'):
                continue
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"{container_name} 컨테이너가 이미 학습 실행중"
                )

        train_model(request)

        return {
            "status": "in_progress",
            "message": "학습 진행 중",
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )