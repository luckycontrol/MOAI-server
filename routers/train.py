from fastapi import APIRouter, HTTPException
from typing import Dict
import os
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

        # # version 이 이미 있는 경우 중복되므로 학습 X
        # path = f"d:/moai_test/{request.project}/{request.subproject}/{request.task}/{request.version}"
        # if os.path.exists(path):
        #     raise HTTPException(
        #         status_code=400,
        #         detail="프로젝트에 버전이 이미 존재함"
        #     )

        running_container_ids = subprocess.check_output(["docker", "ps", "-q"]).decode().strip().split()  # Split into list of IDs
        for container_id in running_container_ids:
            # Get the container name
            container_name = subprocess.check_output([
                "docker", "inspect", "--format={{.Name}}", container_id
            ]).decode().strip().lstrip('/')  # Remove leading slash from name
            if not container_name.endswith('_export'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Non-export container running: {container_name}"
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