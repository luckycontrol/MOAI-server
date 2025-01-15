from fastapi import APIRouter, HTTPException
from typing import Dict
from models.inference import InferenceRequest
from containers.model_container import inference_model
import subprocess

import logging
import os
import shutil

from utils import VOLUME_PATH

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/inference")
async def inference(request: InferenceRequest) -> Dict:
    """
    추론 시작 엔드포인트

    Args:
        request (InferenceRequest): 추론에 필요한 정보
    
    Returns:
        Dict: 요청 처리 결과
    """

    try:
        # 현재 학습중이거나 예측중이면 예측 X
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

        # inference_result 폴더 제거
        inference_result_path = f"{VOLUME_PATH}/{request.project}/{request.subproject}/{request.task}/{request.version}/inference_result"
        if os.path.exists(inference_result_path):
            shutil.rmtree(inference_result_path)

        inference_model(request)

        return {
            "status": "in_progress",
            "message": "예측 진행 중",
        }

    except Exception as e:
        logger.error(e)

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )