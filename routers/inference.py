from fastapi import APIRouter, HTTPException
from typing import Dict
from models.inference import InferenceRequest
from containers.model_container import inference_model
import subprocess

import logging
import os

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
        running_containers = subprocess.check_output(["docker", "ps", "-q"]).decode().strip()
        if running_containers:
            raise HTTPException(
                status_code=400,
                detail="현재 GPU 사용중"
            )

        # inference_name: inference 결과 저장할 폴더 이름
        # inference_name 과 동일한 결과 저장 폴더가 있는지 확인. 있으면 에러.
        inference_path = f"{request.volume_path}/{request.project}/{request.subproject}/{request.task}/{request.version}/inference_results/{request.inference_name}"
        if os.path.exists(inference_path):
            raise HTTPException(
                status_code=400,
                detail=f"{request.inference_name} 이 이미 존재합니다."
            )

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