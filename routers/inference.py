from fastapi import APIRouter, HTTPException
from typing import Dict
from models.inference import InferenceRequest
from containers.model_container import inference_model
from utils.check_yaml import load_yaml, save_yaml

import logging
import os
import yaml

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
        running_params = load_yaml()

        if running_params["is_running"] == True:
            raise HTTPException(
                status_code=400,
                detail=f"현재 GPU 사용중: {running_params['task']}"
            )

        # inference_name: inference 결과 저장할 폴더 이름
        # inference_name 과 동일한 결과 저장 폴더가 있는지 확인. 있으면 에러.
        inference_path = f"{request.volume_path}/{request.project}/{request.subproject}/{request.task}/{request.version}/inference_results/{request.inference_name}"
        if os.path.exists(inference_path):
            raise HTTPException(
                status_code=400,
                detail=f"{request.inference_name} 이 이미 존재합니다."
            )

        save_yaml(
            request=request,
            is_running=True,
            is_train=False
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