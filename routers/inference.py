from fastapi import APIRouter, HTTPException
from typing import Dict
from models.inference import InferenceRequest
from containers.yolo_container import inference_yolo
import logging

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
        return inference_yolo(request)

    except Exception as e:
        logger.error(e)

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )