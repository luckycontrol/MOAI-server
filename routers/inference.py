from fastapi import APIRouter
from typing import Dict
from models.inference import InferenceRequest

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
        return {
            "status": "success",
            "message": "추론 시작",
            "data": request.model_dump()
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )