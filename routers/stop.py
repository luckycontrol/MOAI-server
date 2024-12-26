from fastapi import APIRouter, HTTPException
from typing import Dict
from utils.get_running_container import get_running_container

router = APIRouter()

@router.post("/stop")
async def stop() -> Dict:
    """
    학습 종료 엔드포인트

    Returns:
        Dict: 요청 처리 결과

        status, message
    """

    try:
        message = ""

        container = get_running_container()
        if container is not None:
            container.stop()
            container.remove(force=True)
            message = "학습 종료"
        
        else:
            message = "학습 중이지 않음"

        return {
            "status": "success",
            "message": message
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )