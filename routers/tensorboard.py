from fastapi import APIRouter, HTTPException
from typing import Dict
from models.tensorboard import TensorboardParams
from containers.tensorboard_container import create_tensorboard_container, stop_tensorboard_container

router = APIRouter()

@router.post("/run_tensorboard")
async def run_tensorboard(request: TensorboardParams) -> Dict:
    """
    Tensorboard 시작 엔드포인트

    Args:
        request (TensorboardParams): Tensorboard 시작에 필요한 정보

    Returns:
        Dict: 요청 처리 결과
    """

    try:
        return create_tensorboard_container(request)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.post("/stop_tensorboard")
async def stop_tensorboard(request: TensorboardParams) -> Dict:
    """
    Tensorboard 종료 엔드포인트

    Args:
        request: (TensorboardParams): Tensorboard 종료에 필요한 정보

    Returns:
        Dict: 요청 처리 결과
    """

    try:
        return stop_tensorboard_container(request)
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )