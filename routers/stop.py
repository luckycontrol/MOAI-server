from fastapi import APIRouter, HTTPException
from typing import Dict
import docker
import logging
import subprocess

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/stop")
async def stop() -> Dict:
    """
    학습 종료 엔드포인트

    Args:
        request (StopParams): 학습 종료에 필요한 정보

    Returns:
        Dict: 요청 처리 결과

        status, message
    """

    try:
        running_containers = subprocess.check_output(["docker", "ps", "-q"]).decode().strip()
        if not running_containers:
            raise HTTPException(
                status_code=400,
                detail="현재 학습중인 컨테이너가 없습니다."
            )

        container_name = running_containers.split()[0]

        client = docker.from_env()
        container = client.containers.get(container_name)
        container.stop()

        return {
            "status": "success",
            "message": f"컨테이너({container_name}) 학습 종료"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
