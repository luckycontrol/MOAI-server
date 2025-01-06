from fastapi import APIRouter, HTTPException
from typing import Dict
from utils.get_running_container import get_running_container
from models.stop import StopParams
import docker
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()

client = docker.from_env()

@router.post("/stop")
async def stop(request: StopParams) -> Dict:
    """
    학습 종료 엔드포인트

    Args:
        request (StopParams): 학습 종료에 필요한 정보

    Returns:
        Dict: 요청 처리 결과

        status, message
    """

    try:
        # 컨테이너 이름 구성: project_subproject_task_version
        container_name = f"{request.project}_{request.subproject}_{request.task}_{request.version}"

        # 특정 이름의 컨테이너 가져오기
        containers = client.containers.list(all=True)
        container = [c for c in containers if c.name == f"{container_name}/train" or c.name == f"{container_name}/inference"][0]
        
        if container.status == "running":
            if container.name.endswith("inference"):
                os.removedirs(f"{request.volume_path}/{request.project}/{request.subproject}/{request.task}/{request.version}/inference_results")

            # 컨테이너가 실행 중이면 stop 후 remove
            container.stop()
            
            logger.info(f"[STOP] 컨테이너({container_name}) 학습 및 예측 종료")

        else:
            # 존재하지 않거나 실행 중인 컨테이너 없음
            logger.info(f"[STOP] 학습 중인 컨테이너({container_name})가 없음")

        return {
            "status": "success",
            "message": f"컨테이너({container_name}) 학습 종료"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
