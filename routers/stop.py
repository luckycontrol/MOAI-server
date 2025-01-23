from fastapi import APIRouter, HTTPException
from typing import Dict
import docker
import logging
import shutil

from models.stop import StopParams  # stopParams가 정의된 모델

from utils import VOLUME_PATH

# 필요하다면 pydantic도 import 하세요. (이미 stop.py에서 StopParams를 import하는 것으로 가정)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/stop")
async def stop(stop_params: StopParams) -> Dict:
    """
    학습 또는 추론용 컨테이너 종료 엔드포인트

    Args:
        stop_params (StopParams): 학습 종료에 필요한 정보

    Returns:
        Dict: 요청 처리 결과
    """

    # 프로젝트, 서브프로젝트, 태스크, 버전을 바탕으로 컨테이너 이름 생성
    train_container_name = (
        f"{stop_params.project}_{stop_params.subproject}_{stop_params.task}_{stop_params.version}_train"
    )
    inference_container_name = (
        f"{stop_params.project}_{stop_params.subproject}_{stop_params.task}_{stop_params.version}_inference"
    )

    # Docker 클라이언트 초기화
    client = docker.from_env()

    # 중단한 컨테이너를 기록할 리스트
    stopped_containers = []

    # train 컨테이너 중단 시도
    try:
        train_container = client.containers.get(train_container_name)
        train_container.kill()
        train_container.remove()
        stopped_containers.append(train_container_name)

        # 학습중이던 컨테이너의 모델을 training_result 폴더 밖으로 이동
        training_result_path = f"{VOLUME_PATH}/{stop_params.project}/{stop_params.subproject}/{stop_params.task}/{stop_params.version}/training_result"
        if os.path.exists(training_result_path) and os.path.exists(f"{training_result_path}/weights"):
            shutil.move(f"{training_result_path}/weights", training_result_path)

    except docker.errors.NotFound:
        logger.info(f"컨테이너 {train_container_name} 는 동작 중이 아닙니다.")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"{train_container_name} 컨테이너 중단 중 오류 발생: {str(e)}"
        )

    # inference 컨테이너 중단 시도
    try:
        inference_container = client.containers.get(inference_container_name)
        inference_container.kill()
        inference_container.remove()
        stopped_containers.append(inference_container_name)
    except docker.errors.NotFound:
        logger.info(f"컨테이너 {inference_container_name} 는 동작 중이 아닙니다.")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"{inference_container_name} 컨테이너 중단 중 오류 발생: {str(e)}"
        )

    # 실제로 중단된 컨테이너가 하나도 없다면 예외 발생
    if not stopped_containers:
        raise HTTPException(
            status_code=400,
            detail="해당 이름의 컨테이너가 동작 중이지 않습니다."
        )

    # 성공적으로 컨테이너가 중단되었음을 반환
    return {
        "status": "success",
        "message": f"컨테이너({', '.join(stopped_containers)}) 중단 완료"
    }
