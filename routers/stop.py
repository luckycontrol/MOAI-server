from fastapi import APIRouter, HTTPException
from typing import Dict
import docker
import docker.errors
import logging
import os
import shutil
import threading

thread_lock = threading.Lock()

from models.stop import StopParams  # stopParams가 정의된 모델

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

    # 학습 컨테이너 종료 시도
    try:
        train_container = client.containers.get(train_container_name)
    except docker.errors.NotFound:
        train_container = None

    if train_container:
        try:
            train_container.kill()
            # 학습중인 컨테이너의 모델 파일 이동 처리
            training_result_path = f"/moai/{stop_params.project}/{stop_params.subproject}/{stop_params.task}/{stop_params.version}/training_result"
            weights_path = os.path.join(training_result_path, "weights")
            if os.path.exists(training_result_path) and os.path.exists(weights_path) and any(f.endswith('.pt') for f in os.listdir(weights_path)):
                with thread_lock:
                    shutil.move(weights_path, training_result_path)
            return {
                "status": "success",
                "message": f"컨테이너({train_container_name}) 중단 완료"
            }
        except Exception as e:
            logger.exception(f"학습 컨테이너 종료 중 에러 발생: {e}")
            raise HTTPException(status_code=500, detail=f"학습 컨테이너 종료 중 오류 발생: {e}")

    # 학습 컨테이너가 없으면 추론 컨테이너 종료 시도
    try:
        inference_container = client.containers.get(inference_container_name)
    except docker.errors.NotFound:
        inference_container = None

    if inference_container:
        try:
            inference_container.kill()
            return {
                "status": "success",
                "message": f"컨테이너({inference_container_name}) 중단 완료"
            }
        except Exception as e:
            logger.exception(f"추론 컨테이너 종료 중 에러 발생: {e}")
            raise HTTPException(status_code=500, detail=f"추론 컨테이너 종료 중 오류 발생: {e}")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"해당 조합({stop_params.project}, {stop_params.subproject}, {stop_params.task}, {stop_params.version})의 컨테이너가 없습니다."
        )
