import docker
from models.train import TrainRequest
from fastapi import HTTPException
from models.inference import InferenceRequest
import time
import threading
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = docker.from_env()

def train_model(request: TrainRequest):
    try:
        container_name = f"{request.project}_{request.subproject}_{request.task}_{request.version}_train"

        try:
            old_container = client.containers.get(container_name)
            old_container.stop()
            old_container.remove(force=True)
            logger.info(f"Removed old container: {old_container.name}")
        except docker.errors.NotFound:
            logger.info("No old container found.")

        volumes = {
            f"{request.volume_path}": {  # 변경된 볼륨 경로
                "bind": "/moai",
                "mode": "rw"
            }
        }

        # 컨테이너 내부에서 모델 실행 명령어
        train_command = [
            "conda",
            "run",
            "-n",
            f"{request.model_type}", # 가상 환경 이름은 모델 이름과 동일하게
            "python",
            "train.py",
            f"--project={request.project}",
            f"--subproject={request.subproject}",
            f"--task={request.task}",
            f"--version={request.version}"
        ]

        container = client.containers.run(
            image=f"{request.model_type}:latest",  # 이미지 이름 및 태그 지정
            name=container_name,
            volumes=volumes,
            device_requests=[
                docker.types.DeviceRequest(
                    count=-1,  # 모든 GPU 사용
                    capabilities=[["gpu"]]
                )
            ],
            tty=True,
            stdin_open=True, # -i 옵션 추가
            detach=True,
            shm_size="32G",  # 변경된 shm-size
        )

        def run_training(container, train_command):
            """학습을 실제로 수행하는 함수 (별도 스레드에서 실행)"""
            logger.info("[TRAINING] YOLO container training started...")
            exec_result = container.exec_run(train_command, stream=True)
            for output in exec_result.output:
                logger.info(output.decode('utf-8', errors='replace'))

            container.stop()
            container.remove(force=True)

        # 학습을 별도의 스레드에서 실행
        training_thread = threading.Thread(target=run_training, args=(container, train_command))
        training_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
        training_thread.start()

        start_time = time.time()
        timeout = 120  # 2분 타임아웃

        while True:
            file_path = f"{request.volume_path}/{request.project}/{request.subproject}/{request.task}/{request.version}/training_results/results.csv"
            logger.info(f"Checking file path: {file_path}")
            logger.info(f"File exists: {os.path.exists(file_path)}")
            
            if os.path.exists(file_path):
                logger.info("results.csv file has been found!")
                break
            
            if time.time() - start_time > timeout:
                raise HTTPException(status_code=408, detail="Training timeout: results.csv was not generated within 2 minutes")
            
            time.sleep(5)  # 5초마다 확인

    except Exception as e:
        container.stop()
        container.remove(force=True)
        
        raise HTTPException(status_code=400, detail=str(e))

def inference_model(request: InferenceRequest):   
    try:
        # 컨테이너 이름. 형식: project_subproject_task_version
        container_name = f"{request.project}_{request.subproject}_{request.task}_{request.version}_inference"

        try:
            old_container = client.containers.get(container_name)
            old_container.stop()
            old_container.remove(force=True)
            logger.info(f"[INFERENCE] Removed old container: {old_container.name}")
        except docker.errors.NotFound:
            logger.info("[INFERENCE] No old container found.")

        inference_command = [
            "conda",
            "run",
            "-n",
            f"{request.model_type}", # 가상 환경 이름은 모델 이름과 동일하게
            "python",
            "detect.py",
            f"--project={request.project}",
            f"--subproject={request.subproject}",
            f"--task={request.task}",
            f"--version={request.version}",
            f"--name={request.inference_name}",
            f"--imgsz={request.imgsz}",
        ]

        volumes = {
            f"{request.volume_path}": {  # 변경된 볼륨 경로
                "bind": "/moai",
                "mode": "rw"
            }
        }

        container = client.containers.run(
            image=f"{request.model_type}:latest",  # 이미지 이름 및 태그 지정
            name=container_name,
            volumes=volumes,
            device_requests=[
                docker.types.DeviceRequest(
                    count=-1,  # 모든 GPU 사용
                    capabilities=[["gpu"]]
                )
            ],
            tty=True,
            stdin_open=True, # -i 옵션 추가
            detach=True,
            shm_size="32G",  # 변경된 shm-size
        )

        def run_inference(container, inference_command):
            """학습을 실제로 수행하는 함수 (별도 스레드에서 실행)"""
            logger.info("[INFERENCE] YOLO container inference started...")
            exec_result = container.exec_run(inference_command, stream=True)
            for output in exec_result.output:
                logger.info(output.decode('utf-8', errors='replace'))

            container.stop()
            container.remove(force=True)

        # 예측을 별도의 스레드에서 실행
        inference_thread = threading.Thread(target=run_inference, args=(container, inference_command))
        inference_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
        inference_thread.start()

    except Exception as e:
        logger.error(e)

        container.stop()
        container.remove(force=True)
        raise HTTPException(status_code=400, detail=str(e))