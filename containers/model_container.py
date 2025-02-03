import docker
from models.train import TrainRequest
from models.inference import InferenceRequest
from models.export import ExportRequest
from fastapi import HTTPException
import time
import threading
import os
import logging
import yaml

from utils import VOLUME_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

client = docker.from_env()

def train_model(request: TrainRequest):
    try:
        # 컨테이너 이름 형식: project_subproject_task_version_train
        container_name = f"{request.project}_{request.subproject}_{request.task}_{request.version}_train"

        version_path = f"/moai/{request.project}/{request.subproject}/{request.task}/{request.version}"
        if not os.path.exists(version_path):
            os.makedirs(version_path)
            logger.info(f"Created directory: {version_path}")

        train_config_path = f"{version_path}/train_config.yaml"
        with open(train_config_path, "w") as f:
            train_config = {}
            train_config["project"] = request.project
            train_config["subproject"] = request.subproject
            train_config["task"] = request.task
            train_config["version"] = request.version
            train_config["model_type"] = request.model_type
            train_config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            yaml.dump(train_config, f)

        try:
            old_container = client.containers.get(container_name)
            old_container.stop()
            old_container.remove(force=True)
            logger.info(f"Removed old container: {old_container.name}")
        except docker.errors.NotFound:
            logger.info("No old container found.")

        # 모델 컨테이너의 볼륨 관리
        volumes = {
            f"{VOLUME_PATH}": {  # 변경된 볼륨 경로
                "bind": "/moai",
                "mode": "rw"
            }
        }

        # 컨테이너 내부에서 모델 실행 명령어
        train_command = [
            "bash",
            "-c",
            f"source model_env/bin/activate && python train.py "
            f"--project {request.project} "
            f"--subproject {request.subproject} "
            f"--task {request.task} "
            f"--version {request.version} "
        ]

        try:
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
            logger.info(f"Container {container_name} started successfully.")
        except Exception as e:
            logger.error(f"Failed to start container {container_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to start container: {str(e)}")

        def run_training(container, train_command):
            """학습을 실제로 수행하는 함수 (별도 스레드에서 실행)"""
            logger.info("[TRAINING] container training started...")
            exec_result = container.exec_run(train_command, stream=True)

            for output in exec_result.output:
                logger.info(output.decode('utf-8', errors='replace'))
            logger.info("[TRAINING] container training finished...")

            container.stop()
            container.remove(force=True)

        # 학습을 별도의 스레드에서 실행
        training_thread = threading.Thread(target=run_training, args=(container, train_command))
        training_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
        training_thread.start()

    except Exception as e:
        logger.info(f"Training failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

def inference_model(request: InferenceRequest):
    try:
        # 컨테이너 이름 생성
        container_name = f"{request.project}_{request.subproject}_{request.task}_{request.version}_inference"
        logger.info(f"Container name: {container_name}")

        # 타임스탬프 기록 (전체 시작 시간)
        overall_start = time.time()

        # train_config 파일 읽기
        train_config_path = f"/moai/{request.project}/{request.subproject}/{request.task}/{request.version}/train_config.yaml"
        start = time.time()
        with open(train_config_path, "r") as f:
            train_config = yaml.safe_load(f)
        logger.info(f"Read train_config in {time.time() - start:.3f} seconds")

        model_type = train_config["model_type"]

        # 기존 컨테이너 제거
        start = time.time()
        try:
            old_container = client.containers.get(container_name)
            old_container.stop()
            old_container.remove(force=True)
            logger.info(f"[INFERENCE] Removed old container: {old_container.name} in {time.time() - start:.3f} seconds")
        except docker.errors.NotFound:
            logger.info(f"[INFERENCE] No old container found. (Took {time.time() - start:.3f} seconds)")

        # inference 명령어 구성
        inference_command = [
            "bash",
            "-c",
            f"source model_env/bin/activate && python test.py "
            f"--project {request.project} "
            f"--subproject {request.subproject} "
            f"--task {request.task} "
            f"--version {request.version} "
        ]

        # 볼륨 설정
        volumes = {
            f"{VOLUME_PATH}": {
                "bind": "/moai",
                "mode": "rw"
            }
        }

        # 컨테이너 생성 전 타임스탬프
        start = time.time()
        try:
            container = client.containers.run(
                image=f"{model_type}:latest",
                name=container_name,
                volumes=volumes,
                device_requests=[
                    docker.types.DeviceRequest(
                        count=-1,
                        capabilities=[["gpu"]]
                    )
                ],
                tty=True,
                stdin_open=True,
                detach=True,
                # shm_size="32G",
            )
            elapsed = time.time() - start
            logger.info(f"Container {container_name} started successfully in {elapsed:.3f} seconds.")
        except Exception as e:
            logger.error(f"Failed to start container {container_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to start container: {str(e)}")

        # 전체 과정 시간 기록
        logger.info(f"Total time for container start process: {time.time() - overall_start:.3f} seconds.")

        # inference 실행 함수 (스레드 내 실행)
        def run_inference(container, inference_command):
            logger.info("[INFERENCE] YOLO container inference started...")
            start = time.time()
            exec_result = container.exec_run(inference_command, stream=True)
            for output in exec_result.output:
                logger.info(output.decode('utf-8', errors='replace'))
            logger.info(f"[INFERENCE] YOLO container inference finished in {time.time() - start:.3f} seconds.")

            container.kill()
            container.remove(force=True)

        # 별도 스레드에서 inference 실행
        inference_thread = threading.Thread(target=run_inference, args=(container, inference_command))
        inference_thread.daemon = True
        inference_thread.start()

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))

def export_model(request: ExportRequest):
    try:
        container_name = f"{request.project}_{request.subproject}_{request.task}_{request.version}_export"

        train_config_path = f"/moai/{request.project}/{request.subproject}/{request.task}/{request.version}/train_config.yaml"
        with open(train_config_path, "r") as f:
            train_config = yaml.safe_load(f)
        
        model_type = train_config["model_type"]

        try:
            old_container = client.containers.get(container_name)
            old_container.stop()
            old_container.remove(force=True)
            logger.info(f"[EXPORT] Removed old container: {old_container.name}")
        except docker.errors.NotFound:
            logger.info("[EXPORT] No old container found.")

        export_command = [
            "bash",
            "-c",
            f"source model_env/bin/activate && python export.py "
            f"--project={request.project} "
            f"--subproject={request.subproject} "
            f"--task={request.task} "
            f"--version={request.version} "
        ]

        volumes = {
            f"{VOLUME_PATH}": {  # 변경된 볼륨 경로
                "bind": "/moai",
                "mode": "rw"
            }
        }

        try:
            container = client.containers.run(
                image=f"{model_type}:latest",  # 이미지 이름 및 태그 지정
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
            logger.info(f"Container {container_name} started successfully.")
        except Exception as e:
            logger.error(f"Failed to start container {container_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to start container: {str(e)}")

        def run_export(container, export_command):
            """학습을 실제로 수행하는 함수 (별도 스레드에서 실행)"""
            logger.info(f"[EXPORT] container export started...")
            exec_result = container.exec_run(export_command, stream=True)
            for output in exec_result.output:
                logger.info(output.decode('utf-8', errors='replace'))
            logger.info("[EXPORT] container export finished...")

            container.kill()
            container.remove(force=True)

        # 예측을 별도의 스레드에서 실행
        export_thread = threading.Thread(target=run_export, args=(container, export_command))
        export_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
        export_thread.start()

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))