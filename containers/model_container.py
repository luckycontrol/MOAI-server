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
            f"source {request.model_type}/bin/activate && python train.py "
            f"--project {request.project} "
            f"--subproject {request.subproject} "
            f"--task {request.task} "
            f"--version {request.version} "
        ]

        # 현재 학습의 hyp 정보 로드
        hyp_path = f"{VOLUME_PATH}/{request.project}/{request.subproject}/{request.task}/dataset/train_dataset/hyp.yaml"
        with open(hyp_path, 'r') as f:
            hyp = yaml.safe_load(f)

        # 현재 학습에 대한 정보를 yaml 로 저장
        yaml_path = f"{VOLUME_PATH}/{request.project}/{request.subproject}/{request.task}/{request.version}/train_config.yaml"

        yaml_content = {
            "project": request.project,
            "subproject": request.subproject,
            "task": request.task,
            "version": request.version,
            "model_type": request.model_type,
            "imgsz": hyp['imgsz'],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)

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
            logger.info("[TRAINING] YOLO container training started...")
            exec_result = container.exec_run(train_command, stream=True)
            # 학습이 종료될 때 까지 반복문이 돌아야 함
            for output in exec_result.output:
                continue
                # logger.info(output.decode('utf-8', errors='replace'))
            logger.info("[TRAINING] YOLO container training finished...")

            container.stop()
            container.remove(force=True)

        # 학습을 별도의 스레드에서 실행
        training_thread = threading.Thread(target=run_training, args=(container, train_command))
        training_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
        training_thread.start()

        start_time = time.time()
        timeout = 120  # 2분 타임아웃

        # results.csv 파일 찾을 때까지 대기
        while True:
            # results.csv 파일을 찾는 중, 에러로 인해 모델 컨테이너가 종료되면 예외raise
            container = client.containers.get(container_name)
            if container.status != "running":
                raise HTTPException(
                    status_code=400,
                    detail=f"[Training] 학습 실패"
                )

            file_path = f"{VOLUME_PATH}/{request.project}/{request.subproject}/{request.task}/{request.version}/training_result/results.csv"
            logger.info(f"Checking file path: {file_path}")
            logger.info(f"File exists: {os.path.exists(file_path)}")
            
            if os.path.exists(file_path):
                logger.info("results.csv file has been found!")
                break
            
            if time.time() - start_time > timeout:
                raise HTTPException(status_code=408, detail="Training timeout: results.csv was not generated within 2 minutes")
            
            time.sleep(5)  # 5초마다 확인

    except Exception as e:
        logger.info(f"Training failed: {str(e)}")
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
            "bash",
            "-c",
            f"source {request.model_type}/bin/activate && python test.py "
            f"--project {request.project} "
            f"--subproject {request.subproject} "
            f"--task {request.task} "
            f"--version {request.version} "
        ]

        volumes = {
            f"{VOLUME_PATH}": {  # 변경된 볼륨 경로
                "bind": "/moai",
                "mode": "rw"
            }
        }

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

        def run_inference(container, inference_command):
            """학습을 실제로 수행하는 함수 (별도 스레드에서 실행)"""
            logger.info("[INFERENCE] YOLO container inference started...")
            exec_result = container.exec_run(inference_command, stream=True)
            for output in exec_result.output:
                continue
            logger.info("[INFERENCE] YOLO container inference finished...")

            container.stop()
            container.remove(force=True)

        # 예측을 별도의 스레드에서 실행
        inference_thread = threading.Thread(target=run_inference, args=(container, inference_command))
        inference_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
        inference_thread.start()

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))

def export_model(request: ExportRequest):
    try:
        container_name = f"{request.project}_{request.subproject}_{request.task}_{request.version}_export"

        yaml_path = f"{VOLUME_PATH}/{request.project}/{request.subproject}/{request.task}/{request.version}/train_config.yaml"
        with open(yaml_path, 'r') as f:
            yaml_content = yaml.safe_load(f)

        model_type = yaml_content["model_type"]
        imgsz = yaml_content['imgsz']

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
            f"source {model_type}/bin/activate && python export.py "
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
            logger.info("[EXPORT] YOLO container export started...")
            exec_result = container.exec_run(export_command, stream=True)
            for output in exec_result.output:
                logger.info(output.decode('utf-8', errors='replace'))
            logger.info("[EXPORT] YOLO container export finished...")

            container.stop()
            container.remove(force=True)

        # 예측을 별도의 스레드에서 실행
        export_thread = threading.Thread(target=run_export, args=(container, export_command))
        export_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
        export_thread.start()

    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=400, detail=str(e))