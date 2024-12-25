import docker
from models.train import TrainRequest
from fastapi import HTTPException
from models.inference import InferenceRequest
import time
import threading

client = docker.from_env()

def train_yolo(request: TrainRequest):
    try:
        try:
            old_container = client.containers.get("moai_yolo")
            old_container.stop()
            old_container.remove(force=True)
            print(f"Removed old container: {old_container.name}")
        except docker.errors.NotFound:
            print("No old container found.")

        volumes = {
            r"E:\moai_test": {  # 변경된 볼륨 경로
                "bind": "/moai",
                "mode": "rw"
            }
        }

        train_command = [
            "conda",
            "run",
            "-n",
            "yolo",
            "python",
            "train.py",
            f"--project=/moai/{request.project}/{request.subproject}/{request.task}",
            f"--name={request.version}",
            f"--data=/moai/{request.project}/{request.subproject}/{request.task}/dataset/train_dataset/data.yaml",
            f"--imgsz={request.train_params.imgsz}",
            f"--batch-size={request.train_params.batch_size}",
            f"--weights=/MOAI_yolo/weights/yolov5{request.train_params.weight_type}.pt",
            f"--epochs={request.train_params.epoch}",
            f"--patience={request.train_params.patience}",
            f"--hyp=/moai/{request.project}/{request.subproject}/{request.task}/dataset/train_dataset/hyp.yaml",
        ]

        container = client.containers.run(
            image="moai_yolo:latest",  # 이미지 이름 및 태그 지정
            name="moai_yolo",
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
            print("YOLO container training started...")
            exec_result = container.exec_run(train_command, stream=True)
            for output in exec_result.output:
                print(output.decode('utf-8', errors='replace'), end='')

            return {
                "status": "success",
                "message": "학습 완료"
            }

        # 학습을 별도의 스레드에서 실행
        training_thread = threading.Thread(target=run_training, args=(container, train_command))
        training_thread.daemon = True  # 메인 스레드 종료 시 함께 종료
        training_thread.start()

        return {
            "status": "in_progress",
            "message": "학습 진행 중",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def inference_yolo(request: InferenceRequest):
    try:
        command = [
            "python",
            "detect.py",
            f"--project=/moai/{request.project}/{request.subproject}/{request.task}",
            f"--name={request.inference_name}",
            f"--weights=/moai/{request.project}/{request.subproject}/{request.task}/{request.version}/weights/best.pt",
            f"--imgsz={request.imgsz}",
            f"--conf-thres={request.conf_thres}",
            f"--source=/moai/{request.project}/{request.subproject}/{request.task}/dataset/inference_dataset/{request.source}",
        ]

        container = client.containers.get("moai_yolo")
        exec_result = container.exec_run(command, stream=True) # stream=True를 추가하여 실시간 로그 출력

        for output in exec_result.output:
            print(output.decode('utf-8'))
        
        return {
            "status": "success",
            "message": "추론 완료",
            "data": exec_result.output.decode("utf-8")
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )