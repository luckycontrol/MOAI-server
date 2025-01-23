import docker
from fastapi import HTTPException
from models.tensorboard import TensorboardParams
from utils import VOLUME_PATH
import logging
import requests
import time

logger = logging.getLogger(__name__)
client = docker.from_env()

def create_tensorboard_container(tensorboard_params: TensorboardParams):
    try:
        # [1] 우선 prefix (project_subproject_task_version) 구성
        new_prefix = f"{tensorboard_params.project}_{tensorboard_params.subproject}_{tensorboard_params.task}_{tensorboard_params.version}"
        container_name = f"{new_prefix}_tensorboard"

        # [2] 동일 접두어로 만들어진 컨테이너가 이미 존재하는지(실행 중 여부와 무관) 확인
        all_containers = client.containers.list(all=True)
        for c in all_containers:
            # c.name == "project_subproject_task_version_tensorboard" 형태
            if c.name == container_name:
                # 이미 동일 이름의 컨테이너가 존재
                if c.status == "running":
                    raise HTTPException(
                        status_code=400,
                        detail=f"이미 활성화된 TensorBoard 컨테이너가 존재합니다: {c.name}"
                    )
                else:
                    # (운영 정책에 따라) 중지 상태라도 이름 충돌이므로 제거
                    try:
                        c.remove(force=True)
                        logger.info(f"기존 중지된 컨테이너 {c.name} 을(를) 제거했습니다.")
                    except Exception as remove_ex:
                        logger.error(f"기존 컨테이너 제거 실패: {remove_ex}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"기존 컨테이너 제거 중 오류가 발생했습니다: {remove_ex}"
                        )
        
        # [3] 볼륨 설정
        volumes = {
            VOLUME_PATH: {
                "bind": "/moai",
                "mode": "rw"
            }
        }

        # [4] 50000 ~ 51000 범위를 순회하면서 사용 가능한 포트를 찾아 컨테이너 실행
        selected_port = None
        for port in range(50000, 51001):
            try:
                ports_mapping = {f"{port}/tcp": port}

                run_tensorboard_command = [
                    "conda",
                    "run",
                    "-n",
                    "tensorboard",
                    "tensorboard",
                    f"--logdir=/moai/{tensorboard_params.project}/{tensorboard_params.subproject}/{tensorboard_params.task}/{tensorboard_params.version}/training_result",
                    "--port",
                    str(port),
                    "--bind_all",
                ]

                container = client.containers.run(
                    image="moai_tensorboard:latest",
                    command=run_tensorboard_command,
                    name=container_name,
                    volumes=volumes,
                    ports=ports_mapping,
                    detach=True,
                    tty=True,
                    stdin_open=True
                )
                logger.info(f"Created new container: {container_name} on port {port}")
                selected_port = port
                break

            except docker.errors.APIError as e:
                # 'port is already allocated' 오류나 '이름 충돌(Conflict)' 등은 다음 포트로 넘어간다
                err_str = str(e)
                if ("port is already allocated" in err_str) or ("Conflict" in err_str):
                    logger.warning(f"Port {port} 사용 불가({err_str}), 다음 포트를 시도합니다.")
                    continue
                else:
                    logger.error(f"예상치 못한 Docker API 오류: {e}")
                    raise HTTPException(status_code=500, detail=str(e))
        else:
            # for-else 구문: break가 한 번도 걸리지 않으면 else 블럭 실행
            logger.error("Failed to find an available port in range 50000-51000")
            raise HTTPException(
                status_code=500,
                detail="50000~51000 범위 내 사용 가능한 포트를 찾을 수 없습니다."
            )

        # [5] TensorBoard 페이지가 실제로 준비되었는지 검증
        max_retries = 60
        for attempt in range(max_retries):
            try:

                # Use the Docker host IP instead of localhost
                response = requests.get(f"http://192.168.100.40:{selected_port}")

                if response.status_code == 200 and "TensorBoard" in response.text:
                    logger.info(
                        f"TensorBoard UI is successfully loaded on port {selected_port}"
                    )
                    return {
                        "message": f"TensorBoard '{container_name}' 컨테이너가 성공적으로 생성되었습니다.",
                        "port": selected_port
                    }
            except requests.exceptions.RequestException as e:
                # 아직 뜨지 않았을 수 있으니 재시도
                logger.debug(f"TensorBoard 웹 UI 확인 재시도({attempt+1}/{max_retries}): {e}")

            time.sleep(1)

        # 충분히 재시도했음에도 UI 확인이 안 되면 예외 처리
        raise HTTPException(
            status_code=400,
            detail="여러 번의 시도 후에도 TensorBoard UI가 준비되지 않았습니다."
        )

    except Exception as e:
        logger.error(f"create_tensorboard_container 실패: {e}")
        # (선택) 여기서도 혹시 중간에 생성된 컨테이너가 있다면 정리하는 로직을 추가할 수 있다.
        raise HTTPException(status_code=400, detail=str(e))


def stop_tensorboard_container(tensorboard_params: TensorboardParams):
    """
    project/subproject/task/version 조합으로 생성된 tensorboard 컨테이너를 찾아서 종료하고 제거한다.
    """
    try:
        prefix = f"{tensorboard_params.project}_{tensorboard_params.subproject}_{tensorboard_params.task}_{tensorboard_params.version}"
        container_name = f"{prefix}_tensorboard"

        # 이름이 정확히 일치하는 컨테이너 검색(실행 중 여부와 무관)
        containers = client.containers.list(all=True)
        target_container = None
        for c in containers:
            if c.name == container_name:
                target_container = c
                break

        if target_container is None:
            raise HTTPException(
                status_code=404,
                detail=f"해당 조합({prefix})으로 생성된 TensorBoard 컨테이너를 찾을 수 없습니다."
            )

        # 실행 중이면 중지 후 제거
        if target_container.status == "running":
            target_container.kill()
            logger.info(f"Stopped container: {target_container.name}")

        target_container.remove(force=True)
        logger.info(f"Removed container: {target_container.name}")

        return {
            "message": f"TensorBoard 컨테이너 '{container_name}'를 종료 및 제거했습니다."
        }

    except Exception as e:
        logger.error(f"stop_tensorboard_container 실패: {e}")
        raise HTTPException(status_code=400, detail=str(e))
