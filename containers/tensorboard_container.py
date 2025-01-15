import docker
from fastapi import HTTPException
from models.tensorboard import TensorboardParams
from utils import VOLUME_PATH
import logging

logger = logging.getLogger(__name__)

client = docker.from_env()

def create_tensorboard_container(tensorboard_params: TensorboardParams):
    try:
        # 이미 실행 중인 컨테이너 중에서 이름이 '_tensorboard' 로 끝나는 것만 가져오기
        running_containers = client.containers.list(filters={"status": "running"})
        tb_containers = [c for c in running_containers if c.name.endswith("_tensorboard")]

        # 새로 만들 컨테이너 이름 prefix (project_subproject_task_version)
        new_prefix = f"{tensorboard_params.project}_{tensorboard_params.subproject}_{tensorboard_params.task}_{tensorboard_params.version}"

        # 이미 동일 prefix 로 실행 중인 tensorboard 컨테이너가 있는지 확인
        for c in tb_containers:
            # 예: c.name == "project_subproject_task_version_tensorboard"
            #     -> c.name.rsplit("_", 1)[0] == "project_subproject_task_version"
            existing_prefix = c.name.rsplit("_", 1)[0]
            if existing_prefix == new_prefix:
                # 이미 동일한 prefix 로 tensorboard 가 실행 중임
                raise HTTPException(
                    status_code=400,
                    detail=f"이미 활성화된 TensorBoard가 존재합니다: {c.name}"
                )

        # 포트 충돌 방지 등을 위해 필요하다면 여기에서 사용중인 포트를 검사하는 로직 추가
        all_containers = client.containers.list(all=True)
        used_ports = set()  # 이미 사용중인 포트 모음
        for cont in all_containers:
            # 컨테이너의 포트 정보 얻기 (예: {'6006/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '6006'}], '5000/tcp': None, ...})
            container_ports_info = cont.attrs["NetworkSettings"].get("Ports") or {}
            for port_protocol, mappings in container_ports_info.items():
                if mappings:
                    # 여러 바인딩이 있을 수 있으므로 각각 확인
                    for m in mappings:
                        host_port_str = m.get("HostPort")
                        if host_port_str:
                            used_ports.add(int(host_port_str))

        free_port = None
        for port_to_try in range(50000, 51000):
            if port_to_try not in used_ports:
                free_port = port_to_try
                break

        if free_port is None:
            # 50000 ~ 50999 포트가 모두 사용중이라면 에러
            raise HTTPException(
                status_code=400,
                detail="50000~50999 범위 내 사용 가능한 포트를 찾을 수 없습니다."
            )

        # 최종적으로 만들 컨테이너 이름
        container_name = f"{new_prefix}_tensorboard"

        # 볼륨 설정 (예시)
        volumes = {
            VOLUME_PATH: {
                "bind": "/moai",
                "mode": "rw"
            }
        }

        # TensorBoard 실행 명령어 (conda 환경 예시)
        run_tensorboard_command = [
            "conda",
            "run",
            "-n",
            "tensorboard",
            "tensorboard",
            f"--logdir=/moai/{tensorboard_params.project}/{tensorboard_params.subproject}/{tensorboard_params.task}/{tensorboard_params.version}/training_results",
            "--port",
            str(free_port),
            "--bind_all",
        ]

        # 필요하다면 ports 매핑 추가 (예: 6006 -> 6006)
        ports_mapping = {f"{free_port}": free_port}

        # 컨테이너 실행
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
        logger.info(f"Created new container: {container_name}")

        return {
            "message": f"TensorBoard '{container_name}' 컨테이너가 성공적으로 생성되었습니다.",
            "port": free_port
        }

    except HTTPException:
        # FastAPI용 예외는 그대로 raise
        raise
    except Exception as e:
        logger.error(str(e))
        # 내부적으로 컨테이너를 만들다 실패했다면 중간에 생성된 컨테이너가 있을 수 있으니 정리
        # (실제로 컨테이너 객체가 존재하는지 여부를 체크한 뒤 제거하는 로직을 넣어도 됨)
        raise HTTPException(status_code=400, detail=str(e))


def stop_tensorboard_container(tensorboard_params: TensorboardParams):
    """
    project/subproject/task/version 조합으로 생성된 tensorboard 컨테이너를 찾아서 종료하고 제거한다.
    컨테이너가 매핑하여 사용하던 포트(host port)도 반환한다.
    """
    try:
        prefix = f"{tensorboard_params.project}_{tensorboard_params.subproject}_{tensorboard_params.task}_{tensorboard_params.version}"

        # "_tensorboard" 로 끝나는 모든 컨테이너 중 prefix 가 일치하는 컨테이너 찾기
        containers = client.containers.list(all=True)
        tb_containers = [c for c in containers if c.name.endswith("_tensorboard")]

        target_container = None
        for c in tb_containers:
            existing_prefix = c.name.rsplit("_", 1)[0]
            if existing_prefix == prefix:
                target_container = c
                break

        # 대상 컨테이너가 없다면 예외
        if target_container is None:
            raise HTTPException(
                status_code=404,
                detail=f"해당 조합({prefix})으로 실행 중인 TensorBoard 컨테이너를 찾을 수 없습니다."
            )

        # 컨테이너가 사용하고 있던 호스트 포트 추출
        # 예: {'6006/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '6006'}]}
        ports_info = target_container.attrs.get('NetworkSettings', {}).get('Ports', {})
        free_port = None
        for container_port, host_mappings in ports_info.items():
            if host_mappings and len(host_mappings) > 0:
                free_port = host_mappings[0].get('HostPort')
                break

        # 컨테이너가 실행 중이면 중지
        if target_container.status == "running":
            target_container.stop()
            logger.info(f"Stopped container: {target_container.name}")

        # 컨테이너 제거
        target_container.remove(force=True)
        logger.info(f"Removed container: {target_container.name}")

        return {
            "message": f"TensorBoard 컨테이너 {prefix}_tensorboard 를 종료 및 제거했습니다.",
            "port": free_port
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=400, detail=str(e))

