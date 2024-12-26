import docker

def is_container_running():
    client = docker.from_env()

    running_container = client.containers.list(all=False)

    return len(running_container) > 0