import docker

def get_running_container():
    client = docker.from_env()

    running_containers = client.containers.list(all=False)

    return running_containers