import os
import yaml

def load_yaml():
    running_yaml = os.path.join(os.getcwd(), "running.yaml")

    with open(running_yaml, "r") as f:
        running_params = yaml.safe_load(f)

    return running_params

def save_yaml(request, is_running, is_train):
    running_yaml = os.path.join(os.getcwd(), "running.yaml")

    running_params = load_yaml()

    with open(running_yaml, "w") as f:
        running_params['is_running'] = is_running
        running_params['task'] = "학습중" if is_train else "예측중"
        running_params['project'] = request.project
        running_params['subproject'] = request.subproject
        running_params['task'] = request.task
        running_params['version'] = request.version
        running_params['inference_name'] = request.inference_name if is_train == False else ""

        yaml.dump(running_params, f)

    