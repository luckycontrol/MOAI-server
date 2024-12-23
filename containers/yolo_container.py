import docker
from models.train import TrainRequest

client = docker.from_env()  

async def train_yolo(request: TrainRequest):
    try:
        command = [
            "conda",
            "run",
            "-n",
            "yolo",
            "python",
            "train.py",
            f"--project=/moai/{request.project}/{request.subproject}/{request.task}",
            f"--name={request.version}",
            f"--imgsz={request.train_params.imgsz}",
            f"--batch-size={request.train_params.batch_size}",
            f"--weights=yolov5{request.train_params.weight_type}",
            f"--epochs={request.train_params.epoch}",
            f"--patience={request.train_params.patience}",
            f"--lr={request.hyps.lr}",
            f"--hsv={request.hyps.hsv}",
            f"--degrees={request.hyps.degrees}",
            f"--translate={request.hyps.translate}",
            f"--scale={request.hyps.scale}",
        ]

        if request.train_params.resume:
            command.append("--resume")
        
        if request.hyps.flipud:
            command.append("--flipud")
        
        if request.hyps.fliplr:
            command.append("--fliplr")
        
        if request.hyps.mosaic:
            command.append("--mosaic")
    
        container = client.containers.get("moai_yolo")
        exec_result = container.exec_run(command)
        return {
            "status": "success",
            "message": "학습 완료"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

async def inference_yolo(request: InferenceRequest):                                                           
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
                                                                                                            
        container = client.containers.get("")                                                                  
        exec_result = container.exec_run(command)                                                              
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