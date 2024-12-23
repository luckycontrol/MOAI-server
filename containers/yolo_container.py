import docker
from models.train import TrainRequest

client = docker.from_env()

async def inference_yolo(request: InferenceRequest):                                                           
    try:                                                                                                       
        command = [                                                                                            
            "python",                                                                                          
            "detect.py",                                                                                       
            f"--project={request.project}",                                                                    
            f"--subproject={request.subproject}",                                                              
            f"--task={request.task}",                                                                          
            f"--version={request.version}",                                                                    
            f"--weights={request.weights}",                                                                    
            f"--imgsz={request.imgsz}",                                                                        
            f"--conf-thres={request.conf_thres}",                                                              
            f"--iou-thres={request.iou_thres}",                                                                
            f"--source={request.source}",                                                                      
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

async def train_yolo(request: TrainRequest):
    try:
        command = [
            "python",
            "train.py",
            f"--project={request.project}",
            f"--subproject={request.subproject}",
            f"--task={request.task}",
            f"--version={request.version}",
            f"--imgsz={request.train_params.imgsz}",
            f"--batch-size={request.train_params.batch_size}",
            f"--weight-type=yolov5{request.train_params.weight_type}",
            f"--epochs={request.train_params.epoch}",
            f"--patience={request.train_params.patience}",
            "--resume" if request.train_params.resume else None,
            f"--lr={request.hyps.lr}",
            f"--hsv={request.hyps.hsv}",
            f"--degrees={request.hyps.degrees}",
            f"--translate={request.hyps.translate}",
            f"--scale={request.hyps.scale}",
            "--flipud" if request.hyps.flipud else None,
            "--fliplr" if request.hyps.fliplr else None,
            "--mosaic" if request.hyps.mosaic else None,
        ]
    
        container = client.containers.get("")
        exec_result = container.exec_run(command)
        return {
            "status": "success",
            "message": "학습 완료",
            "data": exec_result.output.decode("utf-8")
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )