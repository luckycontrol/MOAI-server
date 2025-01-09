from fastapi import APIRouter, HTTPException
from typing import Dict

from models.export import ExportRequest
from containers.model_container import export_model
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/export")
def export(request: ExportRequest) -> Dict:
    try:
        logger.info(f"[Export] Export 요청 수신: {request}")

        export_model(request)

        return {
            "status": "success",
            "message": "모델 내보내기 진행 중"
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )