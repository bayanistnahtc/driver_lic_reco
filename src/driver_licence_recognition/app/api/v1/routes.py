import logging
import traceback

from fastapi import APIRouter, HTTPException, Depends
from prometheus_client import Counter, Histogram, Gauge
from ml_toolkit.pipelines.cv_detection import CVDetectionPipeline
from ml_toolkit.pipelines.cv_recognition import CVRecognitionPipeline
from ml_toolkit.loaders.image_loader import download_transpose, ImageLoadException
from typing import Dict

from app_settings import settings
from dependencies.pipelines import get_cv_detection_pipeline, get_ocr_pipelines

from schemes.v1.requests import DocumentRequest
from schemes.v1.responses import DriverLicenceResponse

from core.driver_licence_recognition import driver_licence_recognize as driver_licence_recognize_func


logger = logging.getLogger("uvicorn")
router = APIRouter()
config = settings.config

DOCUMENT_FIELDS_RECOGNITION_PIPELINE_TIME = Histogram(
    "document_fields_recognition_pipeline_time",
    "Time spent in document fields recognition pipeline",
)
EXCEPTION_COUNT = Counter("exception_count", "Total number of exceptions")
LOAD_BY_SIZE_TIME = Histogram("load_by_size_time", "Time spent loading image by size")
EXCEPTION_IMAGE_DOWNLOAD_COUNT = Counter(
    "exception_image_download_count", "Total number of image exceptions"
)

@router.post(
    "/recognize/driver_licence", 
    description="Распознавание полей паспорта.", 
    response_model=DriverLicenceResponse
)
async def driver_licence_recognize(
    request: DocumentRequest,
    detection_pipeline: CVDetectionPipeline = Depends(get_cv_detection_pipeline),
    ocr_pipelines: Dict[str, CVRecognitionPipeline] = Depends(get_ocr_pipelines)
) -> DriverLicenceResponse:
    image_token = config['image_download']['token']
    try:
        logger.info(f"Driver Licence recognition request: {request.request_id}")

        with LOAD_BY_SIZE_TIME.time():
            url = config['image_download']['url'].format(request.guid, image_token)
            timeout_sec = config['image_download']['timeout_sec']
            image = download_transpose(url, timeout_download_s=timeout_sec)

        
        logger.info(f"Image loaded: {request.request_id}")
        logger.info(f"Recognizing photo: {request.request_id}")
        with DOCUMENT_FIELDS_RECOGNITION_PIPELINE_TIME.time():
            results = driver_licence_recognize_func(image, config['pipeline_config'], detection_pipeline, ocr_pipelines)
        dl_fields = []

        for result in results:
            dl_fields.append(result.to_dict())

        logger.info(
            f"Image has been recognized. request_id: {request.request_id}"
        )


        response = DriverLicenceResponse(id=request.request_id, 
                                   is_driver_licence_found=len(dl_fields) > 0,
                                   fields_recognition_result=dl_fields)
        return response

    except ImageLoadException as e:
        logger.error(traceback.format_exc())
        EXCEPTION_IMAGE_DOWNLOAD_COUNT.inc()
        raise HTTPException(status_code=400, detail=e.message)
    except Exception as e:
        logger.error(traceback.format_exc())
        EXCEPTION_COUNT.inc()
        raise HTTPException(status_code=400, detail=str(e))
