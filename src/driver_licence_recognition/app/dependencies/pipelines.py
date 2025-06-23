from ml_toolkit.cv.recognition.crnn_processor import CRNNProcessor
from ml_toolkit.cv.detection.yolov8 import YOLOV8Processor
from ml_toolkit.pipelines.cv_detection import CVDetectionPipeline
from ml_toolkit.pipelines.cv_recognition import CVRecognitionPipeline
from typing import Dict

from app_settings import settings

config = settings.config

img_size = config.get('ru_driver_licence_models').get('detector').get('img_size', 1280)
detection_model_name = config.get('ru_driver_licence_models').get('detector').get('model_name')

image_processor = YOLOV8Processor(
    image_height=img_size,
    image_width=img_size,
)
config['triton_config']['model_name'] = config.get('ru_driver_licence_models').get('detector').get('model_name', '')
detection_pipeline = CVDetectionPipeline(config, processor=image_processor)

def get_ocr_pipeline(model: str, config: dict):
    config['triton_config']['model_name'] = config.get('ru_driver_licence_models').get(model).get('model_name', '')
    image_height = config.get('ru_driver_licence_models').get(model).get('image_height', 32)
    image_width = config.get('ru_driver_licence_models').get(model).get('image_width', 120)
    vocab = config.get('ru_driver_licence_models').get(model).get('vocabulary', '')
    use_space_char = config.get('ru_driver_licence_models').get(model).get('use_space_char', True)
    processor = CRNNProcessor(
        image_height=image_height,
        image_width=image_width,
        vocab=vocab,
        use_space_char=use_space_char
    )

    pipeline = CVRecognitionPipeline(config=config, processor=processor)
    return pipeline

fio_pipeline = get_ocr_pipeline('fio', config)
date_pipeline = get_ocr_pipeline('date', config)
serial_pipeline = get_ocr_pipeline('serial', config)

ocr_pipelines = {'fio': fio_pipeline, 'date': date_pipeline, 'serial': serial_pipeline}

def get_cv_detection_pipeline() -> CVDetectionPipeline:
    return detection_pipeline

def get_ocr_pipelines() -> Dict[str, CVRecognitionPipeline]:
    return ocr_pipelines