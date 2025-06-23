import numpy as np

from ml_toolkit.pipelines.cv_recognition import CVRecognitionPipeline
from ml_toolkit.pipelines.cv_detection import CVDetectionPipeline
from ml_toolkit.pipelines.document_recognition import document_fields_recognition
from ml_toolkit.pipelines.document_recognition import ResultFieldRecognition
from typing import Dict


def driver_licence_recognize(
    image: np.ndarray,
    config: dict,
    field_detector: CVDetectionPipeline,
    text_recognizers: Dict[str, CVRecognitionPipeline]
) -> ResultFieldRecognition:
    results = document_fields_recognition(
                image, field_detector, None, text_recognizers, config
            )
    return results