from typing import List, Optional, Any

from pydantic import BaseModel


class ResultFieldRecognition(BaseModel):
    """
    Результат распознавания поля документа
    
    Attributes
    ----------
    field_name : str
        Имя поля
    field_bbox : list, optional
        Координаты ограничивающей рамки
    field_detect_score : float, optional
        Скор детекции
    field_text : str, optional
        Текст поля
    field_text_score : float, optional
        Минимальный скор распознавания букв
    field_symbol_scores : list, optional
        Скоры распознавания букв
    """
    
    field_name: str
    field_bbox: Optional[List] = []
    field_detect_score: float = 0.0
    field_text: Optional[str] = ""
    field_text_score: float = 0.0
    field_symbol_scores: Optional[List] = []

class DriverLicenceResponse(BaseModel):
    id: Any
    is_driver_licence_found: bool
    fields_recognition_result: List[ResultFieldRecognition]


