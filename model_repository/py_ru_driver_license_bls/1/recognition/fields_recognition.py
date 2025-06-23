class ResultFieldRecognition:
    def __init__(self, field_name, is_detection=False, field_bbox=[], field_detect_score=0.0, is_ocr=False,
                 field_text='', field_text_score=0.0, field_symbol_scores=[]):
        self.field_name = field_name
        self.is_detection = is_detection
        self.field_bbox = field_bbox
        self.field_detect_score = field_detect_score
        self.is_ocr = is_ocr
        self.field_text = field_text
        self.field_text_score = field_text_score  # min of symbol_scores
        self.field_symbol_scores = field_symbol_scores

    def to_dict(self):
        return {
            "field_name": self.field_name,
            "is_detection": self.is_detection,
            "field_bbox": self.field_bbox,
            "field_detect_score": self.field_detect_score,
            "is_ocr": self.is_ocr,
            "field_text": self.field_text,
            "field_text_score": self.field_text_score,
            "field_symbol_scores": self.field_symbol_scores
        }
