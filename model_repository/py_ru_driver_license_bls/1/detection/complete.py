from driver_license_classes import DriverLicenseClass


class ResultFieldDetection:
    def __init__(self, field_name, bbox, score):
        """Результат детекции поля

        Parameters
        ----------
        field_name : str
            Наименование поля
        bbox : list(int)
            Координаты рамки
        score : float
            Скор детекции
        """

        self.field_name = field_name
        self.bbox = bbox
        self.score = score


def complete_predictions(bboxes, classes, scores):
    """Упаковка резултатов детекции

    Parameters
    ----------
    bboxes : list(list(int))
        Координаты рамок
    classes : list(int)
        Список id классов
    scores : list(float)
        Скоры детекции

    Returns
    -------
    _type_
        _description_
    """

    predictions = []

    for index in range(len(bboxes)):
        class_id = classes[index]
        field_name = DriverLicenseClass(class_id).name

        field_detection_result = ResultFieldDetection(field_name=field_name, bbox=bboxes[index], score=scores[index])

        predictions.append(field_detection_result)

    return predictions
