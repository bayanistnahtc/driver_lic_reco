import triton_python_backend_utils as pb_utils

from driver_license_classes import DriverLicenseClass
from detection.complete import complete_predictions
from detection.postprocess import scale_bboxes, select_bboxes
from detection.preprocess import preprocess
from detection.rotate import rotate_doc_with_bboxes




class ResultDetection:
    def __init__(self, img, is_correct, is_front_side, predictions, angle):
        """Результат детекции

        Parameters
        ----------
        img : numpy
            Фото паспорта в виде numpy
        predictions : list(ResultFieldDetection)
            Список результатов детекции
        angle : int
            Угол поворота изображения
        is_correct : bool
            Флаг корректности распознавания
        """

        self.img = img
        self.is_correct = is_correct
        self.is_front_side = is_front_side
        self.predictions = predictions
        self.angle = angle



def postprocess(response_dict, img, scale, dw, dh, config):
    """Постобработка результатов модели и упаковка результата

    Parameters
    ----------
    response_dict : dict
        _description_
    img : numpy.array
        Фото паспорта
    scale : float
        коэфф изменения размера (см resize_with_pad)
    dw : int
        дельта цирины
    dh : int
        дельта высоты
    config : dict
        Конфиг сервиса

    Returns
    -------
    tuple
        фото паспорта в правильной ориентации, результаты детекции list(ResultFieldDetection),
        угол поворота фото, флаг корректности детекции
    """
    bboxes, scores, classes = select_bboxes(
        response_dict["detection_boxes"], response_dict["detection_scores"], response_dict["detection_classes"])
    scaled_bboxes = scale_bboxes(
        img, bboxes, scale, dw, dh, detector_img_size=config['detector']['img_size'])

    # rotate img and bboxes
    img, rotated_bboxes, angle = rotate_doc_with_bboxes(
        img, scaled_bboxes, classes)

    # check detection predict
    is_correct, is_front_side = check_detection(classes, scores, config["detector"]['threshold'], config["detector"]['check_front_fields'],
                                                config["detector"]['check_back_fields'])

    # complete predictions to one list
    predictions = complete_predictions(rotated_bboxes, classes, scores)

    return img, predictions, angle, is_correct, is_front_side


def infer_model(img, config, log_msg=""):
    """Функция исполнения модели детекции

    Parameters
    ----------
    img : numpy.array
        Тензор фото паспорта
    config : dict
        Конфиг сервиса
    log_msg : str, optional
        Шаблон строки лога, by default ''

    Returns
    -------
    ResultDetection
        Результат детекции

    Raises
    ------
    pb_utils.TritonModelException
        Ошибка исполнения модели
    """
    log_msg = f"{log_msg} module: detection;"

    # preprocessing
    pb_utils.Logger.log_info(f"{log_msg} preprocess")
    img_tensor, scale, (dw, dh) = preprocess(img, config)

    # infer detect model
    detection_request = pb_utils.InferenceRequest(
        model_name="tf_ru_driver_license_detection",
        requested_output_names=[
            "detection_boxes",
            "detection_classes",
            "detection_scores",
        ],
        inputs=[pb_utils.Tensor("inputs", img_tensor)],
    )
    pb_utils.Logger.log_info(f"{log_msg} send request to model")
    detection_response = detection_request.exec()
    if detection_response.has_error():
        raise pb_utils.TritonModelException(
            detection_response.error().message())

    response_dict = {}
    for out_name in ["detection_boxes", "detection_classes", "detection_scores"]:
        response_dict[out_name] = pb_utils.get_output_tensor_by_name(
            detection_response, out_name
        ).as_numpy()

    # postprocessing
    pb_utils.Logger.log_info(f"{log_msg} postprocess")
    img, predictions, angle, is_correct, is_front_side = postprocess(
        response_dict, img, scale, dw, dh, config
    )
    return ResultDetection(img, is_correct, is_front_side, predictions, angle)



def check_detection(classes, scores, fields_threshold, check_front_fields, check_back_fields):
    """Проверка корректности детекции

    Parameters
    ----------
    classes : list(str)
        Список имен классов
    scores : dict(str, float)
        Словарь скоров детекции
    fields_threshold : float
        Порог скора детекции
    check_fields : list(str)
        Список полей которые нужно проверять

    Returns
    -------
    bool
        Флаг корректности детекции
    """

    is_front_side = True
    for field_name in check_front_fields:
        field_id = DriverLicenseClass[field_name].value
        if field_id in classes:
            field_index = classes.index(field_id)
            field_score = scores[field_index]
            if field_score < fields_threshold:
                is_front_side = False
        else:
            is_front_side = False

    is_back_side = True
    for field_name in check_back_fields:
        field_id = DriverLicenseClass[field_name].value
        if field_id in classes:
            field_index = classes.index(field_id)
            field_score = scores[field_index]
            if field_score < fields_threshold:
                is_back_side = False
        else:
            is_back_side = False

    is_correct = is_front_side or is_back_side

    return is_correct, is_front_side
