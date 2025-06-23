import triton_python_backend_utils as pb_utils

from recognition.postprocess import postprocess
from recognition.preprocess import preprocess
from recognition.validation import validation_recognition


def infer_model(detection_result, field_name, field_bbox, config, log_msg=''):
    """Исполнение моделей детекции

    Parameters
    ----------
    detection_result : _type_
        Результат детекции
    field_name : _type_
        Наименование поля
    field_bbox : _type_
        Координаты рамки поля
    config : _type_
        Конфик сервиса
    log_msg : str, optional
        Шаблон строки лога, by default ''

    Returns
    -------
    ResultRecognition
        Результат распознавания поля

    Raises
    ------
    pb_utils.TritonModelException
        Ошибка исполнения модели распознавания
    """
    log_msg = f'{log_msg} module: recognition; field_name: {field_name};'
    model_name = config['recognized_fields'][field_name]
    model = config[model_name]
    # Определяем какой функцией будем валидировать результат распознавания
    pb_utils.Logger.log_info(f'{log_msg} get validation func')
    validation = validation_recognition(field_name)
    pb_utils.Logger.log_info(f'{log_msg} preprocess')
    crop_tensor = preprocess(detection_result.img,
                             field_bbox, (model['image_height'], model['image_width']))

    text_recognition_request = pb_utils.InferenceRequest(
        model_name=model['model_name'],
        requested_output_names=["output"],
        inputs=[pb_utils.Tensor("input", crop_tensor)],
    )
    pb_utils.Logger.log_info(f'{log_msg} send request to model {model_name}')
    text_recognition_response = text_recognition_request.exec()
    if text_recognition_response.has_error():
        raise pb_utils.TritonModelException(
            text_recognition_response.error().message())

    prediction = pb_utils.get_output_tensor_by_name(
        text_recognition_response, 'output').as_numpy()
    pb_utils.Logger.log_info(f'{log_msg} postprocess')
    result = postprocess(prediction, model['vocabulary'], model['threshold'], validation)
    pb_utils.Logger.log_info(f'{log_msg} result: text: {result.predict_word} correct: {result.is_correct} min_score: {result.word_score}')
    return result
