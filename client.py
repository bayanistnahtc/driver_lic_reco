import json

import requests


def check_models(triton_host='localhost',
                 triton_http_port='8000'):
    models_name = ['py_ru_driver_license_bls',
    'tf_ru_driver_license_recognition_date',
    'tf_ru_driver_license_recognition_fio',
    'tf_ru_driver_license_recognition_serial',
    'tf_ru_driver_license_detection']

    model_status_response = requests.post(
        f'http://{triton_host}:{triton_http_port}/v2/repository/index', json.dumps({}))
    models_state = {x['name']: x['state']
                    for x in model_status_response.json() if x['name'] in models_name}
    for name in models_name:
        assert name in models_state
        assert models_state[name] == 'READY'


def recognize_driver_license(photo_guid: str,
                             request_id: str,
                             model_host='localhost',
                             model_http_port='8000',
                             model_name='py_ru_driver_license_bls') -> dict:
    """Запрос на распознавание водительского удостоверение к Triton Inference Server.
    API Triton расширяет шаблоны KServe, которые заточены на стандартизацию взаимодействия c моделями машинного обучения.
    API Triton предполагает пакетную обработку и все данные запросов/ответов представляются виде списков.
    Текущие модели распознавания не поддерживают пакетную обработку, поэтому данные должны быть представлены как список c одним элементов.
    На одно фото - один запрос.

    Parameters
    ----------
    photo_guid : str
        GUID по которому можно скачать фото с файлового сервера
    request_id : str
        Идентификатор запроса

    Returns
    -------
    dict
        Шаблон ответа
        200
            Response:
                {'id': 'request_id',
                'model_name': 'py_ru_driver_license_bls',
                'model_version': '1',
                'outputs': [
                    {
                        'name': 'recognition_response',
                        'datatype': 'BYTES',
                        'shape': [1],
                        'data': [!!! Строка с результатами распознавания, которую нужно десериализовать !!!]
                    }
                ]
                }

            Response['outputs'][0]['data'][0]:
                {
                    'is_driver_license_found': True,                              Флаг успешности детекции паспорта
                    'fields_recognition_result':                            Список распознанных полей паспорта
                        [
                            {
                                'field_name': 'surname',                    Название поля
                                'is_detection': True,                       Флаг успешности детекции
                                'field_bbox': [1524, 2170, 1964, 2292],     Координаты рамки распознавания поля
                                'field_detect_score': 0.9195603728294373,   Скор детекции
                                'is_ocr': True,                             Флаг успешности распознавания текста
                                'field_text': 'НАМСАРАЕВ',
                                'field_text_score': 0.8317731618881226,     Скор распознавания текста поля (минимальный скор символов)
                                'field_symbol_scores': [0.99716657, ...]    Список скоров распознавания символов
                            }, ...
                        ]
                }
        400
            Response:
                {'error': "Error message"}
    """

    model_url = f'http://{model_host}:{model_http_port}/v2/models/{model_name}/infer'

    input_json = {
        "name": "image_guid",
        "shape": [1],
        "datatype": 'BYTES',
        "data": [json.dumps({'guid': photo_guid})]
    }

    output_json = {
        "name": 'recognition_response',
        'data_type': 'STRING',
    }

    request_json = {
        "id": request_id,
        "inputs": [input_json],
        "outputs": [output_json]
    }

    recognize_response = requests.post(model_url, json.dumps(request_json))
    if recognize_response.status_code == 200:
        # Извлекаем нулевой и единственный элемент из ответа
        # Первый 0 - индекс переменной вывода
        # Второй 0 - индекс запроса
        response_data = json.loads(recognize_response.json()[
                                   'outputs'][0]['data'][0])
        response_data['request_id'] = recognize_response.json()['id']
        return response_data
    else:
        return recognize_response.json()
