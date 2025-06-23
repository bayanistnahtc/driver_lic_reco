import json
import requests
import pandas as pd

from .preprocess import prepare_img
from .postprocess import (
    fio_indices_to_text,
    date_indices_to_text,
    serial_indices_to_text,
    decode,
)
from typing import Dict, List
from tqdm import tqdm
from .load_models import *

# Loading models
device = "cuda:0" if torch.cuda.is_available() else "cpu"

fio_path = "/home/jovyan/ocr_pytorch/models/driver_license_fio_recognition/last.pt"
fio_model = load_dl_fio_model(fio_path, device)

date_path = "../models/driver_license_date_recognition/best.pt"
date_model = load_dl_date_model(date_path, device)

serial_path = "../models/driver_license_serial_recognition/best.pt"
serial_model = load_dl_serial_model(serial_path, device)


def recognize_driver_license(
    photo_guid: str,
    request_id: str,
    model_host="triton.k8s-dev.taximaxim.com",
    model_http_port="8000",
    model_name="py_ru_driver_license_bls",
) -> dict:
    """Запрос на распознавание паспорта к Triton Inference Server.
    API Triton расширяет шаблоны KServe, которые заточены на стандартизацию взаимодействия с моделями машинного обучения.
    API Triton предполагает пакетную обработку и все данные запросов/ответов представляются виде списков.
    Текущие модели распознавания не поддерживают пакетную обработку, поэтому данные должны быть представлены как список с одним элементов.
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

    model_url = f"http://{model_host}:{model_http_port}/v2/models/{model_name}/infer"

    input_json = {
        "name": "image_guid",
        "shape": [1],
        "datatype": "BYTES",
        "data": [json.dumps({"guid": photo_guid})],
    }

    output_json = {
        "name": "recognition_response",
        "data_type": "STRING",
    }

    request_json = {"id": request_id, "inputs": [input_json], "outputs": [output_json]}

    recognize_response = requests.post(model_url, json.dumps(request_json))
    if recognize_response.status_code == 200:
        # Извлекаем нулевой и единственный элемент из ответа
        # Первый 0 - индекс переменной вывода
        # Второй 0 - индекс запроса
        response_data = json.loads(recognize_response.json()["outputs"][0]["data"][0])
        response_data["request_id"] = recognize_response.json()["id"]
        return response_data
    else:
        return recognize_response.json()


def get_tf_predictions(ids: List[str]):
    """
    Получение предсказаний от продовых моделей.
    Предсказания записываются в Dict predictions.
    Если не удается получить предсказание, то оно заполняется пустой строкой
    """
    predictions = {
        "c_guid": [],
        "surname": [],
        "name": [],
        "middle_name": [],
        "front_serial": [],
        "dateout": [],
        "birthday": [],
    }

    name_to_num = {
        "surname": 0,
        "name": 1,
        "middle_name": 2,
        "front_serial": 3,
        "dateout": 5,
        "birthday": 6,
    }

    for c_guid in tqdm(ids):
        response = recognize_driver_license(c_guid, request_id="1")
        predictions["c_guid"].append(c_guid)
        for key in name_to_num.keys():
            try:
                field_text = response["fields_recognition_result"][name_to_num[key]][
                    "field_text"
                ]
                predictions[key].append(field_text)
            except:
                predictions[key].append("")

    return predictions


def get_preds_with_cols(
    model, nums_to_word, ds, rows: Dict[str, List[str]], size: tuple[int, int]
) -> Dict[str, List[str]]:
    """
    model: Модель, которая будет предсказывать значение (для каждой задачи своя)
    nums_to_word: Функция, которая будет переводить числа в символы (для каждой задачи своя)
    ds: Датасет, который содержит c_guid и путь до изображения (может содержать несколько путей, к примеру: путь до кропа фамилии/имени/отчества
    rows: Словарь, в который будут заполняться предсказания (Должен совпадать с колонками тестируемого датасета)
    size: Размер изображения, на котором было произведено обучение модели (для каждого свой)

    Вывод:
    predictions: Словарь, который будет содержать предсказания по каждой колонке, если пути нет, то будет заполнено пустым значением.
    """
    cols = ds.columns[1:]
    assert cols.tolist() == list(
        rows.keys()
    ), "Словарь должен совпадать с названием колонок тестируемого датасета"
    for inx in tqdm(range(len(ds))):
        for col in cols:
            if pd.isnull(ds.loc[inx, col]):
                rows[col].append("")
                continue
            img = prepare_img(ds.loc[inx, col], device, size)
            output = model(img).permute(1, 0, 2)
            word = decode(output, nums_to_word)
            rows[col].append(word)
    return rows


def get_pt_predictions(fio_ds, date_ds, serial_ds) -> Dict[str, List[str]]:
    """
    fio_ds: Датафрейм, который содержит путь до изображения для предсказания
    date_ds: Датафрейм, который содержит путь до изображения для предсказания
    serial_ds: Датафрейм, который содержит путь до изображения для предсказания

    Внутри функции получаются все предсказания для каждого датафрейма и сохранаются в один словарь
    В начале происходит проверка на то, что все датафреймы идут в верном порядке и содержат одинаковое количество изображений (по количеству строк)

    В итоге на выходе получается словарь, который содержит все предсказания для каждой строки. (Если вдруг нет предсказания, то заполняется пустой строкой)
    """
    assert (
        fio_ds["c_guid"].tolist()
        == date_ds["c_guid"].tolist()
        == serial_ds["c_guid"].tolist()
    ), "Не совпадают айди"
    predictions = {"c_guid": fio_ds["c_guid"].tolist()}
    fio_rows = {"surname": [], "name": [], "midlename": []}
    date_rows = {"dateout": [], "birthday": []}
    serial_rows = {"siriestype3": []}
    fio_preds = get_preds_with_cols(
        fio_model, fio_indices_to_text, fio_ds, fio_rows, fio_size
    )
    date_preds = get_preds_with_cols(
        date_model, date_indices_to_text, date_ds, date_rows, date_size
    )
    serial_preds = get_preds_with_cols(
        serial_model, serial_indices_to_text, serial_ds, serial_rows, serial_size
    )

    predictions.update(fio_preds)
    predictions.update(date_preds)
    predictions.update(serial_preds)

    return predictions
