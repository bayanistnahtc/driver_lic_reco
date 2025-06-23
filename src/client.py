import json
import requests


def check_models(triton_host="localhost", triton_http_port="8000"):
    models_name = [
            "onnx_ru_driver_licence_detection",
            "onnx_ru_driver_licence_recognition_fio",
            "onnx_ru_driver_licence_recognition_date",
            "onnx_ru_driver_licence_recognition_serial",
        ]

    model_status_response = requests.post(
        f"http://{triton_host}:{triton_http_port}/v2/repository/index", json.dumps({})
    )
    models_state = {
        x["name"]: x["state"] if "state" in x else None
        for x in model_status_response.json()
    }
    for name in models_name:
        assert name in models_state
        assert models_state[name] == "READY"


def recognize_fields(
    photo_guid: str,
    photo_size: int,
    request_id: str,
    photo_type: str = "driver_licence",
    service_url="http://localhost:8090",
):
    data = {"guid": photo_guid,"size": str(photo_size), "request_id": request_id}

    url = f"{service_url}/api/v1/recognize/{photo_type}"
    recognize_response = requests.post(url, json=data)
    
    return recognize_response.json()
