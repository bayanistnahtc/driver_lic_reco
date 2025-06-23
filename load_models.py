import json
import argparse
import requests


def load_models(triton_host='localhost', 
                triton_http_port='8000'):
    models = ['py_ru_driver_license_bls',
    'tf_ru_driver_license_recognition_date',
    'tf_ru_driver_license_recognition_fio',
    'tf_ru_driver_license_recognition_serial',
    'tf_ru_driver_license_detection']

    for mod in models:
        r = requests.post(f'http://{triton_host}:{triton_http_port}/v2/repository/models/{mod}/load', json.dumps({}))
        print(mod)
        print(r)

    r = requests.post(f'http://{triton_host}:{triton_http_port}/v2/repository/index', json.dumps({}))
    print('\nModels status:\n')
    for mod_state in r.json():
        print(mod_state)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="calculate X to the power of Y")
    parser.add_argument("--host", type=str, help="Triton host")
    parser.add_argument("--port", type=str, help="Triton port")
    args = parser.parse_args()
    load_models(args.host, args.port)
