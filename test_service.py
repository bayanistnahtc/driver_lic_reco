import json

import client


def test_model_ready():
    client.check_models()


def test_recognition():
    with open('test/data/service_test_input.json', 'r') as f:
        inputs_guid = json.load(f)

    with open('test/data/service_test_output.json', 'r') as f:
        outputs_service = json.load(f)

    for indx, guid in enumerate(inputs_guid[:20]):
        pred = client.recognize_driver_license(guid, f'test_{indx}')
        if pred.get('is_driver_license_found') is not None:
            true = outputs_service[indx]
            assert true['is_driver_license_found'] == pred['is_driver_license_found']
            true_fields = [x['field_text'] for x in true['fields_recognition_result']]
            pred_fields = [x['field_text'] for x in pred['fields_recognition_result']]
            assert true_fields == pred_fields
        else:
            assert outputs_service[indx].keys() == pred.keys()
