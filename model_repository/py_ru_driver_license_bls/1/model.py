import json
import time
import yaml
import traceback
import triton_python_backend_utils as pb_utils
import numpy as np

from detection import detect
from driver_license_side import DriverLicenseSide
from recognition import recognize
from recognition.fields_recognition import ResultFieldRecognition
from utils import download_transpose, ImageNotLoad


APP_CONFIG_PATH = "/app_configs/driver_license_config.yaml"


class ResultDriverLicenseRecognition:
    def __init__(self, is_driver_license_found, side, fields_recognition_result):
        """Результат распознавания водительского удостоверение

        Parameters
        ----------
        is_driver_license_found : bool
            Флаг распознавания водительского удостоверение
        fields_recognition_result : list(ResultFieldRecognition)
            Список результатов распознавания полей
        """
        self.is_driver_license_found = is_driver_license_found
        self.side = side
        self.fields_recognition_result = fields_recognition_result

    def to_dict(self):
        """ "Сереализация экземпляра в словарь для отправки в json

        Returns
        -------
        dict
            Словарь результата распознавания
        """
        return {
            "is_driver_license_found": self.is_driver_license_found,
            "side": self.side,
            "fields_recognition_result": [
                x.to_dict() for x in self.fields_recognition_result
            ]
        }


class TritonPythonModel:
    """Каждая создаваемая модель Python должна иметь имя класса «TritonPythonModel»."""

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Model config
        self.model_config = json.loads(args["model_config"])
        self.model_name = f'{args["model_name"]}'
        self.model_version = f'{args["model_version"]}'
        # Output type
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                self.model_config, "recognition_response"
            )["data_type"]
        )
        # String base for log
        self.model_log = f"model_name: {self.model_name}; version: {self.model_version}"
        # Custom app config
        with open(APP_CONFIG_PATH, "r", encoding="utf-8") as conf_f:
            self.config = yaml.load(conf_f, Loader=yaml.Loader)

        # Counter metrics
        self.metric_counter_family = pb_utils.MetricFamily(
            name="passport_counter_metrics",
            description="Counter metrics",
            kind=pb_utils.MetricFamily.COUNTER,  # or pb_utils.MetricFamily.GAUGE
        )
        # Gauge metrics
        self.metric_gauge_family = pb_utils.MetricFamily(
            name="passport_gauge_metrics",
            description="Gauge metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )

        # Image metrics
        # Load
        self.metric_load_image_time = self.metric_counter_family.Metric(
            labels={"model": self.model_name, "version": self.model_version, "metric": "load_image_time"}
        )
        self.metric_load_image_failure = self.metric_counter_family.Metric(
            labels={"model": self.model_name, "version": self.model_version, "metric": "load_image_failure"}
        )

        # Detection metrics
        # Failed
        self.metric_detection_failur = self.metric_counter_family.Metric(
            labels={"model": self.model_name, "version": self.model_version, "metric": "detection_failur"}
        )
        # Scores
        self.metric_detection_min_score = self.metric_gauge_family.Metric(
            labels={"model": self.model_name, "version": self.model_version, "metric": "detection_min_score"}
        )

        # Recognition metrics
        # Failed
        self.metric_recognition_failur = self.metric_counter_family.Metric(
            labels={"model": self.model_name, "version": self.model_version, "metric": "recognition_failur"}
        )
        # Scores
        self.metric_recognition_min_score = self.metric_gauge_family.Metric(
            labels={"model": self.model_name, "version": self.model_version, "metric": "recognition_min_score"}
        )

        self.failed_requests_metric = self.metric_counter_family.Metric(
            labels={"model": self.model_name, "metric": "inference_request_failure"}
        )

    def execute(self, requests):
        """
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`

        Raises
        ------
        Exception
            Ошибки загрузки изображения, детекции и распознавания
        """
        responses = []
        for request in requests:
            log_msg = f"{self.model_log} request_id: {request.request_id()};"
            pb_utils.Logger.log_info(f"{log_msg} start process")
            try:
                recognition_result = ResultDriverLicenseRecognition(is_driver_license_found=False,
                                                                    side=DriverLicenseSide.NoneSide,
                                                                    fields_recognition_result=[])
                # get INPUT
                pb_utils.Logger.log_info(f"{log_msg} get input tensor")
                input_request = pb_utils.get_input_tensor_by_name(
                    request, "image_guid"
                ).as_numpy()[0]
                pb_utils.Logger.log_info(f"{log_msg} get request json")
                img_guid = json.loads(input_request.decode())["guid"]
                pb_utils.Logger.log_info(f"{log_msg} img_guid: {img_guid}")

                # load image
                img_url = self.config["image_download"]["url"].format(
                    img_guid, self.config["image_download"]["token"]
                )
                pb_utils.Logger.log_verbose(f"{log_msg} img_url: {img_url}")
                image_load_start_ns = time.time_ns()
                img = download_transpose(
                    img_url, self.config["image_download"]["timeout_sec"]
                )
                if img is None:
                    raise ImageNotLoad(f"Image with guid: {img_guid} is None")
                image_load_end_ns = time.time_ns()
                image_load_time = image_load_end_ns - image_load_start_ns
                pb_utils.Logger.log_info(
                    f"{log_msg} image load time: {image_load_time / 1e+6} ms"
                )
                self.metric_load_image_time.increment(image_load_time)
                pb_utils.Logger.log_info(f"{log_msg} image shape: {img.shape}")

                # infer detection
                pb_utils.Logger.log_info(f"{log_msg} infer detection")
                detection_result = detect.infer_model(
                    img, self.config["ru_driver_license_models"], log_msg
                )
                if len(detection_result.predictions) > 0:

                    min_detection_score = min(
                        [x.score for x in detection_result.predictions]
                    )
                    pb_utils.Logger.log_info(
                        f"{log_msg} min detection score: {min_detection_score}"
                    )
                    self.metric_detection_min_score.set(min_detection_score)
                else:
                    pb_utils.Logger.log_info(
                        f"{log_msg} detection is empty"
                    )

                pb_utils.Logger.log_info(
                    f"{log_msg} detection is correct: {detection_result.is_correct}"
                )
                if detection_result.is_correct:
                    recognition_result.is_driver_license_found = True

                    # infer recognition
                    pb_utils.Logger.log_info(f"{log_msg} infer recognition")
                    if detection_result.is_front_side:
                        recognition_result.side = DriverLicenseSide.FrontSide
                        recognition_result.fields_recognition_result = self.fields_recognition(detection_result, log_msg)
                    else:
                        recognition_result.side = DriverLicenseSide.BackSide
                    # collecting recognize metrics
                    not_recognize_list = []
                    min_word_score = 1
                    for recognize_res in recognition_result.fields_recognition_result:
                        if not recognize_res.is_ocr:
                            not_recognize_list.append(recognize_res.field_name)
                        elif recognize_res.field_text_score < min_word_score:
                            min_word_score = recognize_res.field_text_score
                    pb_utils.Logger.log_info(
                        f"{log_msg} not recognize fields: {not_recognize_list}"
                    )
                    self.metric_recognition_failur.increment(len(not_recognize_list))
                    self.metric_recognition_min_score.set(min_word_score)
                else:
                    self.metric_detection_failur.increment(1)
                pb_utils.Logger.log_info(f"{log_msg} create response")
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "recognition_response",
                            np.array(
                                [json.dumps(recognition_result.to_dict())],
                                dtype=self.output_dtype,
                            ),
                        )
                    ]
                )
            except ImageNotLoad as img_load_err:
                self.metric_load_image_failure.increment(1)
                trace_msg = "".join(traceback.format_exception(img_load_err))
                err_msg = f'{log_msg} Error loading image with guid {img_guid};'
                pb_utils.Logger.log_error(f'{err_msg} Trace {trace_msg}')
                inference_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f'{err_msg}', pb_utils.TritonError.INTERNAL)
                )
            except Exception as err:
                trace_msg = "".join(traceback.format_exception(err))
                err_msg = f'Exception: {log_msg} Error message: "{str(err)} {trace_msg}'
                self.failed_requests_metric.increment(1)
                pb_utils.Logger.log_error(err_msg)
                inference_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f'{err_msg}', pb_utils.TritonError.INTERNAL)
                )
            responses.append(inference_response)
        return responses

    def fields_recognition(self, detection_result, log_msg=""):
        recognized_fields = self.config["ru_driver_license_models"]["recognized_fields"]
        result_dict = dict()
        for field in recognized_fields.keys():
            result_dict[field] = ResultFieldRecognition(field)

        for field_detection_result in detection_result.predictions:
            if field_detection_result.field_name in recognized_fields:
                field_name = field_detection_result.field_name
                field_bbox = field_detection_result.bbox
                text_recognition_result = recognize.infer_model(
                    detection_result,
                    field_name,
                    field_bbox,
                    self.config["ru_driver_license_models"],
                    log_msg,
                )
                result_dict = self.save_field_recognition_result(
                    result_dict,
                    field_name,
                    field_detection_result,
                    text_recognition_result,
                )
        return list(result_dict.values())

    def save_field_recognition_result(self, result_dict, field_name, field_detection_result, text_recognition_result):
        result_dict[field_name].is_detection = True
        result_dict[field_name].field_bbox = field_detection_result.bbox
        result_dict[field_name].field_detect_score = field_detection_result.score

        if text_recognition_result.is_correct:
            result_dict[field_name].is_ocr = text_recognition_result.is_correct
            result_dict[field_name].field_text = text_recognition_result.predict_word
            result_dict[field_name].field_text_score = text_recognition_result.word_score
            result_dict[field_name].field_symbol_scores = text_recognition_result.symbol_scores

        return result_dict

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
