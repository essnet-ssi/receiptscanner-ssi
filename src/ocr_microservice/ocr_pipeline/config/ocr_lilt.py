from enum import Enum
from importlib import resources
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, LiltForTokenClassification

from ocr_microservice.ocr_pipeline.post_ocr.helpers.do_lilt import run_lilt
from ocr_microservice.ocr_pipeline.post_ocr.helpers.to_json_ocr_lilt import to_json
from ocr_microservice.ocr_pipeline.resources.models import paddle, lilt


class Config:
    def __init__(self, save_results: bool = False, data_dir: str = "./data/output"):
        self.save_results = save_results
        self.data_dir = data_dir


class Pipeline:
    def __init__(self):
        self.pre_ocr_algo = None
        self.post_ocr_steps = [ run_lilt ]
        self.post_ocr_steps_multi_image = [ to_json ]

class Cache:
    ocr_data = None # used when the pre-ocr step uses paddleocr for rotation

class HardwareType(Enum):
    CPU = 1
    CONDA_GPU = 2
    APPLE_SILICON = 3


class Models:
    def _load_paddle_model(self, hardware_type: HardwareType):
        # https://paddlepaddle.github.io/PaddleOCR/en/ppocr/blog/inference_args.html
        rec_model_dir = str(resources.path(paddle, "paddle-model"))
        rec_char_dict_path = str(resources.path(paddle, "dict.txt"))
        use_gpu = hardware_type == HardwareType.CONDA_GPU
        self.paddle_model = PaddleOCR(
            rec_model_dir=rec_model_dir,
            rec_char_dict_path=rec_char_dict_path,
            lang="latin",
            use_gpu=use_gpu,
            show_debug=False,
            show_log=False,
            det_limit_side_len=10000,
            det_db_thresh=0.2,
            #det_db_box_thresh=0.6,
            det_db_unclip_ratio=2.0,
            det_db_score_mode='slow',
        )

    def _load_lilt_model(self, hardware_type: HardwareType):
        lilt_model_path = str(resources.path(lilt, "lilt-model"))
        self.lilt_model = LiltForTokenClassification.from_pretrained(lilt_model_path)
        self.lilt_auto_tokenizer = AutoTokenizer.from_pretrained(lilt_model_path)

    def __init__(self, hardware_type: HardwareType = HardwareType.CPU, load_models: bool = True):
        if load_models:
            self._load_paddle_model(hardware_type)
            self._load_lilt_model(hardware_type)
