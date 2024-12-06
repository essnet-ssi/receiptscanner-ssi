import torch
from enum import Enum
from segformer_pytorch import Segformer
from importlib import resources
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, LiltForTokenClassification

class HardwareType(Enum):
    CPU = 1
    CONDA_GPU = 2
    APPLE_SILICON = 3

class Models:
    def _load_paddle_model(self, hardware_type: HardwareType):
        rec_model_dir = str(resources.path(paddle, "en_PP-OCRv3_rec_best4"))
        rec_char_dict_path = str(resources.path(paddle, "en_dict.txt"))
        use_gpu = hardware_type == HardwareType.CONDA_GPU
        self.paddle_model = PaddleOCR(
            rec_model_dir=rec_model_dir,
            rec_char_dict_path=rec_char_dict_path,
            lang="en",
            use_gpu=use_gpu,
            show_debug=False,
            show_log=False,
            det_limit_side_len=10000,
            det_db_score_mode='slow',
        )

    def __init__(self, hardware_type: HardwareType = HardwareType.CPU):
        self._load_paddle_model(hardware_type)
