import torch
from enum import Enum
from segformer_pytorch import Segformer
from importlib import resources
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, LiltForTokenClassification

from ocr_microservice.ocr_pipeline.pre_ocr.pre_ocr import receipt_segmentation, receipt_object_detection_yolos
from ocr_microservice.ocr_pipeline.pre_ocr.helpers.yolos_inference import load_model
from ocr_microservice.ocr_pipeline.pre_ocr.helpers.segmentation_model import segmentation_model

from ocr_microservice.ocr_pipeline.post_ocr.helpers.do_lilt import run_lilt
from ocr_microservice.ocr_pipeline.post_ocr.helpers.post_process import post_process_lilt
from ocr_microservice.ocr_pipeline.post_ocr.helpers.post_correction import post_correction
from ocr_microservice.ocr_pipeline.post_ocr.helpers.to_json import to_json
from ocr_microservice.ocr_pipeline.resources.models import paddle, lilt, segformer, yolos


class Config:
    def __init__(self, save_results: bool = False, data_dir: str = "./data/output"):
        self.save_results = save_results
        self.data_dir = data_dir


class Pipeline:
    def __init__(self):
        #self.pre_ocr_algo = receipt_segmentation
        self.pre_ocr_algo = receipt_object_detection_yolos
        self.post_ocr_steps = [
            run_lilt,
            post_process_lilt,
            post_correction,
        ]
        self.post_ocr_steps_multi_image = [
            to_json,
        ]

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

    def _load_segformer_model(self, hardware_type: HardwareType):
        self.segformer_model = segmentation_model

        if hardware_type == HardwareType.APPLE_SILICON and torch.backends.mps.is_available():
            self.torch_device = torch.device("mps")
            model_file = "mps_weights_segformer_500.pth"
        elif hardware_type == HardwareType.CONDA_GPU and torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
            raise NotImplementedError("CUDA model is missing.")
        else:
            self.torch_device = torch.device("cpu")
            model_file = "cpu_weights_segformer_110.pth"
        
        segformer_path = str(resources.path(segformer, model_file))
        self.segformer_model.load_state_dict(torch.load(segformer_path))
        self.segformer_model.eval()
        self.segformer_model.to(self.torch_device)
    
    def _load_yolos_model(self, hardware_type: HardwareType):
        if hardware_type == HardwareType.APPLE_SILICON and torch.backends.mps.is_available():
            self.torch_device = torch.device("mps")
            model_file = "TODO"
            raise NotImplementedError("MPS model is missing.")
        elif hardware_type == HardwareType.CONDA_GPU and torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
            raise NotImplementedError("CUDA model is missing.")
        else:
            self.torch_device = torch.device("cpu")
            model_file = "epoch=23-step=1000.ckpt"

        yolos_path = str(resources.path(yolos, model_file))
        self.yolos_model = load_model(yolos_path, self.torch_device)

    def __init__(self, hardware_type: HardwareType = HardwareType.CPU, load_models: bool = True):
        if load_models:
            self._load_paddle_model(hardware_type)
            self._load_lilt_model(hardware_type)
            self._load_segformer_model(hardware_type)
            self._load_yolos_model(hardware_type)
