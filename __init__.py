# __init__.py
from .nodes import ModelToStateDict, FP8QuantizeFormat, SaveAsSafeTensor, FP8OnlineQuantizer

NODE_CLASS_MAPPINGS = {
    "ModelToStateDict": ModelToStateDict,
    "FP8QuantizeFormat": FP8QuantizeFormat,
    "SaveAsSafeTensor": SaveAsSafeTensor,
    "FP8OnlineQuantizer": FP8OnlineQuantizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelToStateDict": "Model → State Dict",
    "FP8QuantizeFormat": "Quantize to FP8",
    "SaveAsSafeTensor": "Save As SafeTensor",
    "FP8OnlineQuantizer": "Online FP8 Quantize",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]