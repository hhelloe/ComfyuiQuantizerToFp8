import torch
import torch.nn as nn
from safetensors.torch import save_file
from tqdm import tqdm
import os


class Fp8QuantLinear(nn.Module):
    """FP8 W8A8 量化线性层"""

    def __init__(self, in_features, out_features, bias=True, fp8_format="float8_e4m3fn", device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fp8_format = fp8_format

        # FP8 量化后的权重
        self.register_buffer("weight_fp8", torch.empty((out_features, in_features), dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale", torch.tensor(1.0))

        # bias 保持高精度
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def quantize_weight(self, weight):
        """将权重量化为 FP8，并保存 scale"""
        fp8_max = 448.0 if self.fp8_format == "float8_e4m3fn" else 57472.0
        abs_max = weight.abs().max()
        self.weight_scale = fp8_max / (abs_max + 1e-6)
        self.weight_fp8 = (weight * self.weight_scale).to(dtype=torch.float8_e4m3fn)

    def quantize_activation(self, x):
        """Per-tensor 动态激活量化"""
        fp8_max = 448.0 if self.fp8_format == "float8_e4m3fn" else 57472.0
        abs_max = x.abs().max() + 1e-6
        scale = fp8_max / abs_max.to(x.dtype)
        return (x * scale).to(dtype=torch.float8_e4m3fn), scale

    def forward(self, x):
        """W8A8 前向传播，输出保持 FP8"""
        x_fp8, x_scale = self.quantize_activation(x)

        # FP8 矩阵乘法
        out = torch.matmul(x_fp8.float(), self.weight_fp8.t().float())
        out = out / (x_scale * self.weight_scale)

        if self.bias is not None:
            out = out + self.bias

        return out.to(torch.float8_e4m3fn)


def replace_linear_with_fp8(module, fp8_format="float8_e4m3fn", stats=None):
    """递归替换 nn.Linear 为 Fp8QuantLinear"""
    if stats is None:
        stats = {"total": 0, "replaced": 0}

    # 处理子模块
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            # 创建新层
            new_layer = Fp8QuantLinear(
                child.in_features, child.out_features,
                child.bias is not None, fp8_format,
                child.weight.device, child.weight.dtype
            )
            # 量化权重并复制 bias
            new_layer.quantize_weight(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data = child.bias.data.clone()
            # 替换
            setattr(module, name, new_layer)
            stats["total"] += 1
            stats["replaced"] += 1
        else:
            replace_linear_with_fp8(child, fp8_format, stats)

    return module, stats

class FP8Quantizer:
    def __init__(self, quant_dtype: str = "float8_e5m2"): #初始化，主要是选中格式
        if not hasattr(torch, quant_dtype):
            raise ValueError(f"不支持的量化格式: {quant_dtype}")
        self.quant_dtype = quant_dtype
        self.scale_factors = {}

    def quantize_weights(self, weight:torch.Tensor, layer_name: str) -> torch.Tensor: 
        '''
        量化的时候需要输入权重，返回量化后的权重
        所以一开始需要把输入权重的格式和量化后权重的格式明确
        
        '''
        if not weight.is_floating_point():
            raise ValueError("输入权重必须是浮点数.")
        
        original_device = weight.device

        can_use_cuda = torch.cuda.is_available()
        target_device = torch.device("cuda") if can_use_cuda else torch.device("cpu")
        

        if not can_use_cuda and "float8" in self.quant_dtype:
            print("cuda不可用，使用CPU进行FP8量化可能会非常慢.")
            target_device = torch.device("cpu")

        weight_on_target = weight.to(target_device)

        max_val = torch.max(torch.abs(weight_on_target))

        if max_val == 0:
            target_torch_dtype = getattr(torch, self.quant_dtype)
            return torch.zeros_like(weight, dtype=target_torch_dtype)
        else:
            scale = max_val / 127.0
            scale = torch.max(scale, torch.tensor(1e-12, device=target_device, dtype=weight_on_target.dtype))

        quantized_weight_simulated = torch.round(weight_on_target / scale * 127.0) / 127.0 * scale

        target_torch_dtype = getattr(torch, self.quant_dtype)
        quantized_weight = quantized_weight_simulated.to(target_torch_dtype)

        return quantized_weight
    
    def apply_quantization(self, state_dict:dict) -> dict:
        quantized_state_dict = {}
        eligible_tensors = {name: param for name, param in state_dict.items() if isinstance(param, torch.Tensor)
                             and param.is_floating_point()}
        progress_bar = tqdm(eligible_tensors.items(), desc="Quantizing weights to FP8", unit="tensor", leave=False)
        for name, param in progress_bar:
            quantized_state_dict[name] = self.quantize_weights(param.clone(), name)

        for name, param in state_dict.items():
            if name not in quantized_state_dict:
                quantized_state_dict[name] = param
        return quantized_state_dict
    
class ModelToStateDict:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"model": ("MODEL",)}}
    RETURN_TYPES = ("MODEL_STATE_DICT",); RETURN_NAMES = ("model_state_dict",)
    FUNCTION = "get_state_dict"; CATEGORY = "Model Quantization/Utils" 
    def get_state_dict(self, model):
        print("Extracting state dict from model...")
        if not hasattr(model, "model"):
            raise ValueError("Provided model does not have a state_dict method.")
        if not hasattr(model, "model") or not hasattr(model.model, "state_dict"):
            raise ValueError("Provided model does not have a state_dict method.")
        try:
            original_state_dict = model.model.state_dict()
            print(f"[ModelToStateDict] Original keys sample: {list(original_state_dict.keys())[:5]}")
            state_dict_to_return = original_state_dict
            prefixes_to_try = ["model.", "diffusion_model."]
            prefix_found = False
            for prefix in prefixes_to_try:
                num_keys = len(original_state_dict)
                matches = sum(1 for key in original_state_dict if key.startswith(prefix))
                if matches / num_keys > 0.5 and matches > 0:
                    print(f"Detected prefix '{prefix}' in more than 50% of keys. Stripping prefix...")
                    state_dict_to_return = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in original_state_dict.items()}
                    prefix_found = True
                    break
            if not prefix_found:print("No common prefix detected in more than 50% of keys.")
            dtypes = {}
            total = 0
            for k, v in state_dict_to_return.items():
                if isinstance(v, torch.Tensor):
                    total += 1
                    dt = str(v.dtype)
                    dtypes[dt] = dtypes.get(dt, 0) + 1
            return (state_dict_to_return,)
        except Exception as e:
            raise ValueError(f"Error extracting state_dict: {e}")
        
class FP8QuantizeFormat:
    @classmethod
    def INPUT_TYPES(s): return { "required": { "model_state_dict": ("MODEL_STATE_DICT",), "fp8_format": (["float8_e4m3fn", "float8_e5m2"], {"default": "float8_e5m2"}), } }
    RETURN_TYPES = ("MODEL_STATE_DICT",); RETURN_NAMES = ("quantized_model_state_dict",)
    FUNCTION = "quantize_model"; CATEGORY = "Model Quantization/FP8 Direct" 
    def quantize_model(self, model_state_dict: dict, fp8_format: str):
        if not isinstance(model_state_dict, dict) or not model_state_dict:
            raise ValueError("Invalid model state dict provided for quantization.")
            print(f"Starting FP8 quantization with format: {fp8_format}")
            return ({},)
        try:
            quantizer = FP8Quantizer(quant_dtype=fp8_format)
            quantized_state_dict = quantizer.apply_quantization(model_state_dict)
            found = False
            for k, v in quantized_state_dict.items():
                if isinstance(v, torch.Tensor) and "float8" in str(v.dtype):
                    print(f"[QuantizeFP8Format] Sample '{k}' dtype: {v.dtype}, dev: {v.device}")
                    found=True
                    break
            if not found:
                print("no Tensor converted to FP8 dtype found in quantized state dict.")
                
            print("FP8 quantization completed.")
        except Exception as e:print(f"Error during FP8 quantization: {e}")
        return (quantized_state_dict,)
            
class SaveAsSafeTensor: # No changes needed
    @classmethod
    def INPUT_TYPES(s): return { "required": { "quantized_model_state_dict": ("MODEL_STATE_DICT",), "absolute_save_path": ("STRING", {"default": "/home/aus/ComfyUI/models/checkpoints", "multiline": False}), } }
    RETURN_TYPES = () ; OUTPUT_NODE = True ; FUNCTION = "save_model"; CATEGORY = "Model Quantization/Save" 
    def save_model(self, quantized_model_state_dict: dict, absolute_save_path: str):
        if not isinstance(quantized_model_state_dict, dict) or not quantized_model_state_dict:
            raise ValueError("Invalid quantized model state dict provided for saving.")
        if not os.path.isabs(absolute_save_path):
            raise ValueError("The save path must be an absolute path.")
        if not absolute_save_path.lower().endswith(".safetensors"):
            absolute_save_path += ".safetensors"
            print(f"Appended .safetensors extension. New path: {absolute_save_path}")
        try:
            output_dir = os.path.dirname(absolute_save_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            cpu_state_dict = {}
            dtype_counts = {}
            total_tensors = 0
            for k, v in quantized_model_state_dict.items():
                if isinstance(v, torch.Tensor):
                    total_tensors += 1
                    tensor_to_save = v.cpu() if v.device.type != "cpu" else v
                    cpu_state_dict[k] = tensor_to_save
                    dt_str = str(tensor_to_save.dtype)
                    dtype_counts[dt_str] = dtype_counts.get(dt_str, 0) + 1
                else:
                    cpu_state_dict[k] = v
            save_file(cpu_state_dict, absolute_save_path)
            return {"ui": {"text": [f"Saved: {absolute_save_path}"]}}    
        except Exception as e:
            print(f"Error saving model as safetensor: {e}")
            return {"ui": {"text": [f"Error: {e}"]}}


class FP8OnlineQuantizer:
    """在线量化 ComfyUI 节点"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "fp8_format": (["float8_e4m3fn", "float8_e5m2"], {"default": "float8_e4m3fn"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("quantized_model",)
    FUNCTION = "quantize"
    CATEGORY = "Model Quantization/FP8 Online"

    def quantize(self, model, fp8_format):
        # 提取实际 PyTorch 模型
        actual_model = self._get_model(model)
        if actual_model is None:
            raise ValueError("无法从 MODEL 对象中提取模型")

        # 替换 Linear 层
        _, stats = replace_linear_with_fp8(actual_model, fp8_format)
        print(f"[FP8OnlineQuantizer] 替换完成: {stats['replaced']}/{stats['total']} 层")

        return (model,)

    def _get_model(self, model):
        """从 ComfyUI MODEL 对象提取实际模型"""
        if hasattr(model, "model") and hasattr(model.model, "model"):
            return model.model.model
        return None

