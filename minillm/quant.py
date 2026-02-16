"""
Quantization utilities for faster inference
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


def quantize_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """
    Quantize float32 tensor to int8
    
    Returns:
        quantized_tensor: int8 tensor
        scale: quantization scale
        zero_point: quantization zero point
    """
    min_val = tensor.min()
    max_val = tensor.max()
    
    scale = (max_val - min_val) / 255.0
    zero_point = min_val
    
    quantized = ((tensor - zero_point) / scale).round().clamp(0, 255).to(torch.int8)
    
    return quantized, scale, zero_point


def dequantize_int8(quantized: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    """Dequantize int8 tensor back to float32"""
    return quantized.float() * scale + zero_point


class Int8Linear(nn.Module):
    """INT8 quantized linear layer"""
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Store as int8
        self.register_buffer('weight', torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8))
        self.register_buffer('weight_scale', torch.tensor(1.0))
        self.register_buffer('weight_zero_point', torch.tensor(0.0))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None
    
    @classmethod
    def from_float(cls, float_linear: nn.Linear):
        """Create Int8Linear from a float Linear layer"""
        int8_linear = cls(float_linear.in_features, float_linear.out_features, float_linear.bias is not None)
        
        # Quantize weight
        weight_quantized, scale, zero_point = quantize_int8(float_linear.weight.data)
        int8_linear.weight.copy_(weight_quantized)
        int8_linear.weight_scale = scale
        int8_linear.weight_zero_point = zero_point
        
        if float_linear.bias is not None:
            int8_linear.bias.copy_(float_linear.bias.data)
        
        return int8_linear
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on-the-fly (can be optimized with INT8 GEMM kernels)
        weight_float = dequantize_int8(self.weight, self.weight_scale.item(), self.weight_zero_point.item())
        output = torch.nn.functional.linear(x, weight_float, self.bias)
        return output


def quantize_model(model: nn.Module, dtype: str = "int8") -> nn.Module:
    """
    Quantize a model's linear layers
    
    Args:
        model: PyTorch model
        dtype: "int8" or "int4" (int4 not implemented yet)
    
    Returns:
        Quantized model
    """
    if dtype == "int8":
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get parent module
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                # Replace with quantized version
                int8_linear = Int8Linear.from_float(module)
                setattr(parent, child_name, int8_linear)
                print(f"Quantized {name}: {module.in_features}x{module.out_features}")
    
    return model


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    
    return {
        "param_mb": param_size / 1024**2,
        "buffer_mb": buffer_size / 1024**2,
        "total_mb": size_mb,
    }
