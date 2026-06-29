import torch
import torch.nn as nn
from safetensors import safe_open
from circuit_tracer.utils import get_default_device
from circuit_tracer.transcoder.activation_functions import JumpReLU

# We need SingleLayerTranscoder for the return type and class instantiation.
# To avoid circular imports, we import inside the function or rely on the fact 
# that this module is imported by single_layer_transcoder.py
# But for type hinting and inheritance we might need it. 
# We will treat SingleLayerTranscoder as passed in or imported inside.

def load_gemma_3_transcoder(
    path: str,
    layer: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    revision: str | None = None,
    **kwargs,
):
    from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder

    if device is None:
        device = get_default_device()

    # Load safetensors
    param_dict = {}
    with safe_open(path, framework="pt", device=device.type) as f:
        for k in f.keys():
            param_dict[k] = f.get_tensor(k)

    # Key mapping
    mapping = {
        "w_enc": "W_enc",
        "w_dec": "W_dec",
        "threshold": "activation_function.threshold"
    }
    
    for old_k, new_k in mapping.items():
        if old_k in param_dict:
            param_dict[new_k] = param_dict.pop(old_k)

    # Handle shapes
    # SingleLayerTranscoder internal: W_enc is (d_transcoder, d_model)
    # File usually (d_model, d_transcoder).
    if param_dict["W_enc"].shape[0] != param_dict["b_enc"].shape[0]:
         param_dict["W_enc"] = param_dict["W_enc"].T.contiguous()

    d_transcoder = param_dict["b_enc"].shape[0]
    d_model = param_dict["b_dec"].shape[0]
    
    threshold = param_dict.get("activation_function.threshold")
    # Default bandwidth 0.1 if not present? 
    # JumpReLU needs threshold.
    if threshold is None:
        # If no threshold, maybe it's ReLU? But Gemma 3 scope is JumpReLU.
        # Let's assume it's present or default to 0 (which makes it ReLU-like + jump 0)
        threshold = torch.tensor(0.0, device=device, dtype=dtype)

    activation_function = JumpReLU(threshold, 0.1)
    
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(
            d_model, 
            d_transcoder, 
            activation_function, 
            layer,
            device=device,
            dtype=dtype
        )
    
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder.to(dtype)

