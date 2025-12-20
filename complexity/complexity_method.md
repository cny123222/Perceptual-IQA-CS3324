# Computing FLOPs Using Python Libraries 

Recommended Libraries: 

1. ptflops (`pip install ptflops`)
```
from ptflops import get_model_complexity_info 
macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
```
Note: 1 MAC (Multiply-Accumulate) ≈ 2 FLOPs

2. fvcore (`pip install fvcore`)
```
from fvcore.nn import FlopCountAnalysis 
flops = FlopCountAnalysis(model, input_tensor)
print(f"Total FLOPs: {flops.total()}")
```

3. thop (`pip install thop`)
```
from thop import profile 
flops, params = profile(model, inputs=(input_tensor,))
```

4. DeepSpeed Flops Profiler
```
from deepspeed.profiling.flops_profiler import get_model_profile 
flops, macs, params = get_model_profile(model, input_shape=(batch_size, channels, height, width))
```