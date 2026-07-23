# TorchDCM CPU/GPU Device Stress (calibration)

Rows use pure synthetic mixed-logit cases with identical initialization and antithetic normal draws on CPU and CUDA. CPU workers are capped by the requested timeout; GPU workers use the same TorchDCM model code with `device='cuda'`.

| case | N | J | K | rho | RC | draws | CPU total model s | GPU total model s | GPU estimate s | GPU memory MB | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| torch_device_calib_mixl_50k | 50000 | 12 | 12 | 0.5 | 8 | 256 | 124.860 | 4.205 | 3.578 | 6422 | True |
| torch_device_calib_mixl_100k | 100000 | 12 | 12 | 0.5 | 8 | 512 | Timeout | 9.155 | 8.556 | 25086 | CPU timeout; GPU completed |

## Diagnostics

- `torch_device_calib_mixl_50k` comparison: {'objective_diff': 5.623223842121661e-08, 'max_param_diff': 1.839942884163115e-07, 'speedup_total_model': 29.69221848131661, 'speedup_estimate': 34.57104229483485, 'consistent': True}
  - cpu: status=Completed, total_model_s=124.860, estimate_s=123.688, loglike=-25253.595426866188
  - cuda: status=Completed, total_model_s=4.205, estimate_s=3.578, loglike=-25253.595426809956
- `torch_device_calib_mixl_100k` comparison: {'consistent': 'CPU timeout; GPU completed', 'speedup_total_model': '>32.8x'}
  - cpu: status=Timeout, total_model_s=Timeout, estimate_s=Timeout, loglike=None
  - cuda: status=Completed, total_model_s=9.155, estimate_s=8.556, loglike=-50143.69839440792
