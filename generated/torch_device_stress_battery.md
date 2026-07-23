# TorchDCM CPU/GPU Device Stress (battery)

Rows use pure synthetic data with identical model specifications, initialization, and data on CPU and CUDA; MixL also uses identical antithetic normal draws. Data generation is excluded, and times are medians over repeated model setup, optimization, and final likelihood evaluation within each worker.

| model | case | N | J | K | rho | RC | draws | CPU s | GPU s | speedup | GPU memory MB | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MNL | torch_device_mnl_250k | 250000 | 12 | 12 | 0.5 | 0 | 0 | 2.994 | 0.210 | 14.3x | 1045 | True |
| MNL | torch_device_mnl_500k | 500000 | 12 | 12 | 0.5 | 0 | 0 | 8.402 | 0.383 | 22.0x | 2030 | True |
| MNL | torch_device_mnl_1m | 1000000 | 12 | 12 | 0.5 | 0 | 0 | 17.247 | 0.752 | 22.9x | 3996 | True |
| NL | torch_device_nl_100k | 100000 | 12 | 12 | 0.5 | 0 | 0 | 2.015 | 0.168 | 12.0x | 486 | True |
| NL | torch_device_nl_250k | 250000 | 12 | 12 | 0.5 | 0 | 0 | 6.083 | 0.326 | 18.7x | 1116 | True |
| NL | torch_device_nl_500k | 500000 | 12 | 12 | 0.5 | 0 | 0 | 17.942 | 0.610 | 29.4x | 2171 | True |
| MixL | torch_device_mixl_5k | 5000 | 12 | 12 | 0.5 | 8 | 256 | 23.910 | 0.545 | 43.9x | 704 | True |
| MixL | torch_device_mixl_10k | 10000 | 12 | 12 | 0.5 | 8 | 256 | 46.876 | 0.833 | 56.3x | 1336 | True |
| MixL | torch_device_mixl_25k | 25000 | 12 | 12 | 0.5 | 8 | 256 | 129.483 | 1.823 | 71.0x | 3243 | True |

## Diagnostics

- `torch_device_mnl_250k` comparison: {'objective_diff': 1.4551915228366852e-11, 'relative_objective_diff': 1.1651091604105688e-16, 'max_param_diff': 4.884981308350689e-15, 'speedup_total_model': 14.285097684447528, 'speedup_estimate': 9.736783200649906, 'consistent': True}
  - cpu: status=Completed, total_model_s=2.994, estimate_s=1.648, loglike=-124897.44071052494
  - cuda: status=Completed, total_model_s=0.210, estimate_s=0.169, loglike=-124897.44071052493
- `torch_device_mnl_500k` comparison: {'objective_diff': 2.9103830456733704e-11, 'relative_objective_diff': 1.159329057661107e-16, 'max_param_diff': 8.881784197001252e-16, 'speedup_total_model': 21.961293153707928, 'speedup_estimate': 15.064524533770902, 'consistent': True}
  - cpu: status=Completed, total_model_s=8.402, estimate_s=4.512, loglike=-251040.2914893666
  - cuda: status=Completed, total_model_s=0.383, estimate_s=0.300, loglike=-251040.29148936656
- `torch_device_mnl_1m` comparison: {'objective_diff': 0.0, 'relative_objective_diff': 0.0, 'max_param_diff': 6.494804694057166e-15, 'speedup_total_model': 22.924410493727603, 'speedup_estimate': 15.580282837339977, 'consistent': True}
  - cpu: status=Completed, total_model_s=17.247, estimate_s=9.157, loglike=-501226.7393727701
  - cuda: status=Completed, total_model_s=0.752, estimate_s=0.588, loglike=-501226.7393727701
- `torch_device_nl_100k` comparison: {'objective_diff': 0.0, 'relative_objective_diff': 0.0, 'max_param_diff': 1.31117339208231e-13, 'speedup_total_model': 12.002683855066868, 'speedup_estimate': 10.340201353981545, 'consistent': True}
  - cpu: status=Completed, total_model_s=2.015, estimate_s=1.507, loglike=-41084.430739856325
  - cuda: status=Completed, total_model_s=0.168, estimate_s=0.146, loglike=-41084.430739856325
- `torch_device_nl_250k` comparison: {'objective_diff': 1.4551915228366852e-11, 'relative_objective_diff': 1.4083290018303009e-16, 'max_param_diff': 1.1607736993823892e-10, 'speedup_total_model': 18.66933656001699, 'speedup_estimate': 16.328102888536144, 'consistent': True}
  - cpu: status=Completed, total_model_s=6.083, estimate_s=4.654, loglike=-103327.5265186956
  - cuda: status=Completed, total_model_s=0.326, estimate_s=0.285, loglike=-103327.52651869561
- `torch_device_nl_500k` comparison: {'objective_diff': 0.0, 'relative_objective_diff': 0.0, 'max_param_diff': 1.645350522494482e-13, 'speedup_total_model': 29.4264874208422, 'speedup_estimate': 26.27164525206823, 'consistent': True}
  - cpu: status=Completed, total_model_s=17.942, estimate_s=13.812, loglike=-206952.2170715286
  - cuda: status=Completed, total_model_s=0.610, estimate_s=0.526, loglike=-206952.2170715286
- `torch_device_mixl_5k` comparison: {'objective_diff': 4.547473508864641e-13, 'relative_objective_diff': 1.6768225019195326e-16, 'max_param_diff': 1.360883350454145e-10, 'speedup_total_model': 43.91001440241408, 'speedup_estimate': 44.73698957616197, 'consistent': True}
  - cpu: status=Completed, total_model_s=23.910, estimate_s=23.770, loglike=-2711.9587813611447
  - cuda: status=Completed, total_model_s=0.545, estimate_s=0.531, loglike=-2711.958781361145
- `torch_device_mixl_10k` comparison: {'objective_diff': 8.276401786133647e-11, 'relative_objective_diff': 1.4928021124337923e-14, 'max_param_diff': 1.368047342076295e-10, 'speedup_total_model': 56.26333195263024, 'speedup_estimate': 56.9979989096162, 'consistent': True}
  - cpu: status=Completed, total_model_s=46.876, estimate_s=46.590, loglike=-5544.2055696452635
  - cuda: status=Completed, total_model_s=0.833, estimate_s=0.817, loglike=-5544.205569645181
- `torch_device_mixl_25k` comparison: {'objective_diff': 0.0, 'relative_objective_diff': 0.0, 'max_param_diff': 2.1132401384349464e-12, 'speedup_total_model': 71.04203256644715, 'speedup_estimate': 71.30784933687208, 'consistent': True}
  - cpu: status=Completed, total_model_s=129.483, estimate_s=128.409, loglike=-14095.854842965113
  - cuda: status=Completed, total_model_s=1.823, estimate_s=1.801, loglike=-14095.854842965113
