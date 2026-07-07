# Mixed logit TorchDCM CPU/CUDA scaling note

Remote host: `baichuan-mo`

Dataset/specification: Swissmetro, full estimation, 10,719 observations, observation-level mixed logit, zero/shared initial values.

Torch device check: PyTorch `2.12.1+cu130`, CUDA available, `NVIDIA GeForce RTX 5090`.

## Results

| Case | Draws | RC | Device | Total s | Estimate s | Covariance s | Loglike |
|---|---:|---:|---|---:|---:|---:|---:|
| Swissmetro mixed | 32 | 1 | CUDA | 0.193 | 0.166 | 0.026 | -8566.953248 |
| Swissmetro mixed | 32 | 2 | CUDA | 0.224 | 0.196 | 0.028 | -8392.698768 |
| Swissmetro mixed | 128 | 2 | CPU | 0.979 | 0.713 | 0.266 | -8371.570710 |
| Swissmetro mixed | 128 | 2 | CUDA | 0.217 | 0.185 | 0.032 | -8371.570710 |
| Swissmetro mixed | 512 | 2 | CPU | 5.463 | 3.801 | 1.662 | -8369.698307 |
| Swissmetro mixed | 512 | 2 | CUDA | 0.307 | 0.248 | 0.059 | -8369.698307 |

## Takeaway

TorchDCM's extra mixed-logit acceleration mainly comes from moving the simulated likelihood and Hessian/covariance calculations to CUDA. The advantage grows with the number of draws: 2RC/128 draws is about 4.5x faster on CUDA than CPU, and 2RC/512 draws is about 17.8x faster. The 128-draw full comparison also remains aligned with Biogeme shared-draw estimation, while Biogeme and Apollo take 76.876s and 40.144s respectively on the same case.
