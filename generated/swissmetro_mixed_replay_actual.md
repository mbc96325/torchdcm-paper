# Swissmetro Mixed-Logit Shared-Draw Replay Rerun

Remote run on `baichuan-mo` using the filtered full Swissmetro data
(`N=10,719`), 64 shared antithetic draws, and panel likelihood.

These rows validate mixed-logit likelihood/probability kernels under identical
parameters and identical simulation draws. They are not full simulated
maximum-likelihood estimation rows.

| Case | Model | N | K | TorchDCM | Biogeme | Apollo | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Swissmetro | Mixed logit replay | 10719 | 5 | 0.013 | 14.172 | 0.671 | Yes |
| Swissmetro | WTP mixed replay | 10719 | 5 | 0.014 | 14.873 | 0.646 | Yes |

The maximum probability difference is `6.661e-16` for both Biogeme and Apollo
replay checks.

