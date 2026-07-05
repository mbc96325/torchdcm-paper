# Real-data Nested Logit Benchmark

| Data | Model | N | TorchDCM | Biogeme | Apollo | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Swissmetro | Nested logit | 10719 | 0.068 | 4.246 | 2.089 | Yes |
| LPMC London | Nested logit | 81086 | 0.325 | 22.918 | 22.769 | Yes |
| NHTS 2022 | Nested logit | 27375 | 0.377 | 66.199 | 13.118 | Yes |
| Parking Spain | Nested logit | 1576 | 0.033 | 5.472 | 1.248 | Yes |

## Nest specifications

- `swissmetro_nested`: PUBLIC=TRAIN,SM; PRIVATE=CAR
- `lpmc_nested`: ACTIVE=walk,cycle; MOTORIZED=pt,drive
- `nhts_2022_nested`: ACTIVE=WALK,BIKE; MOTORIZED=AUTO,TRANSIT,OTHER
- `parking_nested`: FACILITY=FSP,PSP; PUP_NEST=PUP
