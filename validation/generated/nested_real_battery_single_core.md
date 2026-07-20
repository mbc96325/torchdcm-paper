# Real-data Nested Logit Benchmark

Runtimes report estimation plus covariance on one logical CPU.

| Data | Model | N | TorchDCM | Biogeme | Apollo | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Swissmetro | Nested logit | 10719 | 0.237 | 5.806 | 1.127 | Yes |
| LPMC London | Nested logit | 81086 | 0.878 | 25.374 | 14.051 | Yes |
| NHTS 2022 | Nested logit | 27375 | 0.667 | 112.485 | 10.879 | Yes |
| Parking Spain | Nested logit | 1576 | 0.162 | 6.646 | 0.473 | Yes |
| Airline itinerary | Nested logit | 3609 | 0.200 | 6.720 | 0.468 | Yes |
| Catsup | Nested logit | 2798 | 0.189 | 10.810 | 0.424 | Yes |
| Cracker | Nested logit | 3292 | 0.172 | 10.820 | 0.463 | Yes |
| Electricity | Nested logit | 4308 | 0.216 | 20.676 | 0.585 | Yes |
| Fishing | Nested logit | 1182 | 0.190 | 9.756 | 0.391 | Yes |
| HC | Nested logit | 250 | 0.206 | 34.091 | 0.407 | Yes |
| Heating | Nested logit | 900 | 0.192 | 10.250 | 0.390 | Yes |
| Mode | Nested logit | 453 | 0.164 | 9.928 | 0.376 | Yes |

## Nest specifications

- `swissmetro_nested`: PUBLIC=TRAIN,SM; PRIVATE=CAR
- `lpmc_nested`: ACTIVE=walk,cycle; MOTORIZED=pt,drive
- `nhts_2022_nested`: ACTIVE=WALK,BIKE; MOTORIZED=AUTO,TRANSIT,OTHER
- `parking_nested`: FACILITY=FSP,PSP; PUP_NEST=PUP
- `airline_nested`: ALT12=ALT1,ALT2; ALT3_NEST=ALT3
- `mlogit_catsup_nested`: HEINZ=heinz41,heinz32,heinz28; HUNTS=hunts32
- `mlogit_cracker_nested`: NATIONAL=sunshine,kleebler,nabisco; PRIVATE=private
- `mlogit_electricity_nested`: PLAN_12=1,2; PLAN_34=3,4
- `mlogit_fishing_nested`: SHORE=beach,pier; BOAT=boat,charter
- `mlogit_hc_nested`: CURRENT=gcc,ecc,erc; NEW=gc,ec,er; HPC_NEST=hpc
- `mlogit_heating_nested`: GAS=gc,gr; ELECTRIC=ec,er; HEATPUMP=hp
- `mlogit_mode_nested`: PRIVATE=car,carpool; TRANSIT=bus,rail
