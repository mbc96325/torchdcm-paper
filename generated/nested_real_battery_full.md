# Real-data Nested Logit Benchmark

| Data | Model | N | TorchDCM | Biogeme | Apollo | Consistent? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Swissmetro | Nested logit | 10719 | 0.068 | 4.246 | 2.089 | Yes |
| LPMC London | Nested logit | 81086 | 0.325 | 22.918 | 22.769 | Yes |
| NHTS 2022 | Nested logit | 27375 | 0.377 | 66.199 | 13.118 | Yes |
| Parking Spain | Nested logit | 1576 | 0.033 | 5.472 | 1.248 | Yes |
| Airline itinerary | Nested logit | 3609 | 0.089 | 6.364 | 1.255 | Yes |
| Catsup | Nested logit | 2798 | 0.071 | 8.782 | 1.209 | Yes |
| Cracker | Nested logit | 3292 | 0.054 | 8.988 | 1.256 | Yes |
| Electricity | Nested logit | 4308 | 0.118 | 18.301 | 1.386 | Yes |
| Fishing | Nested logit | 1182 | 0.095 | 7.558 | 1.148 | Yes |
| HC | Nested logit | 250 | 0.076 | 30.457 | 1.153 | Yes |
| Heating | Nested logit | 900 | 0.087 | 7.588 | 1.137 | Yes |
| Mode | Nested logit | 453 | 0.043 | 7.045 | 1.139 | Yes |

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
