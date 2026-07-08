# Complete Real-data Mixed Logit Battery

All cross-estimator rows use CPU for TorchDCM and Biogeme. Each runnable model uses 2-4 independent normal random coefficients selected from observed-variable coefficients first, then ASC terms only when needed. Skipped rows are recorded rather than hard-run when the aligned specification is unsupported or the remote process is killed.

| case | N | RC | TorchDCM s | Biogeme s | LL diff | Param diff | Prob diff | Consistent? |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| swissmetro | 10719 | B_TIME, B_COST | 0.217 | 16.627 | -2.90e-06 | 2.83e-04 | 3.11e-05 | Yes |
| airline | 3609 | B_TRIP_TIME, B_FARE, B_LEGROOM | 0.069 | 22.599 | 1.96e-10 | 3.99e-06 | 4.70e-07 | Yes |
| parking | 1576 | B_ACCESS_TIME, B_SEARCH_TIME, B_FEE | 0.099 | 23.789 | -4.98e-03 | 3.81e+00 | 2.60e-02 | No |
| telephone | 434 | B_COST, ASC_A2 | 0.032 | 20.069 | -1.37e-01 | 2.29e+00 | 4.48e-02 | No |
| lpmc | 81086 | B_TIME, B_COST | 7.894 | 31.805 | -1.14e+02 | 1.72e+00 | 6.94e-02 | No |
| mlogit_car | NA | skipped | NA | NA | NA | NA | NA | No |
| mlogit_catsup | 2798 | B_DISP, B_FEAT, B_PRICE | 0.100 | 29.881 | -7.40e-06 | 5.98e-03 | 2.10e-04 | Yes |
| mlogit_cracker | 3292 | B_DISP, B_FEAT, B_PRICE | 0.118 | 262.630 | -2.20e+01 | 1.07e+00 | 1.30e-01 | No |
| mlogit_electricity | 4308 | B_PF, B_CL, B_LOC, B_WK | 0.431 | 57.346 | NA | NA | NA | No |
| mlogit_fishing | 1182 | B_PRICE, B_CATCH | 0.077 | 19.435 | -5.62e+00 | 7.81e-01 | 6.03e-02 | No |
| mlogit_game | NA | skipped | NA | NA | NA | NA | NA | No |
| mlogit_game2 | NA | skipped | NA | NA | NA | NA | NA | No |
| mlogit_hc | 250 | B_ICH, B_OCH | 0.152 | 34.262 | -6.37e+01 | 9.17e-03 | 4.66e-01 | No |
| mlogit_heating | 900 | B_IC, B_OC | 0.149 | 23.032 | 1.11e-09 | 4.25e-08 | 1.58e-10 | Yes |
| mlogit_japanese_fdi | NA | skipped | NA | NA | NA | NA | NA | No |
| mlogit_mode | 453 | B_COST, B_TIME | 0.237 | 18.436 | 3.18e-10 | 6.30e-07 | 9.11e-07 | Yes |
| mlogit_modecanada | 4324 | B_COST, B_IVT, B_OVT, B_FREQ | 38.682 | 54.785 | 1.06e-09 | 4.81e-10 | 5.20e-09 | Yes |
| mlogit_nox | 632 | B_POST, B_VCOST, B_KCOST | 0.058 | 156.158 | -6.26e-08 | 1.17e-04 | 8.66e-06 | Yes |
| mlogit_risky_transport | 1793 | B_COST, B_RISK, B_SEATS, B_NOISE | 11.101 | 60.486 | -2.50e+00 | 2.91e+00 | 7.48e-02 | No |
| mlogit_train | 2929 | B_PRICE, B_TIME, B_CHANGE, B_COMFORT | 0.332 | 30.663 | 1.34e+02 | 5.79e+02 | 5.00e-01 | No |

## Specifications

- `swissmetro`: random coefficients = B_TIME, B_COST; all free parameters = ASC_TRAIN, B_TIME, B_COST, ASC_CAR.
  - TRAIN: ASC_TRAIN + random(B_TIME*time_train) + random(B_COST*cost_train)
  - SM: random(B_TIME*time_sm) + random(B_COST*cost_sm)
  - CAR: ASC_CAR + random(B_TIME*time_car) + random(B_COST*cost_car)
- `airline`: random coefficients = B_TRIP_TIME, B_FARE, B_LEGROOM; all free parameters = B_TRIP_TIME, B_FARE, B_LEGROOM, ASC_ALT2, ASC_ALT3.
  - ALT1: random(B_TRIP_TIME*trip_time_alt1) + random(B_FARE*fare_alt1) + random(B_LEGROOM*legroom_alt1)
  - ALT2: ASC_ALT2 + random(B_TRIP_TIME*trip_time_alt2) + random(B_FARE*fare_alt2) + random(B_LEGROOM*legroom_alt2)
  - ALT3: ASC_ALT3 + random(B_TRIP_TIME*trip_time_alt3) + random(B_FARE*fare_alt3) + random(B_LEGROOM*legroom_alt3)
- `parking`: random coefficients = B_ACCESS_TIME, B_SEARCH_TIME, B_FEE; all free parameters = B_ACCESS_TIME, B_SEARCH_TIME, B_FEE, ASC_PSP, ASC_PUP.
  - FSP: random(B_ACCESS_TIME*access_time_fsp) + random(B_SEARCH_TIME*search_time_fsp) + random(B_FEE*fee_fsp)
  - PSP: ASC_PSP + random(B_ACCESS_TIME*access_time_psp) + random(B_SEARCH_TIME*search_time_psp) + random(B_FEE*fee_psp)
  - PUP: ASC_PUP + random(B_ACCESS_TIME*access_time_pup) + random(B_SEARCH_TIME*search_time_pup) + random(B_FEE*fee_pup)
- `telephone`: random coefficients = B_COST, ASC_A2; all free parameters = B_COST, ASC_A2, ASC_A3, ASC_A4, ASC_A5.
  - A1: random(B_COST*cost_a1)
  - A2: random(ASC_A2) + random(B_COST*cost_a2)
  - A3: ASC_A3 + random(B_COST*cost_a3)
  - A4: ASC_A4 + random(B_COST*cost_a4)
  - A5: ASC_A5 + random(B_COST*cost_a5)
- `lpmc`: random coefficients = B_TIME, B_COST; all free parameters = B_TIME, B_COST, ASC_CYCLE, ASC_PT, ASC_DRIVE.
  - walk: random(B_TIME*time_walk) + random(B_COST*cost_walk)
  - cycle: ASC_CYCLE + random(B_TIME*time_cycle) + random(B_COST*cost_cycle)
  - pt: ASC_PT + random(B_TIME*time_pt) + random(B_COST*cost_pt)
  - drive: ASC_DRIVE + random(B_TIME*time_drive) + random(B_COST*cost_drive)
- `mlogit_car` skipped: RuntimeError: Error in if (abs(x - oldx) < ftol) { : 
  missing value where TRUE/FALSE needed
Calls: fit_long -> mlogit -> eval -> eval -> mlogit.optim
Execution halted
- `mlogit_catsup`: random coefficients = B_DISP, B_FEAT, B_PRICE; all free parameters = B_DISP, B_FEAT, B_PRICE.
  - heinz41: random(B_DISP*disp_heinz41_1) + random(B_FEAT*feat_heinz41_1) + random(B_PRICE*price_heinz41_1)
  - heinz32: random(B_DISP*disp_heinz32_2) + random(B_FEAT*feat_heinz32_2) + random(B_PRICE*price_heinz32_2)
  - heinz28: random(B_DISP*disp_heinz28_3) + random(B_FEAT*feat_heinz28_3) + random(B_PRICE*price_heinz28_3)
  - hunts32: random(B_DISP*disp_hunts32_4) + random(B_FEAT*feat_hunts32_4) + random(B_PRICE*price_hunts32_4)
- `mlogit_cracker`: random coefficients = B_DISP, B_FEAT, B_PRICE; all free parameters = B_DISP, B_FEAT, B_PRICE.
  - sunshine: random(B_DISP*disp_sunshine_1) + random(B_FEAT*feat_sunshine_1) + random(B_PRICE*price_sunshine_1)
  - kleebler: random(B_DISP*disp_kleebler_2) + random(B_FEAT*feat_kleebler_2) + random(B_PRICE*price_kleebler_2)
  - nabisco: random(B_DISP*disp_nabisco_3) + random(B_FEAT*feat_nabisco_3) + random(B_PRICE*price_nabisco_3)
  - private: random(B_DISP*disp_private_4) + random(B_FEAT*feat_private_4) + random(B_PRICE*price_private_4)
- `mlogit_electricity`: random coefficients = B_PF, B_CL, B_LOC, B_WK; all free parameters = B_PF, B_CL, B_LOC, B_WK, B_TOD, B_SEAS.
  - 1: random(B_PF*pf_alt_1_1) + random(B_CL*cl_alt_1_1) + random(B_LOC*loc_alt_1_1) + random(B_WK*wk_alt_1_1) + B_TOD*tod_alt_1_1 + B_SEAS*seas_alt_1_1
  - 2: random(B_PF*pf_alt_2_2) + random(B_CL*cl_alt_2_2) + random(B_LOC*loc_alt_2_2) + random(B_WK*wk_alt_2_2) + B_TOD*tod_alt_2_2 + B_SEAS*seas_alt_2_2
  - 3: random(B_PF*pf_alt_3_3) + random(B_CL*cl_alt_3_3) + random(B_LOC*loc_alt_3_3) + random(B_WK*wk_alt_3_3) + B_TOD*tod_alt_3_3 + B_SEAS*seas_alt_3_3
  - 4: random(B_PF*pf_alt_4_4) + random(B_CL*cl_alt_4_4) + random(B_LOC*loc_alt_4_4) + random(B_WK*wk_alt_4_4) + B_TOD*tod_alt_4_4 + B_SEAS*seas_alt_4_4
- `mlogit_fishing`: random coefficients = B_PRICE, B_CATCH; all free parameters = B_PRICE, B_CATCH.
  - beach: random(B_PRICE*price_beach_1) + random(B_CATCH*catch_beach_1)
  - pier: random(B_PRICE*price_pier_2) + random(B_CATCH*catch_pier_2)
  - boat: random(B_PRICE*price_boat_3) + random(B_CATCH*catch_boat_3)
  - charter: random(B_PRICE*price_charter_4) + random(B_CATCH*catch_charter_4)
- `mlogit_game` skipped: RuntimeError: Need at least two observed variables for 2+ random coefficients; found ['own'].
- `mlogit_game2` skipped: RuntimeError: Need at least two observed variables for 2+ random coefficients; found ['own'].
- `mlogit_hc`: random coefficients = B_ICH, B_OCH; all free parameters = B_ICH, B_OCH.
  - gcc: random(B_ICH*ich_gcc_1) + random(B_OCH*och_gcc_1)
  - ecc: random(B_ICH*ich_ecc_2) + random(B_OCH*och_ecc_2)
  - erc: random(B_ICH*ich_erc_3) + random(B_OCH*och_erc_3)
  - hpc: random(B_ICH*ich_hpc_4) + random(B_OCH*och_hpc_4)
  - gc: random(B_ICH*ich_gc_5) + random(B_OCH*och_gc_5)
  - ec: random(B_ICH*ich_ec_6) + random(B_OCH*och_ec_6)
  - er: random(B_ICH*ich_er_7) + random(B_OCH*och_er_7)
- `mlogit_heating`: random coefficients = B_IC, B_OC; all free parameters = B_IC, B_OC.
  - gc: random(B_IC*ic_gc_1) + random(B_OC*oc_gc_1)
  - gr: random(B_IC*ic_gr_2) + random(B_OC*oc_gr_2)
  - ec: random(B_IC*ic_ec_3) + random(B_OC*oc_ec_3)
  - er: random(B_IC*ic_er_4) + random(B_OC*oc_er_4)
  - hp: random(B_IC*ic_hp_5) + random(B_OC*oc_hp_5)
- `mlogit_japanese_fdi` skipped: Remote process was killed during 4RC full-estimation/JAX compilation; not rerun to avoid hard-running an unstable heavy specification.
- `mlogit_mode`: random coefficients = B_COST, B_TIME; all free parameters = B_COST, B_TIME.
  - car: random(B_COST*cost_car_1) + random(B_TIME*time_car_1)
  - carpool: random(B_COST*cost_carpool_2) + random(B_TIME*time_carpool_2)
  - bus: random(B_COST*cost_bus_3) + random(B_TIME*time_bus_3)
  - rail: random(B_COST*cost_rail_4) + random(B_TIME*time_rail_4)
- `mlogit_modecanada`: random coefficients = B_COST, B_IVT, B_OVT, B_FREQ; all free parameters = B_COST, B_IVT, B_OVT, B_FREQ.
  - train: random(B_COST*cost_train_1) + random(B_IVT*ivt_train_1) + random(B_OVT*ovt_train_1) + random(B_FREQ*freq_train_1)
  - car: random(B_COST*cost_car_2) + random(B_IVT*ivt_car_2) + random(B_OVT*ovt_car_2) + random(B_FREQ*freq_car_2)
  - bus: random(B_COST*cost_bus_3) + random(B_IVT*ivt_bus_3) + random(B_OVT*ovt_bus_3) + random(B_FREQ*freq_bus_3)
  - air: random(B_COST*cost_air_4) + random(B_IVT*ivt_air_4) + random(B_OVT*ovt_air_4) + random(B_FREQ*freq_air_4)
- `mlogit_nox`: random coefficients = B_POST, B_VCOST, B_KCOST; all free parameters = B_POST, B_VCOST, B_KCOST.
  - 1: random(B_POST*post_alt_1_1) + random(B_VCOST*vcost_alt_1_1) + random(B_KCOST*kcost_alt_1_1)
  - 2: random(B_POST*post_alt_2_2) + random(B_VCOST*vcost_alt_2_2) + random(B_KCOST*kcost_alt_2_2)
  - 3: random(B_POST*post_alt_3_3) + random(B_VCOST*vcost_alt_3_3) + random(B_KCOST*kcost_alt_3_3)
  - 4: random(B_POST*post_alt_4_4) + random(B_VCOST*vcost_alt_4_4) + random(B_KCOST*kcost_alt_4_4)
  - 5: random(B_POST*post_alt_5_5) + random(B_VCOST*vcost_alt_5_5) + random(B_KCOST*kcost_alt_5_5)
  - 6: random(B_POST*post_alt_6_6) + random(B_VCOST*vcost_alt_6_6) + random(B_KCOST*kcost_alt_6_6)
  - 7: random(B_POST*post_alt_7_7) + random(B_VCOST*vcost_alt_7_7) + random(B_KCOST*kcost_alt_7_7)
  - 8: random(B_POST*post_alt_8_8) + random(B_VCOST*vcost_alt_8_8) + random(B_KCOST*kcost_alt_8_8)
  - 9: random(B_POST*post_alt_9_9) + random(B_VCOST*vcost_alt_9_9) + random(B_KCOST*kcost_alt_9_9)
  - 10: random(B_POST*post_alt_10_10) + random(B_VCOST*vcost_alt_10_10) + random(B_KCOST*kcost_alt_10_10)
  - 11: random(B_POST*post_alt_11_11) + random(B_VCOST*vcost_alt_11_11) + random(B_KCOST*kcost_alt_11_11)
  - 12: random(B_POST*post_alt_12_12) + random(B_VCOST*vcost_alt_12_12) + random(B_KCOST*kcost_alt_12_12)
  - 13: random(B_POST*post_alt_13_13) + random(B_VCOST*vcost_alt_13_13) + random(B_KCOST*kcost_alt_13_13)
  - 14: random(B_POST*post_alt_14_14) + random(B_VCOST*vcost_alt_14_14) + random(B_KCOST*kcost_alt_14_14)
  - 15: random(B_POST*post_alt_15_15) + random(B_VCOST*vcost_alt_15_15) + random(B_KCOST*kcost_alt_15_15)
- `mlogit_risky_transport`: random coefficients = B_COST, B_RISK, B_SEATS, B_NOISE; all free parameters = B_COST, B_RISK, B_SEATS, B_NOISE, B_CROWDNESS, B_CONVLOC, B_CLIENTELE.
  - WaterTaxi: random(B_COST*cost_watertaxi_1) + random(B_RISK*risk_watertaxi_1) + random(B_SEATS*seats_watertaxi_1) + random(B_NOISE*noise_watertaxi_1) + B_CROWDNESS*crowdness_watertaxi_1 + B_CONVLOC*convloc_watertaxi_1 + B_CLIENTELE*clientele_watertaxi_1
  - Ferry: random(B_COST*cost_ferry_2) + random(B_RISK*risk_ferry_2) + random(B_SEATS*seats_ferry_2) + random(B_NOISE*noise_ferry_2) + B_CROWDNESS*crowdness_ferry_2 + B_CONVLOC*convloc_ferry_2 + B_CLIENTELE*clientele_ferry_2
  - Hovercraft: random(B_COST*cost_hovercraft_3) + random(B_RISK*risk_hovercraft_3) + random(B_SEATS*seats_hovercraft_3) + random(B_NOISE*noise_hovercraft_3) + B_CROWDNESS*crowdness_hovercraft_3 + B_CONVLOC*convloc_hovercraft_3 + B_CLIENTELE*clientele_hovercraft_3
  - Helicopter: random(B_COST*cost_helicopter_4) + random(B_RISK*risk_helicopter_4) + random(B_SEATS*seats_helicopter_4) + random(B_NOISE*noise_helicopter_4) + B_CROWDNESS*crowdness_helicopter_4 + B_CONVLOC*convloc_helicopter_4 + B_CLIENTELE*clientele_helicopter_4
- `mlogit_train`: random coefficients = B_PRICE, B_TIME, B_CHANGE, B_COMFORT; all free parameters = B_PRICE, B_TIME, B_CHANGE, B_COMFORT.
  - A: random(B_PRICE*price_a_1) + random(B_TIME*time_a_1) + random(B_CHANGE*change_a_1) + random(B_COMFORT*comfort_a_1)
  - B: random(B_PRICE*price_b_2) + random(B_TIME*time_b_2) + random(B_CHANGE*change_b_2) + random(B_COMFORT*comfort_b_2)
