# Reservoir head sensitivity data

Prepared: 2026-10-17 (Asia/Shanghai).

This supplement contains a five-reservoir **assumed sensitivity scenario** derived from the existing project inputs. It contains no measured tailwater, calibrated loss curve, or externally verified head relationship. The companion CSV marks every row as `assumed_sensitivity_only`.

## Scope and assumptions

The supplied `typical_net_head_m` values do not identify their reference forebay elevation, tailwater elevation, or operating discharge. The working reference elevation is therefore assumed to be the midpoint of the configured operating elevation range. This choice centers the head sensitivity around the supplied typical head. It is not evidence that the typical head was measured at this elevation, nor is it a long-term operating average or a calibration.

For each reservoir, with all elevations and heads in metres:

`Z_ref = (Z_min + Z_max) / 2`

`H_ref = supplied typical_net_head_m`

`B_assumed = Z_ref - H_ref`

`H_scenario(Z) = H_ref + (Z - Z_ref) = Z - B_assumed`

`B_assumed` is a constant combined tailwater-plus-loss offset. It must not be interpreted as a measured tailwater elevation. Separate tailwater and hydraulic losses remain unknown; losses have not been assigned a zero value. The head slope of 1 assumes that the combined offset is insensitive to discharge and downstream conditions.

The CSV endpoints are `H_scenario(Z_min)` and `H_scenario(Z_max)`. They are scenario values, not verified operating-head limits. Validity is restricted to the configured operating elevation interval. The implementation uses the existing storage-elevation curve to convert beginning/end storage into forebay elevations, then takes their arithmetic mean. This is a daily time-step approximation, not an additional measured input or a resolved within-day water-level trajectory.

Efficiency remains the existing constant 0.90. It is a model assumption, not a newly calibrated value. Installed capacities, nominal turbine-flow limits, total-release limits, hydrology, initial storages and reward normalization are independent of this data supplement. Permitted next-state water levels are supplied separately in `five_reservoir_happo_input_bounds_1965_2024.csv`; no final-storage target is imposed.

## CSV fields

| Field                                     | Meaning and provenance                                                                                                                                                   |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `reservoir_id`                            | Existing hydraulic order: WDD, BHT, XLD, XJB, THR.                                                                                                                       |
| `data_status`                             | `assumed_sensitivity_only`; no row is measured or calibrated.                                                                                                            |
| `operating_level_min_m`                   | Lower elevation from the existing environment configuration, checked against the storage-elevation curve.                                                                |
| `operating_level_max_m`                   | Upper elevation from that configuration, checked against the curve and all 60 calendar-year-start storage records. The full curve may extend above this operating limit. |
| `assumed_reference_forebay_level_m`       | Arithmetic midpoint of the two operating elevations; an explicitly assumed anchor.                                                                                       |
| `reference_net_head_m`                    | Existing `reservoir_parameters.csv:typical_net_head_m`, paired with the assumed anchor only for this scenario.                                                           |
| `assumed_effective_tailwater_plus_loss_m` | Derived `Z_ref - H_ref`; not a separately identified physical tailwater or loss.                                                                                         |
| `head_slope_m_per_m`                      | Assumed 1.0 for the constant combined-boundary scenario.                                                                                                                 |
| `scenario_net_head_min_m`                 | Derived head at the lower operating elevation.                                                                                                                           |
| `scenario_net_head_max_m`                 | Derived head at the upper operating elevation.                                                                                                                           |
| `efficiency`                              | Existing constant efficiency, stored as a fraction.                                                                                                                      |

## Data still missing

- Tailwater-elevation versus discharge relationships, including the discharge definition and any downstream-backwater dependence, for all five reservoirs.
- Hydraulic losses as a justified constant or a relationship with turbine/powerhouse flow. Unit-level data must identify unit counts and aggregation assumptions.
- The actual operating conditions associated with each supplied typical net head, including reference forebay elevation.
- Confirmation of the vertical datum and any gauge-zero offsets. The derived combined offsets inherit the source forebay coordinate system; their physical datum has not been independently verified.
- Observed forebay/tailwater, turbine flow, spill, and generation records for calibration and independent physical verification.
- Efficiency and head-dependent turbine-capacity curves if a subsequent model is to represent those effects. They are not required for this initial constant-efficiency scenario.

These missing quantities cannot be recovered uniquely from daily inflows, storage-elevation curves, or installed capacity. No discharge-dependent tailwater/loss curve has been fabricated. A measured or calibrated relationship should replace this sensitivity anchor before making claims of plant-specific accuracy or paper reproduction.

## Integration status

The loader now consumes and validates this CSV when `head_model: storage_sensitivity` is selected (the environment configuration default). Missing or inconsistent scenario inputs raise errors; there is no automatic fallback. Fixed-head comparisons remain available with `--head-model fixed`, which does not require the supplement. Both startup output and evaluation rows identify the selected head model and the assumption status.

Each transition computes storage and forebay elevations before calculating `H_day = H_ref + (Z_start + Z_end)/2 - Z_ref`. The turbine-flow cap is recomputed as `min(nominal_turbine_cap, installed_capacity / (rho*g*eta*H_day/1e6))`; remaining release is spill. Energy is daily mean power times 24 hours. The installed-power limit is therefore enforced at the modeled daily mean head; this does not establish compliance with an unmodeled instantaneous within-day power limit. Training and evaluation use the same generation routine and selected head model throughout the complete 1,825-day episode. Initial storage is loaded at the episode start; storage and head carry across subsequent calendar-year boundaries.

Daily net head and effective turbine-flow capacity are available in environment transitions; evaluation additionally reports each reservoir's episode minimum, mean, and maximum modeled daily head. Efficiency remains fixed at 0.90. This adds storage-to-head feedback, not a calibrated efficiency curve or a discharge-dependent tailwater/loss relationship.

For future flow-dependent losses or efficiency, turbine flow and power may require a coupled solve. The current sensitivity scenario's constant combined offset does not require one. A historical head series alone would not provide storage-dependent feedback.

CSV values are numeric snapshots of the formulas above. Derived decimal values are rounded to six places to remove binary arithmetic artifacts; this does not indicate physical measurement precision.

## Local input provenance

Original source CSVs are preserved. Relative paths below are from the repository root. SHA-256 hashes identify the inputs at the time the supplement was prepared; the environment configuration has subsequently gained the head-model selection above without changing the elevation bounds or efficiency used to derive the CSV.

| Input                                                      | SHA-256                                                            |
| ---------------------------------------------------------- | ------------------------------------------------------------------ |
| `data/cascade_reservoir_new/reservoir_parameters.csv`      | `095a9ff1a0486ab716b069895370120e40304a72d06ff0c81d50e301104a528c` |
| `data/cascade_reservoir_new/reservoir_stage_storage.csv`   | `86f1ba09cee090002bc349a8d97377c76a94c1a782f5c589d2e2d0bea0411988` |
| `data/cascade_reservoir_new/initial_storage_1965_2024.csv` | `6b5feb8b7a5f77992ff0a7e5a2d046631203f28e1378b81b7222613b8304fe64` |
| `harl/configs/envs_cfgs/cascade_reservoir.yaml`            | `d7ff9afe573ae403c648f3b05bc474aa51174b5dbf54cf20fa2172f7d4f909f7` |
