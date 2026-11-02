# Five-reservoir HAPPO

Daily operation of the WDD, BHT, XLD, XJB and THR reservoir cascade, based on
PKU-MARL HARL. Five independent categorical actors use one centralized critic.
The environment resolves releases upstream to downstream with zero travel delay.

## Current experiment

- Data cover 1965--2024: **55 training years and five evaluation years**.
- Evaluation years, in sequence order: **1998, 1999, 1970, 1997, 2006**.
- Training years are the full complement; inflow scaling uses training data only.
- Each episode contains **1,825 daily decisions**, with **six parallel workers**.
- Each reservoir has **100 discrete actions** by default.
- Fixed training-only scenarios are evaluated at **Round 0 and after each update**.
- The final trained policy is evaluated after all training seeds finish.
- Training uses a reproducible schedule that targets balanced year-day coverage, with distinct windows within each parallel batch.
- Current service-threshold sensitivity: **50% guaranteed output and 70% ecological flow**, relative to the reference CSV.

The standard budget is 300 completed episodes per seed: 50 HAPPO updates and
547,500 daily environment transitions. Each update collects six complete
episodes under one frozen policy, giving buffers of shape `[1825, 6, ...]`.
Episode counts must be divisible by the number of workers.

## Launch from VS Code

```bash
cd /Users/hujingxin/Documents/Code_python/HARL-main

PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --episodes 300 --seeds 1 2 3 --gamma 1 --eval-interval 1 --diagnostics \
  --episode-log runs/five_year_periodic_eval_seeds123_run2.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

The episode horizon, worker count and action count already default to
1,825 / 6 / 100. They can be stated explicitly with
`--episode-days 1825 --parallel-envs 6 --action-bins 100`.
Use a new log filename for every run; existing files are never overwritten.
For console-only execution, replace `--plot-python ...` with `--no-plot`.

Launch the file-backed entry point above so spawned workers can re-import
the main module. Model persistence/resume is not implemented; start a fresh
run to use changed source or settings. Metrics are saved only when an
`--episode-log` path is supplied. With plotting enabled, all three figures are
automatically saved beside the log before their windows open:

```text
runs/five_year_periodic_eval_seeds123_run2_training.png
runs/five_year_periodic_eval_seeds123_run2_evaluator_return.png
runs/five_year_periodic_eval_seeds123_run2_violation_rates.png
```

`--no-plot` skips automatic figure creation and display; saved logs can still
be plotted or exported afterward.

`--episode-log` writes run, episode and update summaries to JSONL, and compact
daily metrics to one CSV per seed. For the command above the daily files are:

```text
runs/five_year_periodic_eval_seeds123_run2_seed1_steps.csv
runs/five_year_periodic_eval_seeds123_run2_seed2_steps.csv
runs/five_year_periodic_eval_seeds123_run2_seed3_steps.csv
```

Each completed daily transition produces one CSV row containing 15 scalar
values: phase, seed, episode, one-based episode step, source action date,
collected training steps, team reward, generation reward, guaranteed-output
penalty, ecological-flow penalty, generation (MWh), guaranteed-output shortfall
(MW and MWh), and ecological-flow shortfall (m3/s and m3).

Generation and physical deficits are summed over the five reservoirs. Penalties
use the weights and reservoir means from the reward function. Daily rewards
are undiscounted; apply `gamma**(episode_step-1)` to recover the discounted
episode total. CSV values use 10 significant digits; internal training and
summary calculations retain full precision. An unavailable service measurement
is an empty cell, while a measured zero deficit is written as zero.

Training and final evaluation share each seed's CSV, distinguished by the
`phase` column. Six environments produce six rows per collector step. A
300-episode seed produces 547,500 training rows and 1,825 evaluation rows. Each
row is flushed to disk as it is produced. The table can be opened in Excel or
filtered in Python using `seed`, `episode` and `episode_step`. Existing full
trajectory logs retain their original format; restart with a new output name
to use compact logging. Periodic evaluations add only summary records to JSONL,
so they do not expand the daily CSVs.

To reopen training, evaluator-return and violation-rate curves from a saved or currently running log:

```bash
/Library/Frameworks/Python.framework/Versions/3.9/bin/python3 examples/plot_cascade_parallel.py \
  --episode-log runs/five_year_periodic_eval_seeds123_run2.jsonl
```

To export all three figures without opening windows, add
`--output result/periodic_training.png --evaluator-output result/periodic_evaluator_return.png --violations-output result/periodic_violation_rates.png --no-show`.
With only `--output NAME.png`, the additional figures are named
`NAME_evaluator_return.png` and `NAME_violation_rates.png`. The plot command streams the JSONL summaries.
The evaluator and violation-rate figures are available as soon as Round 0 has been recorded;
training curves become available after the first update. Running logs show
available seeds and label their mean accordingly. Means use only common
observed rounds/steps. A failed evaluation remains a gap, including in the
seed mean; surviving seeds or windows are not silently reweighted.
Each subplot displays one across-seed mean curve and a shaded +/-1 sample
standard deviation band. Individual seed curves are hidden; all per-seed values
remain in the logs. No smoothing or standard-error conversion is applied.

Older periodic logs, including `five_year_periodic_eval_seeds123_run1.jsonl`,
contain penalty totals but omit the violation counts required for these rates.
The plotter still opens their training and return curves, with an explicit
warning that violation-rate curves are unavailable. Requesting
`--violations-output` for such a log reports an error. Penalty magnitudes cannot
reconstruct violation frequencies, and historical intermediate models were
not saved; start a new run to record rates at every evaluation round.

Check data, calendars and reset observations for all training windows and the
evaluation sequence without training:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --gamma 1 --check-data-only
```

`--skip-final-evaluation` reserves held-out data during tuning.
`--log-interval N` controls training diagnostic reporting in update rounds.
`--eval-interval N` controls periodic policy evaluation independently. It must
be positive and defaults to 1. Round 0 and the last update are always evaluated;
for example, a 50-update run with interval 7 evaluates rounds 0, 7, 14, 21, 28,
35, 42, 49 and 50.
`--evaluation-years` accepts exactly five unique years and uses the remaining
55 for training. `--help` describes the supported flags. The CLI exposes only
the current multi-year training pipeline. These checks do not predict whether
all later daily transitions will be feasible under a particular policy.

## Configuration and networks

`harl/configs/algos_cfgs/happo_cascade.yaml` contains the complete training,
network, optimizer and split settings. There is no separate collection overlay.
`harl/configs/envs_cfgs/cascade_reservoir.yaml` contains physical assumptions and
reward settings. CLI arguments override their corresponding YAML values.

| Setting | Default |
| --- | --- |
| Episode horizon / parallel environments | 1,825 days / 6 |
| Discrete actions per reservoir | 100 |
| Episodes per seed / seeds | 300 / 1, 2, 3, 4, 5 |
| Encoder / additional hidden head | `[64, 128, 256]` / 256 |
| Stored hidden widths / activation | `[64, 128, 256, 256]` / ReLU |
| Actor / critic learning rate | 0.0003 / 0.0003 |
| Actor / critic optimization epochs | 5 / 5 |
| Actor / critic minibatches per epoch | 1 / 1 |
| Gamma / GAE lambda | 0.99 / 0.95 |
| PPO clip / entropy coefficient | 0.2 / 0.001 |
| Maximum gradient norm | 0.5 |
| ValueNorm / shared advantage normalization | Enabled / enabled |
| Actor / joint KL limits | Disabled unless requested |
| Daily water-level limits | New action-date CSV, enabled |
| Periodic evaluation interval / scenarios | Every update / 3 fixed training-only windows |
| Training-window sampling | `coverage_balanced`; `uniform` restores the previous sampler |
| Guaranteed-output / ecological-flow threshold scale | 0.50 / 0.70 of reference CSV values |

The launch command explicitly selects gamma=1 for undiscounted five-year net
return. A separate lambda experiment can use `--gae-lambda 1`; it is not the
default. Episode duration, split terminology and action defaults were updated
without changing reward coefficients or optimizer hyperparameters.

With daily limits enabled, each actor observes 15 local/neighbor values and
the critic receives 38 centralized state values:

```text
Each actor: 15 -> 64 -> 128 -> 256 -> 256 -> 100 categorical logits
Critic:     38 -> 64 -> 128 -> 256 -> 256 -> 1 value
```

Each network has separate weights. Hidden layers use
`Linear -> ReLU -> LayerNorm`; logits and the final value remain linear.
`--no-daily-level-limits` selects global operating bounds only, with input
dimensions 11/18.

The actor's 15 features, in order, are:

1. Own storage fraction, standardized current external inflow and previous release fraction.
2. Immediate upstream storage fraction, standardized external inflow and previous release fraction.
3. Immediate downstream storage fraction and room below its next-state upper storage limit.
4. Own current-state lower and upper storage fractions.
5. Own next-state lower and upper water-level fractions.
6. Day-of-year sine/cosine and fraction of the episode remaining.

Water-level fractions use the reservoir's global operating elevation span;
storage fractions use its global operating storage span. The current-state
limits come from the previous executed action's CSV row, with global bounds
at reset. The centralized critic receives storage, current external inflows,
previous releases, three time features and all five reservoirs' four limit
features. The actor observation layout is `local_neighbors_next_day_limits_v2`.

## Episode calendar and evaluation

Training windows start on January 1 and contain exactly 1,825 consecutive
source days. Leap days are retained, so the horizon can finish one or two
days before its fifth calendar anniversary. Every executed day must belong
to training data; the default split provides 39 eligible windows.

Storage, previous releases, GAE and discount clocks carry across calendar
boundaries. Time-to-go is scaled by the complete episode horizon. Final storage
is subject to the final action's daily bounds; it need not equal initial storage.

Periodic evaluation measures the initial policy at Round 0 and the updated
policy after each configured interval. Under the default split, fixed scenarios
are **2007--2011 (dry), 1991--1995 (median), and 1973--1977 (wet)**. Each contains
exactly 1,825 consecutive days, ending December 30 of its fifth source year.
The three scenarios are selected once by minimum, median and maximum mean
total external inflow among complete training-only windows, with start year
breaking ties. Selection uses no RNG. Their source dates, inflow hashes and
initial storage/levels are recorded in `run_configuration.periodic_evaluation`.
All seeds and rounds use those same scenarios and initial conditions. Labels
describe training-window flow ranks, not the held-out year classifications.

Each scenario uses a fresh environment and deterministic argmax actions from
all five actors under `torch.no_grad()`. Random-number states and every actor
module's training/evaluation mode are restored afterward. Evaluation samples
never enter collection buffers or optimizer updates. Training progress counts
only collector steps. These diagnostics do not rank/select policies and do not
use any held-out source days.

One `periodic_evaluation` JSONL record per seed/round preserves the three
individual window results plus their arithmetic mean: `evaluator_return`,
`generation_MWh`/`generation_TWh`, `guaranteed_output_penalty`,
`ecological_flow_penalty`, `final_storage_m3` and `final_level_m`. Storage/level
vectors follow WDD/BHT/XLD/XJB/THR order. Return and penalties use the unchanged
configured discount and reward formula. The record also identifies
`evaluation_round`, `seed`, `training_steps`, status and successful-window count.
Physical infeasibility is recorded with its window, failed date/step and reason;
partial metrics are labelled separately. If any window fails, the aggregate
metrics are null and the evaluator curve leaves a gap. Other windows are still
checked and training continues. Unexpected programming errors propagate.

The additional unsmoothed figure is titled **Evaluator Return vs. Evaluation
Round**. Each seed's recorded return is the mean over its three fixed scenarios;
the displayed curve and band show only the across-seed mean and +/-1 sample SD. The
default 50-update experiment records rounds 0--50. This measures performance
and does not impose convergence or alter rewards to obtain a plateau.

A separate two-panel figure shows **Ecological-flow violation rate** and
**Guaranteed-output violation rate** against evaluation round. Each rate is
the number of reservoir-days with a positive shortfall divided by all
reservoir-days with an active requirement, pooled over the three fixed windows.
For the default five active reservoirs, each seed/round has a denominator of
`3 * 1825 * 5 = 27,375` reservoir-days per constraint. Counts use the existing
shortfall tolerance and are undiscounted, independent of gamma and penalty
weights. This is violation frequency, not relative shortfall severity or the
fraction of days with any reservoir violating a requirement.

Periodic records and individual windows now retain
`guaranteed_output_violation_rate` and `ecological_flow_violation_rate` as
fractions in [0, 1], their `*_violation_count` and
`*_active_reservoir_days` fields, and per-window `service_shortfalls` summaries.
The figure converts rates to percent and shows one across-seed mean curve
and +/-1 sample SD without smoothing. Failed windows leave aggregate rates
null and curve gaps; no active requirement also yields a null rate.

Final independent evaluation joins the five representative years in their configured order,
carries storage between them, and uses the same episode horizon. Their forcing
and date-specific daily bounds are retained. This is a constructed representative sequence,
rather than a continuous historical record. The evaluator scores the final
policy without ranking intermediate policies. A physical failure is logged
explicitly with partial progress before the existing final-evaluation error is
re-raised; successful final evaluation behavior is unchanged.

`--episode-days` also supports explicitly shortened collection checks. Periodic
scenarios still retain their full 1,825-day horizon. The collector checks'
actual dates and horizon are recorded; their metrics do not describe a
full-length experiment.

## Training-window coverage

`--window-sampling coverage_balanced` is the default. It targets equal numbers
of simulated training days per calendar year, rather than equal probabilities
for overlapping five-year windows. The coverage matrix uses actual source
dates, including leap days and truncated final calendar years. It never adds
held-out days or joins nonconsecutive years to form training episodes.

The sampler computes constrained window probabilities, converts them to a
fixed-budget integer schedule and shuffles that schedule using a private RNG
for each seed. Parallel batches contain distinct window starts. Every window
is included when the episode budget permits; short runs cannot guarantee full
coverage. There are no additional training episodes or optimizer updates.

Exact equal coverage is generally impossible under the fixed five-year horizon
and split gaps. Logs retain the residual imbalance instead of claiming perfect
balance. `run_configuration.training_window_sampling` and per-seed
`training_sampling_plan` records describe the target probabilities and planned
coverage. Each `training_update.training_sampling_coverage` reports cumulative
window counts, year appearances and actual simulated days per year. Final
coverage is also recorded in `run_complete.training_sampling_coverage_per_seed`.
These counts include successfully collected full training batches; periodic
evaluations and partial failed batches are excluded. A failed batch is logged
as `training_sampling_aborted` and is not replaced to satisfy the schedule.

For comparison, `--window-sampling uniform` preserves the previous independent,
uniform draws with replacement, using the same per-seed random sequence. It
allows duplicate windows within a batch. Both modes leave the fixed periodic
evaluation windows and final held-out evaluation unchanged. There is no new
wet/normal/dry classification or hydrological-class stratification.

To compare the new sampler with the most recent G40/E60 experiment, keep its
other settings fixed:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --episodes 300 --action-bins 200 --seeds 1 2 3 --gamma 1 --eval-interval 1 \
  --window-sampling coverage_balanced \
  --guaranteed-output-scale 0.4 --ecological-flow-scale 0.6 --diagnostics \
  --episode-log runs/five_year_bins200_g40_e60_coverage_balanced_seeds123_run1.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

## Service-threshold sensitivity

`--guaranteed-output-scale` and `--ecological-flow-scale` apply positive
multipliers to the reference thresholds. The current environment configuration
sets guaranteed output to **0.50** and ecological flow to **0.70** of reference,
reductions of 50% and 30% from the original requirements respectively. Multipliers always
apply directly to the original CSV, so reductions do not compound. The reference
CSV remains unchanged; effective and
base values, multipliers and provenance are recorded in the run configuration
and service-requirement metadata. Adjusted values are labelled
`sensitivity_adjusted`, rather than paper-reported requirements.

For the current sensitivity experiment, use 50% guaranteed output and 70% ecological flow, keeping
200 actions, seeds 1/2/3 and the existing training settings:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/harl/bin/python -B examples/train_cascade.py \
  --episodes 300 --action-bins 200 --seeds 1 2 3 --gamma 1 --eval-interval 1 \
  --guaranteed-output-scale 0.5 --ecological-flow-scale 0.7 --diagnostics \
  --episode-log runs/five_year_bins200_g50_e70_mean_sd_seeds123_run1.jsonl \
  --plot-python /Library/Frameworks/Python.framework/Versions/3.9/bin/python3
```

| Reservoir | Effective guaranteed output (MW) | Effective ecological flow (m3/s) |
| --- | ---: | ---: |
| WDD | 1,145 | 630 |
| BHT | 2,283.75 | 812 |
| XLD | 2,672.85 | 812 |
| XJB | 1,375 | 840 |
| THR | 2,495 | 4,200 |

The effective thresholds are used consistently by training, periodic evaluation,
final held-out evaluation, reward penalties and violation counts. Penalty weights,
reward formula, reservoir physics and hard operating bounds are unchanged. All
requirements remain active, so the periodic rate denominator remains 27,375
reservoir-days per seed/round. Use both scales equal to 1 for the reference setup.
Use both equal to 0.9 to reproduce the previous relaxed-threshold experiment.
Use both equal to 0.75 to reproduce the previous 75% experiment.
Changing thresholds changes the optimization and reporting problem: lower
violations or higher returns under relaxed requirements alone do not establish
improvement against the original requirements. Physical generation remains in
the same units for comparison.

## Physics, actions and reward

Raw data and source notes live in `data/cascade_reservoir_new/`. Hard constraints
cover water balance, global operating storage, daily CSV bounds, outlet capacity
and compatibility with downstream storage and outlet limits in the same transition.
Release intervals use current storage, the current day's external inflows and
the bounds on the resulting state.

`five_reservoir_happo_input_bounds_1965_2024.csv` is the authoritative daily limit
table. A row with `action_date=t` already gives the allowed level and storage
of the state resulting from action t. Both physics and observations use that
row directly. There is no additional shift to the following date. At source-year
joins, the previous executed row constrains current storage and the newly selected
action-date row constrains the next storage.

There is no remaining-horizon reachability calculation, future-inflow lookup,
refill target, or return-to-initial-storage condition. Some supplied daily ranges
have equal lower and upper limits; those rows still require that single level.
The environment raises an error when a current transition is infeasible instead
of silently changing the storage balance. Without an end-storage target, episode
generation can include depletion of initially stored water, which remains reported.

The default `storage_sensitivity` head model uses mean starting/ending forebay
elevation, constant efficiency 0.9, and supplied slope/offset assumptions.
This is a sensitivity scenario, not a calibrated tailwater model. Turbine flow
and installed capacity limit generation. `--head-model fixed` uses typical heads.

For index `k` among `K=100` actions:

```text
u = k / (K - 1)
release = feasible_low + u^2 * (feasible_high - feasible_low)
```

Daily team reward is normalized generation minus mean relative guaranteed-output
and ecological-flow deficits, each with weight 1. Generation is normalized by
1,624,800 MWh. Requirements and their source are retained in
`reservoir_service_requirements.csv`. `--no-reward-penalties` selects a separate
generation-only objective.

Gamma applies consistently to return estimation, actor time weights and episode
reward accounting. Physical violation counts, deficit volumes and deficit
energy remain undiscounted. Episode generation and return are totals over the
whole horizon. Soft costs express a tradeoff; hard feasibility does not imply
zero service shortfalls.

## Source and output structure

| File or directory | Responsibility |
| --- | --- |
| `examples/train_cascade.py` | CLI, complete configuration and 55/5 split |
| `examples/train_cascade_parallel.py` | Episode checks, fixed scenarios, seed orchestration, logging and final evaluation |
| `examples/plot_cascade.py` | Interactive backend availability check |
| `examples/plot_cascade_parallel.py` | Separate unsmoothed collector and evaluator figures from launch payloads or JSONL |
| `harl/runners/cascade_reservoir_runner.py` | Parallel collection, unchanged HAPPO updates, periodic and final evaluation |
| `harl/envs/cascade_reservoir/periodic_evaluation.py` | Fixed training-only scenarios, state isolation, window metrics and failure accounting |
| `harl/runners/on_policy_ha_runner.py` | Sequential HAPPO updates and likelihood factors |
| `harl/algorithms/` | Actor and critic objectives, optimizer steps and optional KL controls |
| `harl/envs/cascade_reservoir/` | Data, episode views, workers, current-transition physics, reward and training diagnostics |
| `harl/common/` and `harl/models/` | Buffers, normalization and feed-forward networks |

`--diagnostics` reports parameter changes, entropy/KL, surrogate changes and
Monte Carlo/GAE fit on collected training trajectories. Collector curves show recorded
stochastic returns and generation, averaged across the six complete episodes
collected before an update. They begin after collection and have no fabricated
episode-zero point. Sampled inflow windows affect their variation.

New JSONL records use `run_configuration`, `training_episode`,
`training_update`, `training_diagnostics`,
`periodic_evaluation`, `final_evaluation_sequence` and `run_complete`.
Per-episode records include source years, dates and the actual horizon;
generation units refer to the full episode. Historical files in `runs/` and
`result/` retain the settings and schemas of the runs that produced them.

Dependencies are declared in `setup.py`; install into a suitable Python
environment with `python -m pip install -e .`.
