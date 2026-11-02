# Supplied daily water-level envelopes

The authoritative table is  
`five_reservoir_happo_input_bounds_1965_2024.csv`. It contains one row for each  
action date and reservoir over 1965--2024: 21,915 dates and five reservoirs.  
Rows are matched by date and reservoir identifier; physical order is  
WDD, BHT, XLD, XJB, THR regardless of file row order.

## Timing and meaning

The row labeled `action_date = t` supplies the permitted state reached after  
executing that day's action:

```text
action on date t -> next state H[t+1], V[t+1]
next_state_min_level_m <= H[t+1] <= next_state_max_level_m
```

The table is therefore indexed directly by action date. The loader and actor  
input do not shift it to the following row. A row's level limits are the next  
state's limits, even though its date identifies the action that creates that  
state.

These columns define a permissible interval, not a prescribed trajectory to  
track. There are no refill targets, future-deadline time features, final-storage  
targets or remaining-horizon trajectory calculations. Decisions use current  
storage, today's known inflow and the row's immediate next-state bounds. When  
the supplied lower and upper bounds happen to coincide, that data row itself  
permits only one next-state level; the loader does not add an equality target.

## Columns

| Column                      | Units and interpretation                         |
| --------------------------- | ------------------------------------------------ |
| `action_date`               | ISO date identifying the action                  |
| `reservoir_id`              | WDD, BHT, XLD, XJB or THR                        |
| `next_state_min_level_m`    | Minimum permitted next-state water level, metres |
| `next_state_max_level_m`    | Maximum permitted next-state water level, metres |
| `next_state_min_storage_m3` | Corresponding minimum next-state storage, m³     |
| `next_state_max_storage_m3` | Corresponding maximum next-state storage, m³     |

Water levels in the supplied file have six decimal places; storage values are  
in whole m³. The loader checks finite and ordered limits, complete unique  
date/reservoir coverage, the configured operating range, and the level-storage  
relationship. Curve consistency is checked in metres with tolerance 1e-6 to  
respect the two supplied rounding precisions. Arrays are immutable after load.

`use_daily_level_limits: true` loads this table and enables the daily envelopes.  
Its SHA-256 is included in the run's input-data hashes. The supplied CSV values  
are retained without interpolation across dates or changes to their contents.  
If daily limits are disabled, the global operating range is used instead.

Training and evaluation both use the row associated with each executed source  
date. Calendar-year and representative-year joins do not reset storage. A  
1,825-day episode uses exactly its executed 1,825 rows; no terminal water-level  
condition is imposed at its endpoint.
