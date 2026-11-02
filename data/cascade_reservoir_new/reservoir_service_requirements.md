# Guaranteed-output and ecological-flow requirements

Source: Xie et al. (2026), *Shared versus reservoir-specific policy architectures  
in multi-agent reinforcement learning for physically heterogeneous cascade  
reservoir operation*, Journal of Hydrology 679, 136160, **Table A.1, page 14**.  
DOI: <https://doi.org/10.1016/j.jhydrol.2026.136160>.

The companion CSV transcribes only the guaranteed-output and ecological-flow  
columns from that table. The source identity and table locator are retained in  
each row. These values are recorded as reported in the paper.

| Repository ID | Paper ID | Guaranteed output (MW) | Ecological flow (m³/s) |
| ------------- | -------- | ---------------------: | ---------------------: |
| WDD           | WDD      |                 2290.0 |                  900.0 |
| BHT           | BHT      |                 4567.5 |                 1160.0 |
| XLD           | XLD      |                 5345.7 |                 1160.0 |
| XJB           | XJB      |                 2750.0 |                 1200.0 |
| THR           | TGR      |                 4990.0 |                 6000.0 |

## Interpretation

- Each threshold is constant across days and years, matching the paper's fixed  
  experimental requirements. Permitted next-state water levels are supplied  
  separately by the daily water-level envelope table.
- Guaranteed output is power in MW, not energy in MWh. In the daily simulator it  
  will be compared with actual daily mean power; a shortfall energy is the MW  
  shortfall multiplied by the step length in hours.
- Ecological flow is the mean total release at the reservoir outlet in m³/s,  
  including turbine discharge and spill. It is not an inflow, turbine-only flow,  
  or downstream-gauge measurement. Deficit volume is the flow shortfall  
  multiplied by the step length in seconds.
- The paper's TGR means Three Gorges Reservoir, called THR in this repository.  
  Both identifiers are kept so that the mapping is explicit.
- The loader accepts a nonnegative threshold; zero explicitly means that the  
  corresponding requirement is inactive. Every requirement supplied here is  
  positive. Later reward calculations must handle inactive requirements without  
  division by zero.

## Reward use

`load_service_requirements: true` loads and validates this table even when  
reward penalties are disabled. Enabling the penalties also requires loading the  
table. Its SHA-256 and provenance appear in startup, episode, and evaluation  
metadata. No other column of the paper's Table A.1 is imported.

The shared daily reward is cascade energy divided by maximum daily cascade  
energy, minus two weighted averages of relative shortfalls. Each shortfall is  
`max(requirement - actual, 0) / requirement`, averaged over that term's fixed  
positive-requirement set. Both weights default to 1.0; they are explicit  
tradeoffs, not calibrated costs. This follows the paper's relative-deficit penalty  
form while retaining the repository's common team objective, rather than  
reproducing the paper's entire local/global reward mixture.

For the current five positive requirements, generation is in [0,1], each mean  
relative shortfall is in [0,1], and the unit-weight daily reward is in [-2,1].  
Negative rewards are retained. A relative tolerance of 1e-10 excludes floating  
point roundoff from both costs and shortfall counts. All reward components use  
the same configured episode discount (YAML default gamma=0.99; gamma=1 gives  
the undiscounted five-year episode objective); physical compliance rates,  
shortfall energy, and deficit volumes are never discounted. A cascade sum of  
ecological deficits is a sum over outlet service requirements, not unique  
missing water volume in the basin.

Episode totals cover all 1,825 daily decisions across the five-year horizon.  
With five active reservoirs, one complete episode contains 9,125 eligible  
reservoir-days for each requirement. Calendar-year boundaries do not reset  
rewards, shortfall counts, integrated deficits or storage. The 60 source  
years are partitioned into 55 training years and five held-out evaluation  
years; the same requirement table is used in both phases.

Loading this service table does not change hydraulics or the observation  
interface. These requirements remain soft penalties, so compliance is measured  
rather than guaranteed by an added hard release or power lower bound. The daily  
water-level envelopes are applied independently; there is no remaining-horizon  
calculation or fixed-date storage target.

The loader checks one row per reservoir, units through named columns, finite  
nonnegative thresholds, guaranteed output no larger than the configured  
installed capacity, and ecological flow no larger than the total release  
capacity. These capacity checks are necessary consistency checks; they do not  
assert that every daily requirement is achievable under every inflow and  
storage condition.
