"""Five-year HAPPO training on 55 years with five held-out evaluation years."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from harl.algorithms.actors.happo import (
    validate_policy_optimization, validate_kl_rollback, validate_joint_target_kl)
from harl.algorithms.critics.v_critic import validate_value_normalization
from harl.envs.cascade_reservoir.data_loader import (
    CascadeData, HEAD_MODELS, resolve_year_split, validate_service_scale)
from harl.envs.cascade_reservoir.episodes import positive_integer
from harl.envs.cascade_reservoir.reward import reward_settings, validate_discount_factor
from harl.envs.cascade_reservoir.training_sampler import SAMPLING_MODES, validate_sampling_mode
from harl.runners.cascade_reservoir_runner import validate_gae_lambda, validate_seed
from examples.train_cascade_parallel import run_parallel_experiment, step_metrics_paths, plot_output_paths


def load_configuration():
    directory = ROOT/'harl/configs'
    def read(relative):
        return yaml.safe_load((directory/relative).read_text(encoding='utf-8'))
    return read('algos_cfgs/happo_cascade.yaml'), read('envs_cfgs/cascade_reservoir.yaml')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/cascade_reservoir_new')
    parser.add_argument('--episodes',type=int,help='Completed multi-year episodes per seed (default: 300)')
    parser.add_argument('--episode-days',type=int,help='Daily decisions per episode (default: 1825)')
    parser.add_argument('--parallel-envs',type=int,help='Independent worker environments (default: 6)')
    parser.add_argument('--window-sampling',choices=SAMPLING_MODES,
                        help='Training-window schedule: coverage_balanced (default) or legacy uniform sampling')
    parser.add_argument('--log-interval',type=int,help='Print training diagnostics every N updates')
    parser.add_argument('--eval-interval',type=int,
                        help='Evaluate fixed training scenarios every N updates (default: 1); also at round 0 and the last update')
    parser.add_argument('--evaluation-years',nargs='+',type=int,metavar='YEAR',
                        help='Exactly five held-out years; the other 55 years are used for training')
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--action-bins',type=int,help='Discrete actions per reservoir (default: 100)')
    parser.add_argument('--gamma',type=float,help='Discount factor in [0,1]; 1 gives undiscounted episode return')
    parser.add_argument('--gae-lambda',type=float,help='GAE trace parameter in [0,1]')
    parser.add_argument('--ppo-epochs',type=int,help='Maximum actor optimization epochs per collection round')
    kl = parser.add_mutually_exclusive_group()
    kl.add_argument('--target-kl',type=float,help='Exact actor KL early-stopping budget')
    kl.add_argument('--no-kl-limit',action='store_true')
    rollback = parser.add_mutually_exclusive_group()
    rollback.add_argument('--kl-rollback',dest='kl_rollback',action='store_true')
    rollback.add_argument('--no-kl-rollback',dest='kl_rollback',action='store_false')
    parser.set_defaults(kl_rollback=None)
    joint = parser.add_mutually_exclusive_group()
    joint.add_argument('--joint-target-kl',type=float,help='Cumulative joint KL budget; requires actor rollback')
    joint.add_argument('--no-joint-kl-limit',action='store_true')
    parser.add_argument('--head-model',choices=HEAD_MODELS)
    parser.add_argument('--guaranteed-output-scale',type=float,
                        help='Positive multiplier for reference guaranteed-output thresholds; defaults to environment YAML, 1 restores reference values')
    parser.add_argument('--ecological-flow-scale',type=float,
                        help='Positive multiplier for reference ecological-flow thresholds; defaults to environment YAML, 1 restores reference values')
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument('--daily-level-limits',dest='daily_level_limits',action='store_true',
                        help='Use the action-date CSV bounds on each next state (default)')
    limits.add_argument('--no-daily-level-limits',dest='daily_level_limits',action='store_false',
                        help='Use global operating bounds only')
    parser.set_defaults(daily_level_limits=None)
    penalties = parser.add_mutually_exclusive_group()
    penalties.add_argument('--reward-penalties',dest='reward_penalties',action='store_true')
    penalties.add_argument('--no-reward-penalties',dest='reward_penalties',action='store_false')
    parser.set_defaults(reward_penalties=None)
    parser.add_argument('--check-data-only',action='store_true',
                        help='Check data, episode calendars and observations without training')
    parser.add_argument('--diagnostics',action='store_true',help='Record learning signals on collected training episodes')
    parser.add_argument('--skip-final-evaluation',action='store_true',help='Reserve evaluation years during tuning')
    parser.add_argument('--episode-log',type=Path,
                        help='New summary JSONL path; also creates compact <stem>_seedN_steps.csv files')
    parser.add_argument('--plot-python',default=sys.executable,help='Python interpreter with interactive Matplotlib')
    parser.add_argument('--no-plot',action='store_true',help='Run without opening plot windows')
    args = parser.parse_args()
    if args.episode_log is not None and args.episode_log.exists():
        parser.error('episode-log must name a new file')
    config, physical = load_configuration()
    exp, train, model, algo = (config[key] for key in ('experiment','train','model','algo'))
    for name in ('episodes','log_interval','seeds','eval_interval','window_sampling'):
        value = getattr(args,name)
        if value is not None:
            exp[name] = value
    if args.episode_days is not None:
        train['episode_length'] = args.episode_days
    if args.parallel_envs is not None:
        train['n_rollout_threads'] = args.parallel_envs
    if args.evaluation_years is not None:
        exp['evaluation_years'] = args.evaluation_years
        exp.pop('evaluation_labels',None)
    if args.action_bins is not None:
        model['action_bins'] = args.action_bins
    if args.ppo_epochs is not None:
        algo['ppo_epoch'] = args.ppo_epochs
    if args.target_kl is not None:
        algo['target_kl'] = args.target_kl
    elif args.no_kl_limit:
        algo['target_kl'] = None
    if args.kl_rollback is not None:
        algo['kl_rollback'] = args.kl_rollback
    if args.joint_target_kl is not None:
        algo['joint_target_kl'] = args.joint_target_kl
    elif args.no_joint_kl_limit:
        algo['joint_target_kl'] = None
    if args.head_model is not None:
        physical['head_model'] = args.head_model
    for name in ('guaranteed_output_scale','ecological_flow_scale'):
        value = getattr(args,name)
        if value is not None:
            physical[name] = value
    if args.daily_level_limits is not None:
        physical['use_daily_level_limits'] = args.daily_level_limits
    physical['reward_penalties'] = dict(physical.get('reward_penalties',{}))
    if args.reward_penalties is not None:
        physical['reward_penalties']['enabled'] = args.reward_penalties
    try:
        train['episode_length'] = positive_integer(train['episode_length'],'episode_days')
        train['n_rollout_threads'] = positive_integer(train['n_rollout_threads'],'parallel_envs')
        exp['episodes'] = positive_integer(exp['episodes'],'episodes')
        exp['log_interval'] = positive_integer(exp['log_interval'],'log_interval')
        exp['eval_interval'] = positive_integer(exp.get('eval_interval',1),'eval_interval')
        exp['window_sampling'] = validate_sampling_mode(exp.get('window_sampling','coverage_balanced'))
        if exp['episodes'] % train['n_rollout_threads']:
            raise ValueError('episodes must be divisible by parallel_envs')
        if not isinstance(exp['seeds'],(list,tuple)) or not exp['seeds']:
            raise ValueError('seeds must be a nonempty sequence')
        exp['seeds'] = [validate_seed(seed) for seed in exp['seeds']]
        if len(set(exp['seeds'])) != len(exp['seeds']):
            raise ValueError('seeds must be unique')
        model['action_bins'] = positive_integer(model['action_bins'],'action_bins')
        if model['action_bins'] < 2 or model['action_distribution'] != 'categorical':
            raise ValueError('Categorical control requires at least two action bins')
        algo['gamma'] = validate_discount_factor(algo['gamma'] if args.gamma is None else args.gamma)
        algo['gae_lambda'] = validate_gae_lambda(algo['gae_lambda'] if args.gae_lambda is None else args.gae_lambda)
        algo['ppo_epoch'],algo['target_kl'] = validate_policy_optimization(algo['ppo_epoch'],algo.get('target_kl'))
        algo['kl_rollback'] = validate_kl_rollback(algo.get('kl_rollback',False),algo['target_kl'])
        algo['joint_target_kl'] = validate_joint_target_kl(algo.get('joint_target_kl'),algo['kl_rollback'])
        validate_value_normalization(algo['use_clipped_value_loss'],train['use_valuenorm'])
        for name in ('critic_epoch','actor_num_mini_batch','critic_num_mini_batch'):
            algo[name] = positive_integer(algo[name],name)
        for name in ('actor_num_mini_batch','critic_num_mini_batch'):
            if train['episode_length']*train['n_rollout_threads'] % algo[name]:
                raise ValueError(name+' must divide all collected daily transitions')
        if not isinstance(physical.get('use_daily_level_limits'),bool):
            raise ValueError('use_daily_level_limits must be a boolean')
        for name in ('guaranteed_output_scale','ecological_flow_scale'):
            physical[name] = validate_service_scale(physical.get(name,1.),name)
        reward_settings(physical)
        split = resolve_year_split(exp)
        exp.update({name:list(values) for name,values in split.items()})
    except ValueError as exc:
        parser.error(str(exc))
    for path in step_metrics_paths(args.episode_log,exp['seeds']).values():
        if path.exists():
            parser.error('Daily metrics file already exists: '+str(path))
    if not args.no_plot:
        for path in plot_output_paths(args.episode_log).values():
            if path.exists():
                parser.error('Figure file already exists: '+str(path))
    exp['diagnostics'] = args.diagnostics
    data = CascadeData(args.data_dir,physical,exp)
    run_parallel_experiment(data,physical,config,args)


if __name__=='__main__':
    main()
