"""Coverage-aware historical window scheduling, without training or output files."""
import collections
from contextlib import ExitStack, redirect_stdout
import copy
from datetime import date, timedelta
import io
import json
from pathlib import Path
import random
import sys
import unittest
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harl.envs.cascade_reservoir.data_loader import CascadeData
from harl.envs.cascade_reservoir.episodes import training_windows
from harl.envs.cascade_reservoir.training_sampler import (
    TrainingWindowSampler, validate_sampling_mode)


class CalendarData:
    """Small actual-calendar fixture; exposes only the sampler's date inputs."""
    def __init__(self, start, end, evaluation_years=()):
        self.dates = tuple(start + timedelta(days=i) for i in range((end-start).days+1))
        self.years = np.asarray([day.year for day in self.dates])
        self.evaluation_years = tuple(evaluation_years)
        self.training_years = tuple(year for year in sorted(set(self.years.tolist()))
                                    if year not in self.evaluation_years)

    def indices(self, start_year, end_year):
        selected = np.flatnonzero((self.years >= start_year) & (self.years <= end_year))
        if not len(selected):
            raise ValueError("No dates in the requested years")
        return int(selected[0]), int(selected[-1])+1


def independent_counts(data, window_starts, days):
    """Count calendar dates directly, without using sampler coverage matrices."""
    windows = training_windows(data, days)
    counts = collections.Counter(window_starts)
    year_days = dict.fromkeys(map(str, data.training_years), 0)
    year_episodes = dict.fromkeys(map(str, data.training_years), 0)
    for start in window_starts:
        first, _ = data.indices(start, start)
        dates = data.dates[first:first+days]
        for day in dates:
            year_days[str(day.year)] += 1
        for year in set(day.year for day in dates):
            year_episodes[str(year)] += 1
    return ({str(year): counts[year] for year in windows}, year_episodes, year_days)


def relative_year_rmse(year_days):
    values = np.asarray(list(year_days.values()), dtype=float)
    return float(np.sqrt(np.mean(((values-values.mean())/values.mean())**2)))


class CascadeTrainingSamplerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / 'harl/configs/envs_cfgs/cascade_reservoir.yaml').open() as stream:
            physical = yaml.safe_load(stream)
        with (ROOT / 'harl/configs/algos_cfgs/happo_cascade.yaml').open() as stream:
            experiment = yaml.safe_load(stream)['experiment']
        cls.data = CascadeData(ROOT / 'data/cascade_reservoir_new', physical, experiment)
        cls.balanced = {seed: TrainingWindowSampler(cls.data, 1825, 300, 6, seed)
                        for seed in (1, 2, 3)}

    def test_exact_budget_integer_quotas_and_unique_batches(self):
        for seed, sampler in self.balanced.items():
            with self.subTest(seed=seed):
                self.assertEqual(len(sampler.window_starts), 39)
                self.assertEqual(np.asarray(sampler.schedule).shape, (50, 6))
                self.assertTrue(np.issubdtype(sampler.quotas.dtype, np.integer))
                self.assertEqual(int(sampler.quotas.sum()), 300)
                self.assertTrue((sampler.quotas >= 1).all())
                self.assertTrue((sampler.quotas <= 50).all())
                self.assertTrue(all(len(set(batch)) == 6 for batch in sampler.schedule))
                observed = collections.Counter(np.asarray(sampler.schedule).reshape(-1).tolist())
                self.assertEqual(observed, dict(zip(sampler.window_starts, sampler.quotas.tolist())))
                self.assertTrue(np.isfinite(sampler.probabilities).all())
                self.assertAlmostEqual(float(sampler.probabilities.sum()), 1., places=12)
                self.assertGreaterEqual(float(sampler.probabilities.min()), 1./300-1e-12)
                self.assertLessEqual(float(sampler.probabilities.max()), 1./6+1e-12)

    def test_fixed_seed_repeats_schedule_and_other_seed_changes_order(self):
        replay = TrainingWindowSampler(self.data, 1825, 300, 6, 1)
        np.testing.assert_array_equal(replay.schedule, self.balanced[1].schedule)
        np.testing.assert_array_equal(replay.quotas, self.balanced[1].quotas)
        self.assertFalse(np.array_equal(self.balanced[1].schedule, self.balanced[2].schedule))

    def test_sampler_never_consumes_global_random_states(self):
        numpy_before, python_before = np.random.get_state(), random.getstate()
        balanced = TrainingWindowSampler(self.data, 1825, 12, 6, 7)
        uniform = TrainingWindowSampler(self.data, 1825, 12, 6, 7, mode='uniform')
        balanced.next_batch()
        uniform.next_batch()
        balanced.metadata()
        balanced.coverage_summary([])
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(python_before, random.getstate())

    def test_legacy_uniform_sequence_matches_all_scalar_draws(self):
        for seed in (1, 2, 3):
            with self.subTest(seed=seed):
                sampler = TrainingWindowSampler(self.data, 1825, 300, 6, seed, mode='uniform')
                legacy = np.random.RandomState(seed)
                expected = [int(legacy.choice(tuple(training_windows(self.data, 1825))))
                            for _ in range(300)]
                observed = []
                for update in range(50):
                    self.assertEqual(sampler.batches_consumed, update)
                    sampler.metadata()
                    sampler.coverage_summary(observed)
                    observed.extend(sampler.next_batch())
                self.assertEqual(observed, expected)
                self.assertIsNone(sampler.quotas)
                self.assertIsNone(sampler.schedule)
                with self.assertRaises(RuntimeError):
                    sampler.next_batch()

    def test_calendar_matrix_and_coverage_summaries_are_independent(self):
        sampler = self.balanced[1]
        expected = np.zeros((len(self.data.training_years), len(sampler.window_starts)), dtype=int)
        year_index = {year: i for i, year in enumerate(self.data.training_years)}
        for column, start in enumerate(sampler.window_starts):
            first, _ = self.data.indices(start, start)
            dates = self.data.dates[first:first+1825]
            self.assertEqual(len(dates), 1825)
            self.assertTrue(all((b-a).days == 1 for a, b in zip(dates, dates[1:])))
            self.assertFalse(set(day.year for day in dates) & set(self.data.evaluation_years))
            for day in dates:
                expected[year_index[day.year], column] += 1
        np.testing.assert_array_equal(sampler.year_day_counts, expected)
        np.testing.assert_array_equal(expected.sum(axis=0), np.full(39, 1825))
        for seed, candidate in self.balanced.items():
            with self.subTest(seed=seed):
                starts = np.asarray(candidate.schedule).reshape(-1).tolist()
                windows, year_episodes, year_days = independent_counts(self.data, starts, 1825)
                summary = candidate.coverage_summary(starts)
                self.assertEqual(summary['completed_episodes'], 300)
                self.assertEqual(summary['total_training_days'], 547500)
                self.assertEqual(summary['window_counts'], windows)
                self.assertEqual(summary['year_episode_counts'], year_episodes)
                self.assertEqual(summary['year_day_counts'], year_days)
                for year, days in year_days.items():
                    self.assertAlmostEqual(summary['year_day_fractions'][year], days/547500., places=14)
                self.assertAlmostEqual(summary['imbalance']['relative_rmse_to_equal_years'],
                                       relative_year_rmse(year_days), places=12)
                json.dumps(candidate.metadata(), allow_nan=False)
                json.dumps(summary, allow_nan=False)

    def test_balance_improves_year_day_rmse_without_claiming_equality(self):
        expected_uniform = self.balanced[1].year_day_counts.mean(axis=1)
        uniform_rmse = float(np.sqrt(np.mean(((expected_uniform-expected_uniform.mean()) /
                                              expected_uniform.mean())**2)))
        for seed, sampler in self.balanced.items():
            with self.subTest(seed=seed):
                _, _, balanced_days = independent_counts(self.data,
                    np.asarray(sampler.schedule).reshape(-1).tolist(), 1825)
                balanced_rmse = relative_year_rmse(balanced_days)
                legacy = np.random.RandomState(seed)
                legacy_starts = [int(legacy.choice(tuple(sampler.window_starts))) for _ in range(300)]
                _, _, legacy_days = independent_counts(self.data, legacy_starts, 1825)
                self.assertLess(balanced_rmse, uniform_rmse)
                self.assertLess(balanced_rmse, relative_year_rmse(legacy_days))
                self.assertGreater(balanced_rmse, 0.)

    def test_partial_zero_coverage_and_schedule_exhaustion(self):
        sampler = TrainingWindowSampler(self.data, 1825, 6, 6, 1)
        empty = sampler.coverage_summary([])
        self.assertEqual(empty['completed_episodes'], 0)
        self.assertEqual(empty['total_training_days'], 0)
        self.assertTrue(all(count == 0 for count in empty['window_counts'].values()))
        self.assertTrue(all(count == 0 for count in empty['year_day_counts'].values()))
        self.assertIsNone(empty['imbalance']['relative_rmse_to_equal_years'])
        self.assertIsNone(empty['imbalance']['coefficient_of_variation'])
        batch = sampler.next_batch()
        self.assertEqual(len(set(batch)), 6)
        summary = sampler.coverage_summary(batch)
        self.assertEqual(summary['completed_episodes'], 6)
        self.assertTrue(summary['imbalance']['uncovered_years'])
        with self.assertRaises(RuntimeError):
            sampler.next_batch()

    def test_short_horizon_and_leap_calendar_counts(self):
        sampler = TrainingWindowSampler(self.data, 8, 12, 6, 5)
        self.assertEqual(len(sampler.window_starts), 55)
        self.assertEqual(int(sampler.year_day_counts.sum()), 55*8)
        starts = [start for _ in range(2) for start in sampler.next_batch()]
        self.assertEqual(sampler.coverage_summary(starts)['total_training_days'], 96)
        leap = CalendarData(date(2000, 1, 1), date(2005, 12, 31))
        five_year = TrainingWindowSampler(leap, 1825, 12, 2, 2)
        self.assertEqual(tuple(five_year.window_starts), (2000, 2001))
        self.assertEqual(int(five_year.year_day_counts[0, 0]), 366)
        self.assertEqual(int(five_year.year_day_counts[4, 0]), 364)
        summary = five_year.coverage_summary(np.asarray(five_year.schedule).reshape(-1).tolist())
        self.assertGreater(summary['imbalance']['relative_rmse_to_equal_years'], 0.)

    def test_batch_distinctness_constraints_do_not_break_legacy_mode(self):
        data = CalendarData(date(2000, 1, 1), date(2005, 12, 31))
        with self.assertRaises(ValueError):
            TrainingWindowSampler(data, 1825, 6, 3, 1)
        uniform = TrainingWindowSampler(data, 1825, 6, 3, 1, mode='uniform')
        self.assertLess(len(set(uniform.next_batch())), 3)

    def test_invalid_settings_and_unknown_window_are_rejected(self):
        valid = dict(data=self.data, episode_days=1825, episodes=300, batch_size=6, seed=1)
        cases = [('episode_days', 0), ('episode_days', True), ('episodes', 0),
                 ('episodes', 7), ('episodes', 300.), ('batch_size', 0),
                 ('batch_size', False), ('seed', True), ('seed', -1),
                 ('seed', 2**32), ('seed', 1.), ('mode', 'unsupported')]
        for name, value in cases:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                TrainingWindowSampler(**dict(valid, **{name: value}))
        for mode in (None, True, [], 'legacy'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_sampling_mode(mode)
        with self.assertRaises(ValueError):
            self.balanced[1].coverage_summary([1998])


class CascadeTrainingSamplerRunnerTest(unittest.TestCase):
    """Run actual eight-day physics and evaluation; optimizer calls are forbidden."""

    @classmethod
    def setUpClass(cls):
        with (ROOT / 'harl/configs/envs_cfgs/cascade_reservoir.yaml').open() as stream:
            cls.physical = yaml.safe_load(stream)
        with (ROOT / 'harl/configs/algos_cfgs/happo_cascade.yaml').open() as stream:
            cls.args = yaml.safe_load(stream)
        cls.args['train'].update(episode_length=8, n_rollout_threads=6)
        cls.args['experiment'].update(episodes=12, diagnostics=False)
        cls.data = CascadeData(ROOT / 'data/cascade_reservoir_new', cls.physical,
                               cls.args['experiment'])

    def assert_nested_equal(self, left, right):
        import torch
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_evaluation_preserves_sampler_and_actual_collection_counts(self):
        import torch
        from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
        from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
        from harl.runners.on_policy_ha_runner import OnPolicyHARunner

        fixed = select_periodic_windows(self.data, episode_days=8)
        for mode in ('coverage_balanced', 'uniform'):
            with self.subTest(mode=mode), ExitStack() as stack:
                output = io.StringIO()
                stack.enter_context(redirect_stdout(output))
                training = stack.enter_context(mock.patch.object(OnPolicyHARunner, 'train',
                    side_effect=AssertionError('This test must not update HAPPO')))
                args = copy.deepcopy(self.args)
                args['experiment']['window_sampling'] = mode
                runner = CascadeReservoirRunner(self.data, self.physical, args, 1,
                    episode_log=output, periodic_windows=fixed)
                stack.callback(runner.close)
                networks = [actor.actor for actor in runner.actor]+[runner.critic.critic]
                optimizers = [actor.actor_optimizer for actor in runner.actor]+[
                    runner.critic.critic_optimizer]
                for optimizer in optimizers:
                    stack.enter_context(mock.patch.object(optimizer, 'step',
                        side_effect=AssertionError('Optimizer step is forbidden')))
                for index, network in enumerate(networks):
                    network.train(index % 2 == 0)
                modules = [module for network in networks for module in network.modules()]
                modes = [module.training for module in modules]
                parameters = [copy.deepcopy(network.state_dict()) for network in networks]
                optimizer_states = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
                value_normalizer = copy.deepcopy(runner.value_normalizer.state_dict())
                initial_env = (runner.envs[0].index, runner.envs[0].storage.copy())
                numpy_before, python_before = np.random.get_state(), random.getstate()
                torch_before = torch.get_rng_state().clone()
                metadata_before = runner.training_sampler.metadata()
                evaluation = runner.periodic_evaluation()
                self.assertEqual(evaluation['status'], 'ok')
                self.assertEqual([window['completed_steps'] for window in evaluation['windows']], [8]*3)
                self.assertEqual([module.training for module in modules], modes)
                self.assertEqual(runner.training_sampler.batches_consumed, 0)
                self.assertEqual(runner.training_sampler.metadata(), metadata_before)
                self.assert_nested_equal(np.random.get_state(), numpy_before)
                self.assertEqual(random.getstate(), python_before)
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_before))
                self.assertEqual(runner.envs[0].index, initial_env[0])
                np.testing.assert_array_equal(runner.envs[0].storage, initial_env[1])
                self.assertEqual(runner.training_coverage_summary()['completed_episodes'], 0)
                self.assertEqual((runner.completed_episodes, runner.episode_attempts,
                                  runner.update_count, runner.training_steps,
                                  runner.collected_training_steps), (0, 0, 0, 0, 0))
                replay = TrainingWindowSampler(self.data, 8, 12, 6, 1, mode=mode)
                expected_batch = replay.next_batch()
                runner.collect_batch(6)
                self.assertEqual(runner.batch_window_starts, expected_batch)
                self.assertEqual(runner.training_sampler.batches_consumed, 1)
                self.assertEqual(runner.sampled_window_starts, expected_batch)
                summary = runner.training_coverage_summary()
                self.assertEqual(summary['completed_episodes'], 6)
                self.assertEqual(summary['total_training_days'], 48)
                expected_counts = independent_counts(self.data, expected_batch, 8)
                self.assertEqual(summary['window_counts'], expected_counts[0])
                self.assertEqual(summary['year_day_counts'], expected_counts[2])
                self.assertEqual(runner.completed_episodes, 6)
                self.assertEqual(runner.collected_training_steps, 48)
                self.assertEqual((runner.update_count, runner.training_steps), (0, 0))
                self.assertEqual(len(set(runner.worker_pids)), 6)
                for network, saved in zip(networks, parameters):
                    self.assert_nested_equal(network.state_dict(), saved)
                for optimizer, saved in zip(optimizers, optimizer_states):
                    self.assert_nested_equal(optimizer.state_dict(), saved)
                self.assert_nested_equal(runner.value_normalizer.state_dict(), value_normalizer)
                training.assert_not_called()

    def test_supplied_consumed_plan_is_rejected_and_actual_budget_is_replanned(self):
        from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
        from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
        from harl.runners.on_policy_ha_runner import OnPolicyHARunner

        args = copy.deepcopy(self.args)
        args['experiment']['window_sampling'] = 'coverage_balanced'
        fixed = select_periodic_windows(self.data, episode_days=8)
        consumed = TrainingWindowSampler(self.data, 8, 12, 6, 1)
        consumed.next_batch()
        with self.assertRaisesRegex(ValueError, 'sampling plan'):
            CascadeReservoirRunner(self.data, self.physical, args, 1,
                periodic_windows=fixed, training_sampler=consumed)
        output, episode_log = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(output))
            runner = CascadeReservoirRunner(self.data, self.physical, args, 1,
                periodic_windows=fixed, episode_log=episode_log)
            stack.callback(runner.close)
            training = stack.enter_context(mock.patch.object(OnPolicyHARunner, 'train',
                side_effect=AssertionError('This test must not update HAPPO')))
            class PlanObserved(Exception):
                pass
            evaluation = stack.enter_context(mock.patch.object(runner, 'periodic_evaluation',
                side_effect=PlanObserved('Stop immediately after planning, before collection')))
            original = runner.training_sampler
            with self.assertRaises(PlanObserved):
                runner.run(18)
            self.assertIsNot(runner.training_sampler, original)
            self.assertEqual(runner.training_sampler.episodes, 18)
            self.assertEqual(runner.training_sampler.batches_consumed, 0)
            self.assertEqual(runner.training_sampler.quotas.sum(), 18)
            evaluation.assert_called_once_with()
            training.assert_not_called()
            logged = [json.loads(line) for line in episode_log.getvalue().splitlines()]
            self.assertEqual(logged[0]['record_type'], 'training_sampling_plan')
            self.assertEqual(logged[0]['episodes'], 18)
            self.assertEqual(logged[0]['planned_coverage']['completed_episodes'], 18)

    def test_failed_worker_startup_cannot_silently_advance_to_another_batch(self):
        from harl.envs.cascade_reservoir.periodic_evaluation import select_periodic_windows
        from harl.runners.cascade_reservoir_runner import CascadeReservoirRunner
        from harl.runners.on_policy_ha_runner import OnPolicyHARunner

        args = copy.deepcopy(self.args)
        args['experiment']['window_sampling'] = 'coverage_balanced'
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(io.StringIO()))
            runner = CascadeReservoirRunner(self.data, self.physical, args, 1,
                periodic_windows=select_periodic_windows(self.data, episode_days=8))
            stack.callback(runner.close)
            training = stack.enter_context(mock.patch.object(OnPolicyHARunner, 'train',
                side_effect=AssertionError('This test must not update HAPPO')))
            workers = stack.enter_context(mock.patch(
                'harl.runners.cascade_reservoir_runner.ParallelCascadeEnvs',
                side_effect=RuntimeError('Simulated worker startup failure')))
            with self.assertRaisesRegex(RuntimeError, 'Simulated worker startup failure'):
                runner.collect_batch(6)
            workers.assert_called_once()
            self.assertEqual(runner.training_sampler.batches_consumed, 1)
            self.assertEqual(runner.episode_attempts, 0)
            self.assertEqual(runner.training_coverage_summary()['completed_episodes'], 0)
            with self.assertRaisesRegex(RuntimeError, 'already collected'):
                runner.run(12)
            self.assertEqual(runner.training_sampler.batches_consumed, 1)
            training.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
