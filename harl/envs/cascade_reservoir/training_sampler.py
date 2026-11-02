"""Reproducible training-window schedules with explicit calendar-year coverage.

Coverage balancing changes which complete training trajectories are collected;
it never changes their dates, forcing, reward, or optimization weights.
"""
from datetime import timedelta

import numpy as np

from harl.envs.cascade_reservoir.episodes import positive_integer, training_windows


SAMPLING_MODES = ('coverage_balanced', 'uniform')
_SOLVER_TOLERANCE = 1e-11
_SOLVER_MAX_ITERATIONS = 10000


def validate_sampling_mode(value):
    if not isinstance(value, str) or value not in SAMPLING_MODES:
        raise ValueError('window_sampling must be one of '+', '.join(SAMPLING_MODES))
    return value


def _project_bounded_simplex(values, lower, upper):
    """Euclidean projection onto sum(p)=1 and constant lower/upper bounds."""
    count = len(values)
    if count*lower >= 1.-1e-14 or count*upper <= 1.+1e-14:
        return np.full(count, 1./count, dtype=np.float64)
    left, right = float(np.min(values-upper)), float(np.max(values-lower))
    for _ in range(60):
        shift = (left+right)/2.
        if np.clip(values-shift, lower, upper).sum() > 1.:
            left = shift
        else:
            right = shift
    return np.clip(values-(left+right)/2., lower, upper)


class TrainingWindowSampler:
    """Plan whole episodes without using policy, evaluation or global RNG state.

    The balanced mode minimizes squared distance from equal simulated-day
    fractions across the configured training years. Bounds preserve at least
    one episode per window when the budget permits, and prevent any window
    requiring more than one lane per update. Integer quotas and batch geometry
    can leave residual imbalance; all planned and realized coverage is exposed.

    Uniform mode retains the previous independent, with-replacement choice
    sequence exactly, including duplicate windows within a parallel batch.
    """

    def __init__(self, data, episode_days, episodes, batch_size, seed,
                 mode='coverage_balanced'):
        self.episode_days = positive_integer(episode_days, 'episode_days')
        self.episodes = positive_integer(episodes, 'episodes')
        self.batch_size = positive_integer(batch_size, 'parallel_envs')
        if self.episodes % self.batch_size:
            raise ValueError('episodes must be divisible by parallel_envs')
        if (isinstance(seed, (bool, np.bool_)) or
                not isinstance(seed, (int, np.integer)) or not 0 <= seed <= 2**32-1):
            raise ValueError('Seed must be an integer between 0 and 4294967295')
        self.seed = int(seed)
        self.mode = validate_sampling_mode(mode)
        self.windows = training_windows(data, self.episode_days)
        self.window_starts = tuple(self.windows)
        self.training_years = tuple(sorted(int(year) for year in data.training_years))
        self._window_index = {year:i for i, year in enumerate(self.window_starts)}
        self._year_index = {year:i for i, year in enumerate(self.training_years)}
        self._rng = np.random.RandomState(self.seed)
        self._batches_consumed = 0
        self.updates = self.episodes//self.batch_size
        count = len(self.window_starts)
        if self.mode == 'coverage_balanced' and self.batch_size > count:
            raise ValueError('coverage_balanced sampling requires at least parallel_envs distinct training windows')

        self.year_day_counts = np.zeros((len(self.training_years), count), dtype=np.int64)
        for column, year in enumerate(self.window_starts):
            start, _ = data.indices(year, year)
            dates = data.dates[start:start+self.episode_days]
            if (len(dates) != self.episode_days or
                    any(b-a != timedelta(days=1) for a,b in zip(dates, dates[1:]))):
                raise ValueError('Training windows must retain consecutive historical daily dates')
            for day in dates:
                if day.year not in self._year_index:
                    raise ValueError('Training schedule cannot include held-out source days')
                self.year_day_counts[self._year_index[day.year], column] += 1
        if not np.all(self.year_day_counts.sum(axis=0) == self.episode_days):
            raise ValueError('Every training window must contain exactly episode_days daily rows')
        self._year_membership = (self.year_day_counts > 0).astype(np.int64)
        self._fractions = self.year_day_counts.astype(np.float64)/self.episode_days
        self._target = np.full(len(self.training_years), 1./len(self.training_years))
        self._minimum_quota = int(self.episodes >= count)
        self._solver_iterations = self._quota_refinement_iterations = 0
        self._projected_gradient_residual = 0.
        self.quotas = None
        self.schedule = None
        if self.mode == 'uniform':
            self.probabilities = np.full(count, 1./count, dtype=np.float64)
        else:
            self.probabilities = self._balanced_probabilities()
            self.quotas = self._integer_quotas()
            self.schedule = self._batch_schedule()
            self.quotas.setflags(write=False)
        self.year_day_counts.setflags(write=False)
        self.probabilities.setflags(write=False)

    def _balanced_probabilities(self):
        lower = self._minimum_quota/self.episodes
        upper = 1./self.batch_size
        matrix = self._fractions
        p = np.full(len(self.window_starts), 1./len(self.window_starts))
        extrapolated = p.copy()
        momentum = 1.
        lipschitz = float(np.linalg.eigvalsh(2.*matrix.T.dot(matrix))[-1])
        if not np.isfinite(lipschitz) or lipschitz <= 0.:
            raise ValueError('Coverage matrix has no usable training-day exposure')

        def project(values):
            return _project_bounded_simplex(values, lower, upper)

        def gradient(values):
            return 2.*matrix.T.dot(matrix.dot(values)-self._target)

        def objective(values):
            return float(np.square(matrix.dot(values)-self._target).sum())

        for iteration in range(1, _SOLVER_MAX_ITERATIONS+1):
            candidate = project(extrapolated-gradient(extrapolated)/lipschitz)
            # Monotone restart prevents accelerated steps from increasing the
            # explicitly recorded least-squares coverage objective.
            if objective(candidate) > objective(p)+1e-16:
                extrapolated, momentum = p.copy(), 1.
                candidate = project(p-gradient(p)/lipschitz)
            fixed_point = project(candidate-gradient(candidate)/lipschitz)
            residual = float(np.max(np.abs(fixed_point-candidate)))
            if residual <= _SOLVER_TOLERANCE:
                self._solver_iterations = iteration
                self._projected_gradient_residual = residual
                return candidate
            next_momentum = (1.+np.sqrt(1.+4.*momentum*momentum))/2.
            extrapolated = candidate+(momentum-1.)/next_momentum*(candidate-p)
            p, momentum = candidate, next_momentum
        raise RuntimeError('Calendar-year coverage planning did not converge within its iteration limit')

    def _integer_quotas(self):
        target = self.episodes*self.probabilities
        quotas = np.clip(np.floor(target+1e-12).astype(np.int64),
                         self._minimum_quota, self.updates)
        tie_order = self._rng.permutation(len(quotas))
        while quotas.sum() < self.episodes:
            eligible = tie_order[quotas[tie_order] < self.updates]
            if not len(eligible):
                raise RuntimeError('No schedulable quota remains for the episode budget')
            selected = eligible[np.argmax((target-quotas)[eligible])]
            quotas[selected] += 1
        while quotas.sum() > self.episodes:
            eligible = tie_order[quotas[tie_order] > self._minimum_quota]
            if not len(eligible):
                raise RuntimeError('Minimum window quotas exceed the episode budget')
            selected = eligible[np.argmax((quotas-target)[eligible])]
            quotas[selected] -= 1

        # Bounded one-episode exchanges reduce rounding loss without changing
        # total budget or permitting duplicate windows within an update.
        matrix = self._fractions/self.episodes
        gram = matrix.T.dot(matrix)
        diagonal = np.diag(gram)
        pair_distance = diagonal[:,None]+diagonal[None,:]-2.*gram
        for iteration in range(10000):
            residual = matrix.dot(quotas)-self._target
            gradient = 2.*matrix.T.dot(residual)
            change = gradient[None,:]-gradient[:,None]+pair_distance
            change[quotas <= self._minimum_quota, :] = np.inf
            change[:, quotas >= self.updates] = np.inf
            np.fill_diagonal(change, np.inf)
            ordered = change[np.ix_(tie_order, tie_order)]
            position = np.unravel_index(int(np.argmin(ordered)), ordered.shape)
            if ordered[position] >= -1e-16:
                self._quota_refinement_iterations = iteration
                return quotas
            source, destination = tie_order[position[0]], tie_order[position[1]]
            quotas[source] -= 1
            quotas[destination] += 1
        raise RuntimeError('Integer coverage quota refinement did not converge')

    def _batch_schedule(self):
        remaining = self.quotas.copy()
        batches = []
        for _ in range(self.updates):
            tie_order = self._rng.permutation(len(remaining))
            selected = tie_order[np.argsort(-remaining[tie_order], kind='stable')[:self.batch_size]]
            if len(selected) != self.batch_size or (remaining[selected] <= 0).any():
                raise RuntimeError('Coverage quotas cannot fill a duplicate-free parallel batch')
            remaining[selected] -= 1
            batch = [int(self.window_starts[index]) for index in selected]
            self._rng.shuffle(batch)
            batches.append(tuple(batch))
        if remaining.any():
            raise RuntimeError('Coverage schedule does not exhaust its planned quotas')
        # Avoid making early updates systematically favor the largest quotas.
        self._rng.shuffle(batches)
        return tuple(batches)

    def next_batch(self):
        if self._batches_consumed >= self.updates:
            raise RuntimeError('Training-window schedule is exhausted')
        if self.mode == 'uniform':
            # Scalar calls deliberately preserve the original RNG sequence.
            result = [int(self._rng.choice(self.window_starts)) for _ in range(self.batch_size)]
        else:
            result = list(self.schedule[self._batches_consumed])
        self._batches_consumed += 1
        return result

    @property
    def batches_consumed(self):
        """Expose progress without allowing callers to reset a consumed plan."""
        return self._batches_consumed

    def _coverage_from_counts(self, counts, expected=False):
        counts = np.asarray(counts, dtype=np.float64 if expected else np.int64)
        year_days = self.year_day_counts.dot(counts)
        year_episodes = self._year_membership.dot(counts)
        total = float(year_days.sum())
        fractions = year_days/total if total else np.zeros(len(self.training_years))
        mean_days = total/len(self.training_years)
        minimum, maximum = float(year_days.min()), float(year_days.max())
        cast = float if expected else int
        return dict(completed_episodes=cast(counts.sum()),
            total_training_days=cast(year_days.sum()),
            window_counts={str(year):cast(value) for year,value in zip(self.window_starts, counts)},
            year_episode_counts={str(year):cast(value) for year,value in zip(self.training_years, year_episodes)},
            year_day_counts={str(year):cast(value) for year,value in zip(self.training_years, year_days)},
            year_day_fractions={str(year):float(value) for year,value in zip(self.training_years, fractions)},
            imbalance=dict(coefficient_of_variation=float(year_days.std()/mean_days) if total else None,
                relative_rmse_to_equal_years=float(np.sqrt(np.mean(np.square(
                    fractions/self._target-1.)))) if total else None,
                min_year_days=cast(minimum), max_year_days=cast(maximum),
                max_to_min_ratio=maximum/minimum if minimum else None,
                uncovered_years=[int(year) for year,value in zip(self.training_years, year_days) if value == 0]))

    def coverage_summary(self, completed_window_starts):
        """Summarize complete training episodes only; zero exposures stay visible."""
        starts = list(completed_window_starts)
        if len(starts) > self.episodes:
            raise ValueError('Completed training episodes exceed the sampling budget')
        counts = np.zeros(len(self.window_starts), dtype=np.int64)
        for year in starts:
            if (isinstance(year, (bool, np.bool_)) or
                    not isinstance(year, (int, np.integer)) or year not in self._window_index):
                raise ValueError('Completed window start is absent from the training-only window set')
            counts[self._window_index[year]] += 1
        return self._coverage_from_counts(counts)

    def metadata(self):
        uniform = np.full(len(self.window_starts), self.episodes/len(self.window_starts))
        return dict(mode=self.mode, seed=self.seed, episodes=self.episodes,
            batch_size=self.batch_size, episode_horizon_days=self.episode_days,
            window_count=len(self.window_starts), training_years=list(self.training_years),
            window_starts=list(self.window_starts),
            objective=('minimize squared error from equal simulated training-day fractions per source year'
                       if self.mode == 'coverage_balanced' else 'uniform probability over eligible window starts'),
            coverage_target={str(year):float(weight) for year,weight in zip(self.training_years, self._target)},
            forcing_used_for_planning='source dates only; no inflows, rewards, policies or evaluation outcomes',
            sampling_rule=('precomputed integer quotas and seeded duplicate-free batches' if self.mode == 'coverage_balanced'
                           else 'independent uniform window starts with replacement; legacy scalar RNG sequence'),
            duplicate_windows_within_batch_allowed=self.mode == 'uniform',
            window_probabilities={str(year):float(weight) for year,weight in zip(self.window_starts, self.probabilities)},
            planned_window_counts=({str(year):int(value) for year,value in zip(self.window_starts, self.quotas)}
                                   if self.quotas is not None else None),
            eligible_windows_per_year={str(year):int(value) for year,value in zip(
                self.training_years, self._year_membership.sum(axis=1))},
            uniform_expected_coverage=self._coverage_from_counts(uniform, expected=True),
            optimized_expected_coverage=(self._coverage_from_counts(self.episodes*self.probabilities, expected=True)
                                         if self.mode == 'coverage_balanced' else None),
            planned_coverage=self._coverage_from_counts(self.quotas) if self.quotas is not None else None,
            constraints=dict(minimum_window_episodes=self._minimum_quota if self.mode == 'coverage_balanced' else None,
                             maximum_window_episodes=self.updates if self.mode == 'coverage_balanced' else None),
            planner=dict(algorithm=('bounded-simplex accelerated projected gradient; integer rounding and improving exchanges'
                                    if self.mode == 'coverage_balanced' else 'legacy scalar uniform choices; no coverage optimization'),
                         solver_iterations=self._solver_iterations,
                         projected_gradient_tolerance=_SOLVER_TOLERANCE if self.mode == 'coverage_balanced' else None,
                         projected_gradient_residual=self._projected_gradient_residual if self.mode == 'coverage_balanced' else None,
                         quota_refinement_iterations=self._quota_refinement_iterations),
            limitation='Residual year imbalance can be unavoidable under complete consecutive episodes and held-out isolation; '
                'balancing calendar-year exposure does not stratify wet/normal/dry hydrological classes.')
