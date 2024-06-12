from __future__ import annotations

import copy
import json
import os
import random
from typing import Callable, Dict, TYPE_CHECKING

import numpy as np
from loguru import logger
from tqdm import tqdm

from ofa.run_manager.run_manager import RunManager
from ofa.utils import EvolutionParams

if TYPE_CHECKING:
    from ..arch_encoders import ArchDesc, CompositeEncoder
    from .utils import SubnetInfo

__all__ = [
    "EvolutionarySearch",
    "SampleTimeOut",
]


def save_best_model(
    constraint_type,
    constraint_value,
    speed_type,
    speed_value,
    accuracy_value,
    arch_json,
    save_path,
):
    keywords = [
        f"{constraint_type}_constraint_{constraint_value}",
        f"best_{speed_type}_{speed_value:.4}",
        f"accuracy_{accuracy_value:.4}.json",
    ]
    name = "_".join(keywords)
    with open(os.path.join(save_path, name), "w") as file:
        json.dump(arch_json, file)
    logger.info(name)


class SampleTimeOut(RuntimeError):
    pass


class EvolutionarySearch:
    _valid_metrics = [
        "accuracy",
        "latency",
        "efficiency",
        # 'flops',
    ]

    _max_metrics = [
        "accuracy",
        "efficiency",
    ]

    def __init__(
        self,
        run_manager: RunManager,
        arch_encoder: CompositeEncoder,
        predictor_methods: Dict[str, Callable],
    ) -> None:
        params: EvolutionParams = run_manager.cur_stage_config.evolution
        if params.optimize_val not in self._valid_metrics:
            raise NotImplementedError(f"Unknown optimize_val: {params.optimize_val}")
        for constraint in [params.constraint_type]:
            if constraint not in self._valid_metrics:
                raise NotImplementedError(f"Unknown constraint: {constraint}")
        self.all_metrics = sorted([params.optimize_val] + [params.constraint_type])

        self.optimize_val = params.optimize_val
        self.optimize_position = 1 + self.all_metrics.index(self.optimize_val)
        self.reverse_sort = self.optimize_val in self._max_metrics
        self.constraints = [params.constraint_type]
        self.constraint_thresholds = {
            "accuracy": params.evolution_accuracy_constraint,
            "latency": params.latency_constraint,
            "flops": params.flops_constraint,
            "efficiency": params.efficiency_constraint,
        }
        self.predictors = predictor_methods
        self.arch_encoder = arch_encoder

        self.mutate_prob = params.mutate_probability
        self.population_size = params.population_size
        self.generations = params.generations
        self.parent_ratio = params.parent_ratio
        self.mutation_ratio = params.mutation_ratio
        self.mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        self.parents_size = int(round(self.parent_ratio * self.population_size))
        self.population = []  # (sample, *[metrics]) tuples
        self.child_pool = []
        self.scores_pool = []
        self.best_info = None
        self.best_optimized_value: float = None

        self.sample_count_threshold = max(20000, self.population_size * 100)
        self.__sample_attempt = 0

        self.is_root = run_manager.is_root
        self.rank = run_manager._rank
        self.tensorboard = run_manager.tensorboard if self.is_root else None

    def is_valid(self, constraint: str, score: float) -> bool:
        """
        constraints passed if:
          - accuracy >= accuracy_threshold
          - efficiency >= efficiency_threshold
          - flops <= flops_threshold
          - latency <= latency_threshold
        """
        threshold = self.constraint_thresholds[constraint]
        if constraint in self._max_metrics:
            return score >= threshold
        return score <= threshold

    def get_arch_scores(self, arch_sample: ArchDesc):
        scores: Dict[str, float] = {}
        for constraint in self.constraints:
            score = self.predictors[constraint]([arch_sample]).item()
            if not self.is_valid(constraint, score):
                return None
            scores[constraint] = score
        return scores

    def random_sample_arch(self):
        while self.__sample_attempt < self.sample_count_threshold:
            arch_sample = self.arch_encoder.random_sample_arch()
            scores = self.get_arch_scores(arch_sample)
            if scores is not None:
                return arch_sample, scores
            self.__sample_attempt += 1
        else:
            raise SampleTimeOut("Sample limit out")

    def extend_population(self):
        predictor = self.predictors[self.optimize_val]
        for arch_sample, scores in zip(self.child_pool, self.scores_pool):
            scores[self.optimize_val] = predictor([arch_sample]).item()
            # arch_sample: dict[arch parameter: str, values: list]
            population_element = [arch_sample]
            for metric in self.all_metrics:
                population_element.append(scores[metric])
            self.population.append(population_element)
        self.child_pool = []
        self.scores_pool = []

    def generate_population(self, verbose=False):
        if verbose:
            logger.info("Generate random population...")
        for _ in range(self.population_size):
            arch_sample, scores = self.random_sample_arch()
            self.child_pool.append(arch_sample)
            self.scores_pool.append(scores)
        self.extend_population()
        self.__sample_attempt = 0

    def random_mutation(self, sample: ArchDesc):
        new_sample = copy.deepcopy(sample)
        self.arch_encoder.random_resample(new_sample, self.mutate_prob)
        return new_sample

    def mutate_sample(self, sample: ArchDesc):
        while True:
            new_sample = self.random_mutation(sample)
            scores = self.get_arch_scores(new_sample)
            if scores is not None:
                return new_sample, scores

    def random_crossover(self, sample1: ArchDesc, sample2: ArchDesc):
        new_sample = copy.deepcopy(sample1)
        for part in new_sample.keys():
            new_sample_part = new_sample[part]
            sample1_part = sample1[part]
            sample2_part = sample2[part]

            if new_sample_part is None:
                continue
            elif not isinstance(new_sample_part, dict):
                new_sample[part] = random.choice([sample1_part, sample2_part])
            else:
                for key in new_sample_part.keys():
                    value_new = new_sample_part[key]
                    value1 = sample1_part[key]
                    value2 = sample2_part[key]
                    if not isinstance(value_new, list):
                        new_sample_part[key] = random.choice([value1, value2])
                    else:
                        for i in range(len(value_new)):
                            value_new[i] = random.choice([value1[i], value2[i]])
        return new_sample

    def crossover_sample(self, sample1: dict, sample2: dict):
        while True:
            new_sample = self.random_crossover(sample1, sample2)
            scores = self.get_arch_scores(new_sample)
            if scores is not None:
                return new_sample, scores

    def mutation(self):
        for _ in range(self.mutation_numbers):
            par_sample = self.population[np.random.randint(self.parents_size)][0]
            new_sample, scores = self.mutate_sample(par_sample)
            self.child_pool.append(new_sample)
            self.scores_pool.append(scores)
        self.extend_population()

    def crossingover(self):
        for _ in range(self.population_size - self.mutation_numbers):
            par_sample1 = self.population[np.random.randint(self.parents_size)][0]
            par_sample2 = self.population[np.random.randint(self.parents_size)][0]
            new_sample, scores = self.crossover_sample(par_sample1, par_sample2)
            self.child_pool.append(new_sample)
            self.scores_pool.append(scores)
        self.extend_population()

    def sort_population(self):
        self.population.sort(
            key=lambda x: x[self.optimize_position], reverse=self.reverse_sort
        )

    def choose_best(self, current_element):
        current_value = current_element[self.optimize_position]
        best_value = self.best_optimized_value
        if not best_value:
            is_best = not best_value
        elif self.optimize_val in self._max_metrics:
            is_best = current_value > best_value
        else:
            is_best = current_value < best_value
        if is_best:
            self.best_info = current_element
            self.best_optimized_value = current_value

    def run_evolution_search(self, verbose=True) -> SubnetInfo:
        self.generate_population(verbose=verbose)
        self.sort_population()

        if verbose:
            logger.info("Start evolution")
            t = tqdm(
                range(self.generations),
                desc="Searching for optimal {} with {} constraints".format(
                    self.optimize_val, ", ".join(self.constraints)
                ),
            )
        for itr in range(self.generations):
            self.population = self.population[: self.parents_size]
            self.mutation()
            self.crossingover()

            self.sort_population()
            current_best = self.population[0]
            self.choose_best(current_best)

            if self.is_root:
                self.tensorboard.write_metrics(
                    {
                        f"{metric}_{self.rank}": current_best[i + 1]
                        for i, metric in enumerate(self.all_metrics)
                    }
                )
            if verbose:
                t.set_postfix(
                    {
                        "Iter": f"{itr+1}/{self.generations}",
                        f"{self.optimize_val}: ": f"{self.best_info[self.optimize_position]}",
                    }
                )
                t.update(1)

        best_info = dict(net_config=self.best_info[0], metrics={}, predicted={})
        for metric, predict_score in zip(self.all_metrics, self.best_info[1:]):
            best_info["predicted"][metric] = predict_score
        self.best_info = best_info

        return self.best_info
