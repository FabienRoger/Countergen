from pathlib import Path
from typing import Iterable, Mapping, TypeVar
from countergen.evaluation.aggregators import PerformanceStatsPerCategory
import countergen.config

from countergen.types import (
    AugmentedSample,
    Category,
    ModelEvaluator,
    Performance,
    Results,
    StatsAggregator,
)
from countergen.tools.utils import maybe_tqdm
from countergen.tools.math_utils import mean


T = TypeVar("T")


def compute_performances(samples: Iterable[AugmentedSample], model: ModelEvaluator) -> Results:
    """Compute and return the performance of each variation in each sample.

    Preserves the order in which samples and variations are given."""
    performances = []
    for sample in maybe_tqdm(samples, countergen.config.verbose >= 2):
        performance = [
            (model(variation.text, sample.outputs), variation.categories) for variation in sample.get_variations()
        ]
        performances.append(performance)
    return performances


def evaluate(
    samples: Iterable[AugmentedSample],
    model: ModelEvaluator,
    aggregator: StatsAggregator[T] = PerformanceStatsPerCategory(),
) -> T:
    """Return the aggregate of the performance of each variation in each sample."""
    return aggregator(compute_performances(samples, model))


def evaluate_and_print(
    samples: Iterable[AugmentedSample],
    model: ModelEvaluator,
    aggregator: StatsAggregator[T] = PerformanceStatsPerCategory(),
):
    aggregator.save_aggregation(evaluate(samples, model, aggregator))


def evaluate_and_save(
    samples: Iterable[AugmentedSample],
    model: ModelEvaluator,
    path: str,
    aggregator: StatsAggregator[T] = PerformanceStatsPerCategory(),
    also_print: bool = True,
):
    with Path(path).open("w", encoding="utf-8") as f:
        perfs = evaluate(samples, model, aggregator)
        aggregator.save_aggregation(perfs, file=f)
        if also_print:
            aggregator.save_aggregation(perfs)
