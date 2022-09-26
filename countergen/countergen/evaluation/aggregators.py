from collections import defaultdict
from typing import Any, DefaultDict, Iterable, List, Mapping, Optional, Sequence, TextIO, TypeVar

import numpy as np
from attrs import define

from countergen.tools.plot_utils import plot_mutli_bars
from countergen.types import AugmentedSample, Category, Performance, Results, StatsAggregator
from countergen.tools.math_utils import geometric_mean, mean


@define
class AveragePerformancePerCategory(StatsAggregator):
    use_geometric_mean: bool = False

    def __call__(self, performances: Results) -> Mapping[Category, float]:
        performances_per_category: DefaultDict[Category, List[float]] = defaultdict(lambda: [])
        for sample_perfs in performances:
            for perf, categories in sample_perfs:
                for c in categories:
                    performances_per_category[c].append(perf)

        mean_ = geometric_mean if self.use_geometric_mean else mean
        avg_performances_per_category = {c: mean_(perfs) for c, perfs in performances_per_category.items()}
        return avg_performances_per_category

    def save_aggregation(self, aggregate: Mapping[Category, float], file: Optional[TextIO] = None):
        print("Average performance per category:", file=file)
        for c, perf in aggregate.items():
            print(f"{c}: {perf:.6f}", file=file)

    def load_aggregation(self, file: TextIO) -> Mapping[Category, float]:
        lines = file.readlines()
        r = {}
        for l in lines[1:]:
            c, p = l.split(": ")
            r[c] = float(p)
        return r

    def display(self, aggregates: Mapping[str, Mapping[Category, float]]):
        plot_mutli_bars(aggregates, xlabel="Model name", ylabel="Performance", title="Performance by model & category")


@define
class Stats:
    mean: float
    uncertainty_2sig: float

    @classmethod
    def from_seq(cls, s: Sequence[float]):
        assert s, "empty seq"
        a = np.array(s)
        uncertainty_2sig = 2 * a.std() / np.sqrt(len(s))
        return Stats(mean=a.mean(), uncertainty_2sig=uncertainty_2sig)


@define
class PerformanceStatsPerCategory(StatsAggregator):
    """Compute mean and uncertainty over mean."""

    def __call__(self, performances: Results) -> Mapping[Category, Stats]:
        performances_per_category: DefaultDict[Category, List[Performance]] = defaultdict(lambda: [])
        for sample_perfs in performances:
            for perf, categories in sample_perfs:
                for c in categories:
                    performances_per_category[c].append(perf)

        aggregate = {c: Stats.from_seq(perfs) for c, perfs in performances_per_category.items()}
        return aggregate

    def save_aggregation(self, aggregate: Mapping[Category, Stats], file: Optional[TextIO] = None):
        print("Average performance per category:", file=file)
        for c, perf in aggregate.items():
            print(f"{c}: {perf.mean:.6f} +- {perf.uncertainty_2sig:.6f}", file=file)

    def load_aggregation(self, file: TextIO) -> Mapping[Category, Stats]:
        lines = file.readlines()
        r = {}
        for l in lines[1:]:
            c, p = l.split(": ")
            m, u = p.split(" +- ")
            r[c] = Stats(float(m), float(u))
        return r

    def display(self, aggregates: Mapping[str, Mapping[Category, Stats]]):
        means = {n: {c: s.mean for c, s in d.items()} for n, d in aggregates.items()}
        errs = {n: {c: s.uncertainty_2sig for c, s in d.items()} for n, d in aggregates.items()}
        plot_mutli_bars(
            means,
            xlabel="Model name",
            ylabel="Performance",
            title="Performance by model & category",
            err_by_type_by_cat=errs,
        )


@define
class AverageDifference(StatsAggregator):
    positive_category: Category
    negative_category: Category
    relative_difference: bool = False

    def __call__(self, performances: Results) -> float:
        differences = []
        for sample_perfs in performances:
            positive_perfs = [perf for perf, categories in sample_perfs if self.positive_category in categories]
            negative_perfs = [perf for perf, categories in sample_perfs if self.negative_category in categories]
            if positive_perfs and negative_perfs:
                diff = mean(positive_perfs) - mean(negative_perfs)
                if self.relative_difference:
                    diff /= mean(positive_perfs)
                differences.append(diff)
        return mean(differences)

    def save_aggregation(self, aggregate: float, file: Optional[TextIO] = None):
        relative = "relative " if self.relative_difference else ""
        print(
            f"The {relative}performance between category {self.positive_category} and category {self.negative_category}:",
            file=file,
        )
        print(f"{aggregate:.6f}", file=file)

    def load_aggregation(self, file: TextIO) -> float:
        lines = file.readlines()
        return float(lines[1])

    def display(self, aggregates: Mapping[str, float]):
        raise NotImplementedError()
