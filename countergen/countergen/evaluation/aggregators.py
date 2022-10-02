from collections import defaultdict
from typing import Any, DefaultDict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple, TypeVar

import numpy as np
from attrs import define

from countergen.tools.plot_utils import plot_mutli_bars
from countergen.types import AugmentedSample, Category, Input, Performance, Results, StatsAggregator, Outputs
from countergen.tools.math_utils import geometric_mean, mean
from countergen.tools.utils import FromAndToJson


@define
class AveragePerformancePerCategory(StatsAggregator):
    """Compute the average performance for each category."""

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
            print(f"{c}: {perf:.3e}", file=file)

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
class Stats(FromAndToJson):
    mean: float
    uncertainty_2sig: float

    def __str__(self) -> str:
        return f"{self.mean:.3e} +- {self.uncertainty_2sig:.3e}"

    @classmethod
    def from_str(cls, s: str) -> "Stats":
        m, u = s.split(" +- ")
        return Stats(float(m), float(u))

    @classmethod
    def from_seq(cls, s: Sequence[float]) -> "Stats":
        assert s, "empty seq"
        a = np.array(s)
        uncertainty_2sig = 2 * a.std() / np.sqrt(len(s))
        return Stats(mean=a.mean(), uncertainty_2sig=uncertainty_2sig)


@define
class PerformanceStatsPerCategory(StatsAggregator):
    """Compute performance mean and the 2 sigma uncertainty over mean for each category."""

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
            print(f"{c}: {perf}", file=file)

    def load_aggregation(self, file: TextIO) -> Mapping[Category, Stats]:
        lines = file.readlines()
        r = {}
        for l in lines[1:]:
            c, p = l.split(": ")
            r[c] = Stats.from_str(p)
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
class DifferenceStats(StatsAggregator):
    category1: Category
    category2: Category
    relative_difference: bool = False  #: If True, return how much cat2er the cat2 category is on average

    def __call__(self, performances: Results) -> Stats:
        differences = []
        for sample_perfs in performances:
            cat1_perfs = [perf for perf, categories in sample_perfs if self.category1 in categories]
            cat2_perfs = [perf for perf, categories in sample_perfs if self.category2 in categories]
            if cat1_perfs and cat2_perfs:
                diff = mean(cat1_perfs) - mean(cat2_perfs)
                if self.relative_difference:
                    diff /= max(mean(cat1_perfs), mean(cat2_perfs))
                differences.append(diff)
        return Stats.from_seq(differences)

    def save_aggregation(self, aggregate: Stats, file: Optional[TextIO] = None):
        relative = "relative " if self.relative_difference else ""
        print(
            f"The {relative}performance between category {self.category1} and category {self.category2}:",
            file=file,
        )
        print(f"{aggregate}", file=file)

    def load_aggregation(self, file: TextIO) -> float:
        lines = file.readlines()
        return Stats.from_str(lines[1])


OutlierData = Tuple[Input, Outputs, Tuple[Category, ...], Performance]


@define
class OutliersAggregator(StatsAggregator):
    """Return the variations with the biggest (relative) performance gap."""

    aug_samples: Iterable[AugmentedSample]  #: Contains data about the inputs used
    top_k: int = 5
    use_relative_performance: bool = True
    espilon: float = 1e-10

    def __call__(self, performances: Results) -> List[Tuple[OutlierData, OutlierData]]:
        possibles_outliers: List[Tuple[OutlierData, OutlierData, float]] = []
        for aug_sample, variations_perf in zip(self.aug_samples, performances):
            if len(variations_perf) < 2:
                continue
            outliers_data = list(
                zip(
                    [v.text for v in aug_sample.get_variations()],
                    [aug_sample.outputs for _ in range(len(aug_sample.get_variations()))],
                    [cats for _, cats in variations_perf],
                    [perf for perf, _ in variations_perf],
                )
            )
            sorted_outliers = sorted(outliers_data, key=lambda t: t[3])
            small_outlier = sorted_outliers[0]
            big_outlier = sorted_outliers[-1]
            possibles_outliers.append((small_outlier, big_outlier, self.gap(small_outlier[3], big_outlier[3])))

        best_outliers = sorted(possibles_outliers, key=lambda t: t[2], reverse=True)
        return [(small, big) for small, big, _ in best_outliers][: self.top_k]

    def display(self, aggregates: Mapping[str, List[Tuple[OutlierData, OutlierData]]]):
        for model_name, aggregate in aggregates.items():
            print(f"Biggest performance gaps for {model_name}:\n")
            for (inp1, outs1, cats1, perf1), (inp2, outs2, cats2, perf2) in aggregate:
                print(f"Performance={perf1:.3e} on input in categories {cats1}: {inp1} -> {outs1}")
                print(f"Performance={perf2:.3e} on input in categories {cats2}: {inp2} -> {outs2}\n")
            print("-----\n")

    def gap(self, small_perf: float, big_perf: float):
        delta = big_perf - small_perf
        if not self.use_relative_performance:
            return delta

        if small_perf < 0:
            raise ValueError("Can't use relative performance when performance is negative.")

        return delta / (big_perf + self.espilon) if self.use_relative_performance else delta
