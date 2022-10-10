from collections import defaultdict
from typing import Any, DefaultDict, Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple, TypeVar

import numpy as np
from attrs import define

from countergen.tools.plot_utils import plot_mutli_bars
from countergen.types import AugmentedSample, Category, Input, Performance, Results, Aggregator, Outputs
from countergen.tools.math_utils import geometric_mean, mean
from countergen.tools.utils import FromAndToJson, unwrap_float, unwrap_list_of_floats


@define
class AveragePerformancePerCategory(Aggregator):
    """Compute the average performance for each category."""

    use_geometric_mean: bool = False

    def __call__(self, performances: Results) -> Mapping[Category, float]:
        performances_by_category: DefaultDict[Category, List[float]] = defaultdict(lambda: [])
        for sample_perfs in performances:
            for perf, categories in sample_perfs:
                for c in categories:
                    if isinstance(perf, float):
                        performances_by_category[c].append(perf)
                    else:  # perf is list of floats
                        for p in perf:
                            performances_by_category[c].append(p)

        mean_ = geometric_mean if self.use_geometric_mean else mean
        avg_performances_by_category = {c: mean_(perfs) for c, perfs in performances_by_category.items()}
        return avg_performances_by_category

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
class PerformanceStatsPerCategory(Aggregator):
    """Compute performance mean and the 2 sigma uncertainty over mean for each category."""

    def __call__(self, performances: Results) -> Mapping[Category, Stats]:
        performances_by_category: DefaultDict[Category, List[float]] = defaultdict(lambda: [])
        for sample_perfs in performances:
            for perf, categories in sample_perfs:
                for c in categories:
                    if isinstance(perf, float):
                        performances_by_category[c].append(perf)
                    else:  # perf is list of floats
                        for p in perf:
                            performances_by_category[c].append(p)

        aggregate = {c: Stats.from_seq(perfs) for c, perfs in performances_by_category.items()}
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
class DifferenceStats(Aggregator):
    """Compute performance mean and the 2 sigma uncertainty (relative) difference of the performance in each samples.

    Return a positive mean if category1 has higher performance that category2, and a negative one otherwise.

    If a sample has mutliple variations of the same category, compute the mean performance of the category.

    Excepts performance to be a float."""

    category1: Category
    category2: Category
    relative_difference: bool = True

    def __call__(self, performances: Results) -> Stats:
        differences = []
        for sample_perfs in performances:
            cat1_perfs = [unwrap_float(perf) for perf, categories in sample_perfs if self.category1 in categories]
            cat2_perfs = [unwrap_float(perf) for perf, categories in sample_perfs if self.category2 in categories]
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

    def load_aggregation(self, file: TextIO) -> Stats:
        lines = file.readlines()
        return Stats.from_str(lines[1])


OutlierData = Tuple[Input, Outputs, Tuple[Category, ...], Performance]


@define
class OutliersAggregator(Aggregator):
    """Return the variations with the biggest (relative) performance gap."""

    aug_samples: Iterable[AugmentedSample]  #: Contains data about the inputs used
    top_k: int = 5
    perf_per_output: bool = False  #: Performance is expected to be a list of float, one per output
    use_relative_performance: bool = True
    epsilon: float = 1e-10

    def __call__(self, performances: Results) -> List[Tuple[OutlierData, OutlierData]]:
        possibles_outliers: List[Tuple[OutlierData, OutlierData, float]] = []
        for aug_sample, variations_perf in zip(self.aug_samples, performances):
            if len(variations_perf) < 2:
                continue

            if not self.perf_per_output:
                outliers_data = list(
                    zip(
                        [v.text for v in aug_sample.get_variations()],
                        [aug_sample.outputs for _ in range(len(aug_sample.get_variations()))],
                        [cats for _, cats in variations_perf],
                        [unwrap_float(perf) for perf, _ in variations_perf],
                    )
                )

                sorted_outliers = sorted(outliers_data, key=lambda t: t[3])
                small_outlier = sorted_outliers[0]
                big_outlier = sorted_outliers[-1]
                possibles_outliers.append((small_outlier, big_outlier, self.gap(small_outlier[3], big_outlier[3])))
            else:
                just_perfs = [unwrap_list_of_floats(p) for p, _ in variations_perf]
                for i, output in enumerate(aug_sample.outputs):

                    outliers_data = list(
                        zip(
                            [v.text for v in aug_sample.get_variations()],
                            [
                                [
                                    output,
                                ]
                                for _ in range(len(aug_sample.get_variations()))
                            ],
                            [cats for _, cats in variations_perf],
                            [perf[i] for perf in just_perfs],
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

        return delta / (big_perf + self.epsilon)


class BiasFromProbsAggregator(Aggregator):
    """Return average of log the biggest relative performance gap per sample accross all outputs and variations.

    Metric from https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/bias_from_probabilities"""

    categories: Optional[str] = None  #: Which categories to take into account
    epsilon: float = 1e-10
    perf_per_output: bool = False  #: Performance is expected to be a list of float, one per output

    def __call__(self, performances: Results) -> float:
        smallest_ratios = []
        for variations_perf in performances:
            if len(variations_perf) < 2:
                continue

            if self.perf_per_output:
                just_perfs = [unwrap_list_of_floats(p) for p, _ in variations_perf]
                smallest_ratios_in_sample = []
                for i in range(len(just_perfs[0])):
                    perf_by_output = [p[i] for p in just_perfs]
                    smallest_ratios_in_sample.append(min(perf_by_output) / max(perf_by_output))
                smallest_ratios.append(min(smallest_ratios_in_sample))
            else:
                just_perf = [unwrap_float(p) for p, _ in variations_perf]
                smallest_ratios.append(min(just_perf) / max(just_perf))

        return mean([np.log(g) for g in smallest_ratios])
