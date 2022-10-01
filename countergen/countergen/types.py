import abc
from typing import (
    Callable,
    Generic,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    TextIO,
    TypeVar,
    List,
    Tuple,
)

from attrs import define
from countergen.tools.utils import FromAndToJson

Input = str  # The input to an NLP mode
Outputs = List[str]  # The different acceptable outputs of the NLP, string label or number, but in string format

Performance = float  # usually between zero & one (one is better)

# Callable that returns the performance of a model given an input and expected outputs.
ModelEvaluator = Callable[[Input, Outputs], Performance]

Category = str  # The different kinds of data produced by augmenters.


class Augmenter(metaclass=abc.ABCMeta):
    """Objects that can transform an input to another input from a target category.

    Also include objects which can do paraphrase."""

    @abc.abstractproperty
    def categories(self) -> Tuple[Category, ...]:
        """Categories the augmenter can output."""
        ...

    @abc.abstractmethod
    def transform(self, inp: Input, to: Category) -> Input:
        """Transform the input to make it a member of the target category.

        Return the input unchanged if the input is already a member of the target category or if
        the transformation is not applicable."""
        ...


class Paraphraser(Augmenter):
    """Objects that can transform an input to another input with the same meaning."""

    @abc.abstractmethod
    def transform(self, inp: Input, to: Category = "") -> Input:
        """Transform the input to make it another input with the same meaning.

        Ignore the target category."""
        ...

    @property
    def categories(self) -> Tuple[Category, ...]:
        return ()


@define
class Variation(FromAndToJson):
    """An input associated with the different categories it belongs to."""

    text: Input
    categories: Tuple[Category, ...]


class AugmentedSample(metaclass=abc.ABCMeta):
    """Hold different variations of a sample with the same meaning, which should have the same expected outputs."""

    @abc.abstractproperty
    def input(self) -> Input:
        """The original input."""
        ...

    @abc.abstractproperty
    def outputs(self) -> Outputs:
        """The expected outputs."""
        ...

    @abc.abstractmethod
    def get_variations(self) -> Sequence[Variation]:
        """The variations of the original input, which still should procude one of the expected outputs."""
        ...


SampleResults = Iterable[Tuple[Performance, Tuple[Category, ...]]]
Results = Iterable[SampleResults]

Aggregate = TypeVar("Aggregate")


class StatsAggregator(Generic[Aggregate], metaclass=abc.ABCMeta):
    """Objects that can aggregate, and optionally save, load and display the performances of models.

    The aggregate can be of any type."""

    @abc.abstractmethod
    def __call__(self, performances: Results) -> Aggregate:
        """Return an aggregate of the performances.

        The performances is a list of the results for each sample:
        for each sample, the performance and the categories of each variations
        are given."""
        ...

    def save_aggregation(self, aggregate: Aggregate, file: Optional[TextIO] = None):
        """Save the aggregate to a file."""
        raise NotImplementedError(f"{self.__class__.__name__} can't save aggregates.")

    def load_aggregation(self, file: TextIO) -> Aggregate:
        """Load the aggregate from a file."""
        raise NotImplementedError(f"{self.__class__.__name__} can't load aggregates.")

    def display(self, aggregates: Mapping[str, Aggregate]):
        """Display and compare the aggregates of different models/datasets.

        The keys of the aggregates map should be a short description of what differs between
        the aggregates (model names if models are different, dataset names if the data changes, ...)"""
        print(aggregates)
