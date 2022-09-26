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


class Variation(NamedTuple):
    text: Input
    categories: Tuple[Category, ...]


class AugmentedSample(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def input(self) -> Input:
        ...

    @abc.abstractproperty
    def outputs(self) -> Outputs:
        ...

    @abc.abstractmethod
    def get_variations(self) -> Sequence[Variation]:
        ...


SampleResults = Iterable[Tuple[Performance, Tuple[Category, ...]]]
Results = Iterable[SampleResults]

Aggregate = TypeVar("Aggregate")


class StatsAggregator(Generic[Aggregate], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, performances: Results) -> Aggregate:
        ...

    @abc.abstractmethod
    def save_aggregation(self, aggregate: Aggregate, file: Optional[TextIO] = None):
        ...

    @abc.abstractmethod
    def load_aggregation(self, file: TextIO) -> Aggregate:
        ...

    @abc.abstractmethod
    def display(self, aggregates: Mapping[str, Aggregate]):
        ...
