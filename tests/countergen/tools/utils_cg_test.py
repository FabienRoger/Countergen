from countergen.tools.utils import FromAndToJson, other, all_same
from attrs import define


def test_other():
    """Should behave well on simple cases."""

    assert other((1, 2), 1) == 2
    assert other(("hey", "lo"), "lo") == "hey"


def test_all_same():
    """Should behave well on simple cases."""

    assert not all_same([2, 2, 3])
    assert not all_same(["3", "2", "2"])
    assert not all_same([1, 2, 3])
    assert all_same(["a", "a", "a"])
    assert all_same([9])
    assert all_same([9, 9])
    assert all_same([])


def test_from_to_json_simple():
    """Inheriting from FromAndToJson should give the right behavior."""

    @define
    class A(FromAndToJson):
        first: int
        second: str = "no"

        def useless():
            print("pass")

    a = A(1, "yes")
    d = {"first": 1, "second": "yes"}

    assert a.to_json_dict() == d
    assert A.from_json_dict(a.to_json_dict()) == a

    assert A.from_json_dict(d) == a
    assert A.from_json_dict(d).to_json_dict() == d


def test_from_to_json_simple():
    """Inheriting from FromAndToJson should work on classes with no attributes."""

    @define
    class Empty(FromAndToJson):
        def useless():
            print("pass")

    assert Empty.to_json_dict() == {}


def test_from_to_json_simple():
    """Inheriting from FromAndToJson should work on nested classes."""

    @define
    class A(FromAndToJson):
        first: int
        second: str = "no"

    @define
    class C:
        cc: int = 0

    @define
    class B(FromAndToJson):
        a: A
        c: C = C()

    b = B(A(1, "yes"))
    d = {"a": {"first": 1, "second": "yes"}, "c": {"cc": 0}}

    assert b.to_json_dict() == d
    assert B.from_json_dict(b.to_json_dict()) == b

    assert B.from_json_dict(d) == b
    assert B.from_json_dict(d).to_json_dict() == d
