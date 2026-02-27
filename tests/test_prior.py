"""Tests for unite.prior."""

import pytest

from unite.prior import (
    Fixed,
    ParameterRef,
    TruncatedNormal,
    Uniform,
    prior_from_dict,
    topological_sort,
)

# ---------------------------------------------------------------------------
# ParameterRef
# ---------------------------------------------------------------------------


class _Tok:
    """Minimal token stub for ParameterRef tests."""

    def __init__(self, name: str):
        self.name = name


def test_parameter_ref_resolve():
    tok = _Tok('x')
    ref = ParameterRef(tok, scale=2.0, offset=1.0)
    assert ref.resolve({tok: 3.0}) == pytest.approx(7.0)


def test_parameter_ref_default_scale_offset():
    tok = _Tok('x')
    ref = ParameterRef(tok)
    assert ref.scale == 1.0
    assert ref.offset == 0.0
    assert ref.resolve({tok: 5.0}) == pytest.approx(5.0)


def test_parameter_ref_mul():
    tok = _Tok('x')
    ref = ParameterRef(tok, scale=2.0, offset=1.0)
    scaled = ref * 3
    assert scaled.scale == pytest.approx(6.0)
    assert scaled.offset == pytest.approx(3.0)


def test_parameter_ref_rmul():
    tok = _Tok('x')
    ref = ParameterRef(tok, scale=1.0)
    scaled = 4 * ref
    assert scaled.scale == pytest.approx(4.0)


def test_parameter_ref_truediv():
    tok = _Tok('x')
    ref = ParameterRef(tok, scale=6.0, offset=3.0)
    divided = ref / 3
    assert divided.scale == pytest.approx(2.0)
    assert divided.offset == pytest.approx(1.0)


def test_parameter_ref_add():
    tok = _Tok('x')
    ref = ParameterRef(tok, scale=1.0, offset=0.0)
    shifted = ref + 5
    assert shifted.offset == pytest.approx(5.0)
    assert shifted.scale == pytest.approx(1.0)


def test_parameter_ref_radd():
    tok = _Tok('x')
    ref = ParameterRef(tok)
    shifted = 5 + ref
    assert shifted.offset == pytest.approx(5.0)


def test_parameter_ref_sub():
    tok = _Tok('x')
    ref = ParameterRef(tok, offset=10.0)
    shifted = ref - 4
    assert shifted.offset == pytest.approx(6.0)


def test_parameter_ref_rsub():
    tok = _Tok('x')
    ref = ParameterRef(tok, scale=1.0, offset=0.0)
    result = 10 - ref
    assert result.scale == pytest.approx(-1.0)
    assert result.offset == pytest.approx(10.0)


def test_parameter_ref_chained():
    tok = _Tok('x')
    ref = ParameterRef(tok) * 2 + 150
    assert ref.resolve({tok: 100.0}) == pytest.approx(350.0)


def test_parameter_ref_unsupported_type_returns_not_implemented():
    tok = _Tok('x')
    ref = ParameterRef(tok)
    assert ref.__mul__('bad') is NotImplemented
    assert ref.__add__('bad') is NotImplemented


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------


def test_uniform_fixed_bounds():
    p = Uniform(0, 1000)
    assert p.low == pytest.approx(0.0)
    assert p.high == pytest.approx(1000.0)


def test_uniform_to_dist_type():
    import numpyro.distributions as dist

    p = Uniform(0, 1000)
    d = p.to_dist({})
    assert isinstance(d, dist.Uniform)


def test_uniform_to_dist_bounds():
    p = Uniform(10, 500)
    d = p.to_dist({})
    assert float(d.low) == pytest.approx(10.0)
    assert float(d.high) == pytest.approx(500.0)


def test_uniform_dependencies_empty():
    p = Uniform(0, 1)
    assert p.dependencies() == set()


def test_uniform_dependencies_with_ref():
    tok = _Tok('x')
    ref = ParameterRef(tok)
    p = Uniform(low=ref, high=1000)
    assert p.dependencies() == {tok}


def test_uniform_to_dist_resolves_ref():
    tok = _Tok('x')
    ref = ParameterRef(tok, scale=2.0, offset=100.0)
    p = Uniform(low=ref, high=1000)
    d = p.to_dist({tok: 50.0})
    assert float(d.low) == pytest.approx(200.0)


def test_uniform_roundtrip():
    p = Uniform(3.0, 7.5)
    d = p.to_dict()
    p2 = Uniform.from_dict(d)
    assert p2.low == pytest.approx(3.0)
    assert p2.high == pytest.approx(7.5)


def test_uniform_repr():
    p = Uniform(0, 1)
    assert 'Uniform' in repr(p)


# ---------------------------------------------------------------------------
# TruncatedNormal
# ---------------------------------------------------------------------------


def test_truncated_normal_construction():
    p = TruncatedNormal(loc=0.0, scale=0.01, low=-0.05, high=0.05)
    assert p.loc == pytest.approx(0.0)
    assert p.scale == pytest.approx(0.01)
    assert p.low == pytest.approx(-0.05)
    assert p.high == pytest.approx(0.05)


def test_truncated_normal_to_dist_type():
    import numpyro.distributions as dist

    p = TruncatedNormal(loc=1.0, scale=0.1, low=0.5, high=1.5)
    d = p.to_dist({})
    assert isinstance(d, dist.Distribution)


def test_truncated_normal_dependencies_empty():
    p = TruncatedNormal(loc=0.0, scale=1.0, low=-3.0, high=3.0)
    assert p.dependencies() == set()


def test_truncated_normal_dependencies_with_ref():
    tok = _Tok('x')
    ref = ParameterRef(tok)
    p = TruncatedNormal(loc=ref, scale=0.1, low=0.0, high=10.0)
    assert tok in p.dependencies()


def test_truncated_normal_roundtrip():
    p = TruncatedNormal(loc=1.0, scale=0.05, low=0.8, high=1.2)
    d = p.to_dict()
    p2 = TruncatedNormal.from_dict(d)
    assert p2.loc == pytest.approx(1.0)
    assert p2.scale == pytest.approx(0.05)
    assert p2.low == pytest.approx(0.8)
    assert p2.high == pytest.approx(1.2)


def test_truncated_normal_repr():
    p = TruncatedNormal(loc=0.0, scale=1.0, low=-3.0, high=3.0)
    assert 'TruncatedNormal' in repr(p)


# ---------------------------------------------------------------------------
# Fixed
# ---------------------------------------------------------------------------


def test_fixed_float():
    p = Fixed(6564.61)
    assert p.value == pytest.approx(6564.61)


def test_fixed_int_coerced_to_float():
    p = Fixed(42)
    assert isinstance(p.value, float)
    assert p.value == pytest.approx(42.0)


def test_fixed_type_error():
    with pytest.raises(TypeError, match='int or float'):
        Fixed('bad')  # type: ignore[arg-type]


def test_fixed_to_dist_returns_none():
    p = Fixed(1.0)
    assert p.to_dist({}) is None


def test_fixed_dependencies_empty():
    p = Fixed(0.0)
    assert p.dependencies() == set()


def test_fixed_roundtrip():
    p = Fixed(3.14)
    d = p.to_dict()
    p2 = Fixed.from_dict(d)
    assert p2.value == pytest.approx(3.14)


def test_fixed_repr():
    p = Fixed(2.718)
    assert repr(p) == 'Fixed(2.718)'


# ---------------------------------------------------------------------------
# prior_from_dict
# ---------------------------------------------------------------------------


def test_prior_from_dict_uniform():
    d = {'type': 'Uniform', 'low': 0.0, 'high': 1.0}
    p = prior_from_dict(d)
    assert isinstance(p, Uniform)


def test_prior_from_dict_truncated_normal():
    d = {'type': 'TruncatedNormal', 'loc': 0.0, 'scale': 1.0, 'low': -3.0, 'high': 3.0}
    p = prior_from_dict(d)
    assert isinstance(p, TruncatedNormal)


def test_prior_from_dict_fixed():
    d = {'type': 'Fixed', 'value': 42.0}
    p = prior_from_dict(d)
    assert isinstance(p, Fixed)
    assert p.value == pytest.approx(42.0)


def test_prior_from_dict_unknown_raises():
    with pytest.raises(KeyError):
        prior_from_dict({'type': 'DoesNotExist'})


# ---------------------------------------------------------------------------
# topological_sort
# ---------------------------------------------------------------------------


def _make_tok(name):
    from unite.line.config import FWHM

    return FWHM(name=name)


def test_topological_sort_independent():
    a = _make_tok('a')
    b = _make_tok('b')
    named_priors = {'a': a.prior, 'b': b.prior}
    param_to_name = {a: 'a', b: 'b'}
    order = topological_sort(named_priors, param_to_name)
    assert set(order) == {'a', 'b'}


def test_topological_sort_chain():
    a = _make_tok('a')
    b = _make_tok('b')
    b.prior = Uniform(low=a * 2 + 150, high=5000)
    named_priors = {'a': a.prior, 'b': b.prior}
    param_to_name = {a: 'a', b: 'b'}
    order = topological_sort(named_priors, param_to_name)
    assert order.index('a') < order.index('b')


def test_topological_sort_diamond():
    # a and b are independent; c depends on both a and b.
    from unite.prior import ParameterRef

    a = _make_tok('a')
    b = _make_tok('b')
    c = _make_tok('c')
    c.prior = Uniform(
        low=ParameterRef(a, scale=1.0) + 0, high=ParameterRef(b, scale=1.0) + 1000
    )
    named_priors = {'a': a.prior, 'b': b.prior, 'c': c.prior}
    param_to_name = {a: 'a', b: 'b', c: 'c'}
    order = topological_sort(named_priors, param_to_name)
    assert order.index('a') < order.index('c')
    assert order.index('b') < order.index('c')


def test_topological_sort_circular_raises():
    from unite.prior import ParameterRef

    a = _make_tok('a')
    b = _make_tok('b')
    a.prior = Uniform(low=ParameterRef(b), high=1000)
    b.prior = Uniform(low=ParameterRef(a), high=1000)
    named_priors = {'a': a.prior, 'b': b.prior}
    param_to_name = {a: 'a', b: 'b'}
    with pytest.raises(ValueError, match='Circular'):
        topological_sort(named_priors, param_to_name)
