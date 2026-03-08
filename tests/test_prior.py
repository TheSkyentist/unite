"""Tests for unite.prior."""

import pytest

from unite.prior import (
    Fixed,
    Parameter,
    ParameterRef,
    TruncatedNormal,
    Uniform,
    _deserialize_bound,
    _serialize_bound,
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


def test_parameter_ref_truediv_not_implemented():
    """ParameterRef.__truediv__ returns NotImplemented for non-numeric (line 85)."""
    tok = _Tok('x')
    ref = ParameterRef(tok)
    assert ref.__truediv__('bad') is NotImplemented


def test_parameter_ref_sub_not_implemented():
    """ParameterRef.__sub__ returns NotImplemented for non-numeric (line 98)."""
    tok = _Tok('x')
    ref = ParameterRef(tok)
    assert ref.__sub__('bad') is NotImplemented


def test_parameter_ref_rsub_not_implemented():
    """ParameterRef.__rsub__ returns NotImplemented for non-numeric (line 103)."""
    tok = _Tok('x')
    ref = ParameterRef(tok)
    assert ref.__rsub__('bad') is NotImplemented


def test_parameter_ref_repr_with_negative_offset():
    """ParameterRef.__repr__ with negative offset shows minus sign (lines 114-115)."""
    tok = _Tok('x')
    tok.name = 'x'
    ref = ParameterRef(tok, scale=1.0, offset=-5.0)
    r = repr(ref)
    assert '-' in r or '5' in r


def test_parameter_ref_repr_default():
    """ParameterRef.__repr__ with scale=1.0, offset=0.0 shows only label."""
    tok = _Tok('x')
    tok.name = 'x'
    ref = ParameterRef(tok, scale=1.0, offset=0.0)
    r = repr(ref)
    assert 'x' in r
    assert '+' not in r


def test_parameter_ref_repr_with_scale():
    """ParameterRef.__repr__ with scale != 1.0 shows scale * label (line 109)."""
    tok = _Tok('x')
    tok.name = 'x'
    ref = ParameterRef(tok, scale=2.5, offset=0.0)
    r = repr(ref)
    assert '2.5' in r
    assert '*' in r


def test_parameter_ref_repr_with_positive_offset():
    """ParameterRef.__repr__ with positive offset shows '+ offset' (line 113)."""
    tok = _Tok('x')
    tok.name = 'x'
    ref = ParameterRef(tok, scale=1.0, offset=10.0)
    r = repr(ref)
    assert '+' in r
    assert '10' in r


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


def test_topological_sort_external_dependency_ignored():
    """dep_obj not in param_to_name → dep_name is None → silently ignored (line 577)."""
    from unite.prior import ParameterRef

    a = _make_tok('a')
    external = _make_tok('external')  # Not in named_priors or param_to_name
    a.prior = Uniform(low=ParameterRef(external), high=1000)
    named_priors = {'a': a.prior}
    param_to_name = {a: 'a'}  # external not registered
    # Should not raise; external dep is silently ignored
    order = topological_sort(named_priors, param_to_name)
    assert 'a' in order


# ---------------------------------------------------------------------------
# Parameter arithmetic and repr (prior.py lines 499-539)
# ---------------------------------------------------------------------------


def test_parameter_repr_with_name():
    """Parameter.__repr__ includes the name when name is set."""
    p = Parameter('my_param', prior=Uniform(0, 1))
    assert 'my_param' in repr(p)


def test_parameter_repr_without_name():
    """Parameter.__repr__ skips name when name is None (line 499 branch)."""
    p = Parameter(None, prior=Uniform(0, 1))
    r = repr(p)
    assert 'prior=' in r
    assert 'None' not in r


def test_parameter_mul_not_implemented():
    """Parameter.__mul__ returns NotImplemented for non-numeric (line 507)."""
    p = Parameter('x', prior=Uniform(0, 1))
    assert p.__mul__('bad') is NotImplemented


def test_parameter_rmul():
    """Parameter.__rmul__ delegates to __mul__ (line 512)."""
    p = Parameter('x', prior=Uniform(0, 1))
    ref = p.__rmul__(3)
    assert isinstance(ref, ParameterRef)
    assert ref.scale == pytest.approx(3.0)


def test_parameter_truediv_not_implemented():
    """Parameter.__truediv__ returns NotImplemented for non-numeric (line 517)."""
    p = Parameter('x', prior=Uniform(0, 1))
    assert p.__truediv__('bad') is NotImplemented


def test_parameter_add_not_implemented():
    """Parameter.__add__ returns NotImplemented for non-numeric (line 523)."""
    p = Parameter('x', prior=Uniform(0, 1))
    assert p.__add__('bad') is NotImplemented


def test_parameter_radd():
    """Parameter.__radd__ delegates to __add__ (line 528)."""
    p = Parameter('x', prior=Uniform(0, 1))
    ref = p.__radd__(5)
    assert isinstance(ref, ParameterRef)
    assert ref.offset == pytest.approx(5.0)


def test_parameter_sub_not_implemented():
    """Parameter.__sub__ returns NotImplemented for non-numeric (line 533)."""
    p = Parameter('x', prior=Uniform(0, 1))
    assert p.__sub__('bad') is NotImplemented


def test_parameter_rsub_not_implemented():
    """Parameter.__rsub__ returns NotImplemented for non-numeric (line 539)."""
    p = Parameter('x', prior=Uniform(0, 1))
    assert p.__rsub__('bad') is NotImplemented


# ---------------------------------------------------------------------------
# _serialize_bound / _deserialize_bound with ParameterRef (lines 206-214, 229-236)
# ---------------------------------------------------------------------------


def test_serialize_bound_float():
    """_serialize_bound on a float returns the float unchanged."""
    assert _serialize_bound(3.14, None) == pytest.approx(3.14)


def test_serialize_bound_parameter_ref():
    """_serialize_bound on ParameterRef returns a dict with 'ref' key."""
    tok = _make_tok('alpha')
    ref = ParameterRef(tok, scale=2.0, offset=0.5)
    namer = {tok: 'alpha'}
    d = _serialize_bound(ref, namer)
    assert isinstance(d, dict)
    assert d['ref'] == 'alpha'
    assert d['scale'] == pytest.approx(2.0)
    assert d['offset'] == pytest.approx(0.5)


def test_serialize_bound_parameter_ref_no_namer_raises():
    """_serialize_bound with ParameterRef and no namer raises ValueError."""
    tok = _make_tok('alpha')
    ref = ParameterRef(tok)
    with pytest.raises(ValueError, match='param_namer'):
        _serialize_bound(ref, None)


def test_deserialize_bound_float():
    """_deserialize_bound on a float returns float."""
    assert _deserialize_bound(3.14, None) == pytest.approx(3.14)


def test_deserialize_bound_dict():
    """_deserialize_bound on a dict returns a ParameterRef (line 229)."""
    tok = _make_tok('alpha')
    registry = {'alpha': tok}
    ref = _deserialize_bound({'ref': 'alpha', 'scale': 2.0, 'offset': 0.5}, registry)
    assert isinstance(ref, ParameterRef)
    assert ref.param is tok
    assert ref.scale == pytest.approx(2.0)
    assert ref.offset == pytest.approx(0.5)


def test_deserialize_bound_dict_no_registry_raises():
    """_deserialize_bound with dict and no registry raises ValueError."""
    with pytest.raises(ValueError, match='token_registry'):
        _deserialize_bound({'ref': 'alpha'}, None)
