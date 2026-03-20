"""Tests for unite.prior."""

import pytest

from unite.prior import (
    Fixed,
    Parameter,
    TruncatedNormal,
    Uniform,
    _BinOpExpr,
    _deserialize_bound,
    _Expr,
    _LiteralLeaf,
    _ParamLeaf,
    _serialize_bound,
    prior_from_dict,
    topological_sort,
)

# ---------------------------------------------------------------------------
# Expression tree — single-parameter arithmetic
# ---------------------------------------------------------------------------


def _make_tok(name):
    from unite.line.config import FWHM

    return FWHM(name=name)


def test_param_mul_scalar_returns_expr():
    tok = _make_tok('x')
    expr = tok * 2
    assert isinstance(expr, _Expr)
    assert expr.resolve({tok: 3.0}) == pytest.approx(6.0)


def test_scalar_rmul_param_returns_expr():
    tok = _make_tok('x')
    expr = 4 * tok
    assert isinstance(expr, _Expr)
    assert expr.resolve({tok: 3.0}) == pytest.approx(12.0)


def test_param_truediv_scalar_returns_expr():
    tok = _make_tok('x')
    expr = tok / 2
    assert isinstance(expr, _Expr)
    assert expr.resolve({tok: 6.0}) == pytest.approx(3.0)


def test_param_add_scalar_returns_expr():
    tok = _make_tok('x')
    expr = tok + 5
    assert isinstance(expr, _Expr)
    assert expr.resolve({tok: 10.0}) == pytest.approx(15.0)


def test_scalar_radd_param_returns_expr():
    tok = _make_tok('x')
    expr = 5 + tok
    assert isinstance(expr, _Expr)
    assert expr.resolve({tok: 10.0}) == pytest.approx(15.0)


def test_param_sub_scalar_returns_expr():
    tok = _make_tok('x')
    expr = tok - 4
    assert isinstance(expr, _Expr)
    assert expr.resolve({tok: 10.0}) == pytest.approx(6.0)


def test_scalar_rsub_param_returns_expr():
    tok = _make_tok('x')
    expr = 10 - tok
    assert isinstance(expr, _Expr)
    assert expr.resolve({tok: 3.0}) == pytest.approx(7.0)


def test_chained_single_param():
    """param * 2 + 150 chains correctly."""
    tok = _make_tok('x')
    expr = tok * 2 + 150
    assert expr.resolve({tok: 100.0}) == pytest.approx(350.0)


# ---------------------------------------------------------------------------
# Expression tree — two-parameter arithmetic
# ---------------------------------------------------------------------------


def test_param_mul_param():
    a = _make_tok('a')
    b = _make_tok('b')
    expr = a * b
    assert isinstance(expr, _Expr)
    assert expr.resolve({a: 3.0, b: 4.0}) == pytest.approx(12.0)


def test_param_div_param():
    a = _make_tok('a')
    b = _make_tok('b')
    expr = a / b
    assert isinstance(expr, _Expr)
    assert expr.resolve({a: 10.0, b: 2.0}) == pytest.approx(5.0)


def test_three_param_ratio():
    """flux_a * flux_b / flux_c — the core use case."""
    from unite.line.config import Flux

    flux_a = Flux('a', prior=Uniform(0, 10))
    flux_b = Flux('b', prior=Uniform(0, 10))
    flux_c = Flux('c', prior=Uniform(0.1, 10))
    expr = flux_a * flux_b / flux_c
    assert isinstance(expr, _Expr)
    assert expr.resolve({flux_a: 2.0, flux_b: 3.0, flux_c: 6.0}) == pytest.approx(1.0)


def test_two_param_dependencies():
    a = _make_tok('a')
    b = _make_tok('b')
    expr = a * b
    assert expr.dependencies() == {a, b}


def test_three_param_dependencies():
    from unite.line.config import Flux

    flux_a = Flux('a', prior=Uniform(0, 10))
    flux_b = Flux('b', prior=Uniform(0, 10))
    flux_c = Flux('c', prior=Uniform(0.1, 10))
    expr = flux_a * flux_b / flux_c
    assert expr.dependencies() == {flux_a, flux_b, flux_c}


def test_expr_mul_param():
    """_Expr * Parameter creates a new BinOpExpr."""
    a = _make_tok('a')
    b = _make_tok('b')
    expr = (a * 2) * b
    assert expr.resolve({a: 3.0, b: 4.0}) == pytest.approx(24.0)


def test_param_mul_expr():
    """Parameter * _Expr creates a new BinOpExpr."""
    a = _make_tok('a')
    b = _make_tok('b')
    expr = a * (b + 1)
    assert expr.resolve({a: 3.0, b: 4.0}) == pytest.approx(15.0)


def test_expr_div_param():
    """_Expr / Parameter."""
    a = _make_tok('a')
    b = _make_tok('b')
    expr = (a * 6) / b
    assert expr.resolve({a: 2.0, b: 3.0}) == pytest.approx(4.0)


def test_param_rtruediv_scalar():
    """scalar / param."""
    tok = _make_tok('x')
    expr = 10.0 / tok
    assert expr.resolve({tok: 2.0}) == pytest.approx(5.0)


# _Expr op scalar/expr — coverage for _Expr arithmetic branches


def test_expr_mul_scalar():
    """_Expr * scalar hits _Expr.__mul__(scalar)."""
    a, b = _make_tok('a'), _make_tok('b')
    expr = (a + b) * 2
    assert expr.resolve({a: 3.0, b: 4.0}) == pytest.approx(14.0)


def test_expr_mul_expr():
    """_Expr * _Expr hits _Expr.__mul__(_Expr)."""
    a, b, c, d = [_make_tok(x) for x in 'abcd']
    expr = (a + b) * (c + d)
    assert expr.resolve({a: 1.0, b: 2.0, c: 3.0, d: 4.0}) == pytest.approx(21.0)


def test_scalar_rmul_expr():
    """scalar * _Expr hits _Expr.__rmul__(scalar)."""
    a, b = _make_tok('a'), _make_tok('b')
    expr = 2 * (a + b)
    assert expr.resolve({a: 3.0, b: 4.0}) == pytest.approx(14.0)


def test_expr_truediv_scalar():
    """_Expr / scalar hits _Expr.__truediv__(scalar)."""
    a, b = _make_tok('a'), _make_tok('b')
    expr = (a + b) / 2
    assert expr.resolve({a: 3.0, b: 7.0}) == pytest.approx(5.0)


def test_scalar_rtruediv_expr():
    """scalar / _Expr hits _Expr.__rtruediv__(scalar)."""
    a, b = _make_tok('a'), _make_tok('b')
    expr = 10.0 / (a + b)
    assert expr.resolve({a: 2.0, b: 3.0}) == pytest.approx(2.0)


def test_expr_add_expr():
    """_Expr + _Expr hits _Expr.__add__(_Expr)."""
    a, b, c, d = [_make_tok(x) for x in 'abcd']
    expr = (a + b) + (c + d)
    assert expr.resolve({a: 1.0, b: 2.0, c: 3.0, d: 4.0}) == pytest.approx(10.0)


def test_expr_add_param():
    """_Expr + Parameter hits _Expr.__add__(Parameter)."""
    a, b, c = [_make_tok(x) for x in 'abc']
    expr = (a + b) + c
    assert expr.resolve({a: 1.0, b: 2.0, c: 3.0}) == pytest.approx(6.0)


def test_scalar_radd_expr():
    """scalar + _Expr hits _Expr.__radd__(scalar)."""
    a, b = _make_tok('a'), _make_tok('b')
    expr = 2 + (a + b)
    assert expr.resolve({a: 3.0, b: 4.0}) == pytest.approx(9.0)


def test_expr_sub_expr():
    """_Expr - _Expr hits _Expr.__sub__(_Expr)."""
    a, b, c, d = [_make_tok(x) for x in 'abcd']
    expr = (a + b) - (c + d)
    assert expr.resolve({a: 10.0, b: 0.0, c: 3.0, d: 2.0}) == pytest.approx(5.0)


def test_expr_sub_param():
    """_Expr - Parameter hits _Expr.__sub__(Parameter)."""
    a, b, c = [_make_tok(x) for x in 'abc']
    expr = (a + b) - c
    assert expr.resolve({a: 10.0, b: 5.0, c: 3.0}) == pytest.approx(12.0)


def test_scalar_rsub_expr():
    """scalar - _Expr hits _Expr.__rsub__(scalar)."""
    a, b = _make_tok('a'), _make_tok('b')
    expr = 20 - (a + b)
    assert expr.resolve({a: 3.0, b: 4.0}) == pytest.approx(13.0)


# Parameter op Parameter / _Expr — coverage for Parameter arithmetic branches


def test_param_add_param():
    """Parameter + Parameter."""
    a, b = _make_tok('a'), _make_tok('b')
    expr = a + b
    assert expr.resolve({a: 3.0, b: 4.0}) == pytest.approx(7.0)


def test_param_add_expr():
    """Parameter + _Expr hits Parameter.__add__(_Expr)."""
    a, b, c = [_make_tok(x) for x in 'abc']
    expr = a + (b + c)
    assert expr.resolve({a: 1.0, b: 2.0, c: 3.0}) == pytest.approx(6.0)


def test_param_sub_param():
    """Parameter - Parameter."""
    a, b = _make_tok('a'), _make_tok('b')
    expr = a - b
    assert expr.resolve({a: 10.0, b: 3.0}) == pytest.approx(7.0)


def test_param_sub_expr():
    """Parameter - _Expr hits Parameter.__sub__(_Expr)."""
    a, b, c = [_make_tok(x) for x in 'abc']
    expr = a - (b + c)
    assert expr.resolve({a: 10.0, b: 2.0, c: 1.0}) == pytest.approx(7.0)


def test_param_truediv_expr():
    """Parameter / _Expr hits Parameter.__truediv__(_Expr)."""
    a, b, c = [_make_tok(x) for x in 'abc']
    expr = a / (b + c)
    assert expr.resolve({a: 6.0, b: 2.0, c: 1.0}) == pytest.approx(2.0)


def test_deserialize_bound_binop():
    """_deserialize_bound with 'op' dict reconstructs a _BinOpExpr."""
    from unite.line.config import FWHM

    tok = FWHM('x', prior=Uniform(0, 100))
    tok.name = 'fwhm_x'
    registry = {tok.name: tok}
    d = {'op': '+', 'left': {'ref': 'fwhm_x'}, 'right': 1.5}
    result = _deserialize_bound(d, registry)
    assert isinstance(result, _BinOpExpr)
    assert result.resolve({tok: 10.0}) == pytest.approx(11.5)


# ---------------------------------------------------------------------------
# unsupported operand types return NotImplemented
# ---------------------------------------------------------------------------


def test_param_mul_unsupported():
    tok = _make_tok('x')
    assert tok.__mul__('bad') is NotImplemented


def test_param_truediv_unsupported():
    tok = _make_tok('x')
    assert tok.__truediv__('bad') is NotImplemented


def test_param_add_unsupported():
    tok = _make_tok('x')
    assert tok.__add__('bad') is NotImplemented


def test_param_sub_unsupported():
    tok = _make_tok('x')
    assert tok.__sub__('bad') is NotImplemented


def test_param_rsub_unsupported():
    tok = _make_tok('x')
    assert tok.__rsub__('bad') is NotImplemented


def test_expr_mul_unsupported():
    tok = _make_tok('x')
    expr = tok + 0
    assert expr.__mul__('bad') is NotImplemented


def test_expr_truediv_unsupported():
    tok = _make_tok('x')
    expr = tok + 0
    assert expr.__truediv__('bad') is NotImplemented


def test_expr_add_unsupported():
    tok = _make_tok('x')
    expr = tok + 0
    assert expr.__add__('bad') is NotImplemented


def test_expr_sub_unsupported():
    tok = _make_tok('x')
    expr = tok + 0
    assert expr.__sub__('bad') is NotImplemented


def test_expr_rsub_unsupported():
    tok = _make_tok('x')
    expr = tok + 0
    assert expr.__rsub__('bad') is NotImplemented


# ---------------------------------------------------------------------------
# _Expr repr
# ---------------------------------------------------------------------------


def test_param_leaf_repr():
    tok = _make_tok('x')
    leaf = _ParamLeaf(tok)
    assert 'x' in repr(leaf)


def test_binop_repr():
    tok = _make_tok('x')
    expr = tok * 2.0 + 150.0
    r = repr(expr)
    assert '*' in r or '+' in r
    assert '2.0' in r
    assert '150.0' in r


def test_binop_invalid_op_raises():
    tok = _make_tok('x')
    leaf = _ParamLeaf(tok)
    with pytest.raises(ValueError, match='Unknown operator'):
        _BinOpExpr('%', leaf, _LiteralLeaf(1.0))


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


def test_uniform_dependencies_with_expr():
    tok = _make_tok('x')
    p = Uniform(low=tok, high=1000)
    assert tok in p.dependencies()


def test_uniform_to_dist_resolves_expr():
    tok = _make_tok('x')
    p = Uniform(low=tok * 2.0 + 100.0, high=1000)
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


def test_truncated_normal_dependencies_with_expr():
    tok = _make_tok('x')
    p = TruncatedNormal(loc=tok, scale=0.1, low=0.0, high=10.0)
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
    with pytest.raises(TypeError, match='int, float, or a parameter expression'):
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


def test_fixed_param_ref():
    """Fixed(param) ties a parameter deterministically to another."""
    from unite.line.config import FWHM

    narrow = FWHM('narrow', prior=Uniform(50, 300))
    broad = FWHM('broad', prior=Fixed(narrow))
    assert narrow in broad.prior.dependencies()
    assert broad.prior.resolved_value({narrow: 200.0}) == pytest.approx(200.0)


def test_fixed_two_param_ratio():
    """Fixed(flux_a * flux_b / flux_c) — three-parameter ratio constraint."""
    from unite.line.config import Flux

    flux_a = Flux('a', prior=Uniform(0, 10))
    flux_b = Flux('b', prior=Uniform(0, 10))
    flux_c = Flux('c', prior=Uniform(0.1, 10))
    flux_d = Flux('d', prior=Fixed(flux_a * flux_b / flux_c))

    assert flux_a in flux_d.prior.dependencies()
    assert flux_b in flux_d.prior.dependencies()
    assert flux_c in flux_d.prior.dependencies()
    assert flux_d.prior.resolved_value(
        {flux_a: 2.0, flux_b: 3.0, flux_c: 6.0}
    ) == pytest.approx(1.0)


def test_fixed_expr_roundtrip():
    """Fixed(flux_a * flux_b / flux_c) survives serialization round-trip."""
    from unite.line.config import Flux, LineConfiguration

    flux_5007_narrow = Flux('5007_narrow', prior=Uniform(0, 10))
    flux_5007_broad = Flux('5007_broad', prior=Uniform(0, 10))
    flux_4363_narrow = Flux('4363_narrow', prior=Uniform(0, 10))
    flux_4363_broad = Flux(
        '4363_broad', prior=Fixed(flux_4363_narrow * flux_5007_broad / flux_5007_narrow)
    )

    import astropy.units as u

    lc = LineConfiguration()
    lc.add_line('OIII_5007_n', 5007.0 * u.AA, flux=flux_5007_narrow)
    lc.add_line('OIII_5007_b', 5007.0 * u.AA, flux=flux_5007_broad)
    lc.add_line('OIII_4363_n', 4363.0 * u.AA, flux=flux_4363_narrow)
    lc.add_line('OIII_4363_b', 4363.0 * u.AA, flux=flux_4363_broad)

    d = lc.to_dict()
    lc2 = LineConfiguration.from_dict(d)

    f5007n = lc2._entries[0].flux
    f5007b = lc2._entries[1].flux
    f4363n = lc2._entries[2].flux
    f4363b = lc2._entries[3].flux

    # All three dependencies must survive
    assert f5007n in f4363b.prior.dependencies()
    assert f5007b in f4363b.prior.dependencies()
    assert f4363n in f4363b.prior.dependencies()

    # Resolution must be correct: 2 * 3 / 6 = 1
    val = f4363b.prior.resolved_value({f4363n: 2.0, f5007b: 3.0, f5007n: 6.0})
    assert val == pytest.approx(1.0)


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
    """a and b are independent; c depends on both."""
    a = _make_tok('a')
    b = _make_tok('b')
    c = _make_tok('c')
    c.prior = Uniform(low=a + 0, high=b + 1000)
    named_priors = {'a': a.prior, 'b': b.prior, 'c': c.prior}
    param_to_name = {a: 'a', b: 'b', c: 'c'}
    order = topological_sort(named_priors, param_to_name)
    assert order.index('a') < order.index('c')
    assert order.index('b') < order.index('c')


def test_topological_sort_circular_raises():
    a = _make_tok('a')
    b = _make_tok('b')
    a.prior = Uniform(low=b, high=1000)
    b.prior = Uniform(low=a, high=1000)
    named_priors = {'a': a.prior, 'b': b.prior}
    param_to_name = {a: 'a', b: 'b'}
    with pytest.raises(ValueError, match='Circular'):
        topological_sort(named_priors, param_to_name)


def test_topological_sort_external_dependency_ignored():
    """dep_obj not in param_to_name is silently ignored."""
    a = _make_tok('a')
    external = _make_tok('external')
    a.prior = Uniform(low=external, high=1000)
    named_priors = {'a': a.prior}
    param_to_name = {a: 'a'}
    order = topological_sort(named_priors, param_to_name)
    assert 'a' in order


def test_topological_sort_three_param_ratio():
    """Ratio constraint: c depends on both a and b."""
    from unite.line.config import Flux

    a = Flux('a', prior=Uniform(0, 10))
    b = Flux('b', prior=Uniform(0, 10))
    c = Flux('c', prior=Fixed(a * b / (b + 1)))  # depends on a and b
    named_priors = {'flux_a': a.prior, 'flux_b': b.prior, 'flux_c': c.prior}
    param_to_name = {a: 'flux_a', b: 'flux_b', c: 'flux_c'}
    order = topological_sort(named_priors, param_to_name)
    assert order.index('flux_a') < order.index('flux_c')
    assert order.index('flux_b') < order.index('flux_c')


# ---------------------------------------------------------------------------
# Parameter arithmetic and repr
# ---------------------------------------------------------------------------


def test_parameter_repr_with_name():
    """Parameter.__repr__ includes the name when name is set."""
    p = Parameter('my_param', prior=Uniform(0, 1))
    assert 'my_param' in repr(p)


def test_parameter_repr_without_name():
    """Parameter.__repr__ skips name when name is None."""
    p = Parameter(None, prior=Uniform(0, 1))
    r = repr(p)
    assert 'prior=' in r
    assert 'None' not in r


def test_parameter_mul_returns_expr():
    p = _make_tok('x')
    expr = p * 3
    assert isinstance(expr, _Expr)
    assert expr.resolve({p: 2.0}) == pytest.approx(6.0)


def test_parameter_rmul_returns_expr():
    p = _make_tok('x')
    expr = p.__rmul__(3)
    assert isinstance(expr, _Expr)
    assert expr.resolve({p: 1.0}) == pytest.approx(3.0)


def test_parameter_truediv_returns_expr():
    p = _make_tok('x')
    expr = p / 4
    assert isinstance(expr, _Expr)
    assert expr.resolve({p: 8.0}) == pytest.approx(2.0)


def test_parameter_add_returns_expr():
    p = _make_tok('x')
    expr = p + 5
    assert isinstance(expr, _Expr)
    assert expr.resolve({p: 10.0}) == pytest.approx(15.0)


def test_parameter_radd_returns_expr():
    p = _make_tok('x')
    expr = p.__radd__(5)
    assert isinstance(expr, _Expr)
    assert expr.resolve({p: 0.0}) == pytest.approx(5.0)


def test_parameter_sub_returns_expr():
    p = _make_tok('x')
    expr = p - 3
    assert isinstance(expr, _Expr)
    assert expr.resolve({p: 10.0}) == pytest.approx(7.0)


def test_parameter_rsub_returns_expr():
    p = _make_tok('x')
    expr = p.__rsub__(10)
    assert isinstance(expr, _Expr)
    assert expr.resolve({p: 3.0}) == pytest.approx(7.0)


def test_parameter_mul_unsupported():
    p = Parameter('x', prior=Uniform(0, 1))
    assert p.__mul__('bad') is NotImplemented


def test_parameter_add_unsupported():
    p = Parameter('x', prior=Uniform(0, 1))
    assert p.__add__('bad') is NotImplemented


def test_parameter_sub_unsupported():
    p = Parameter('x', prior=Uniform(0, 1))
    assert p.__sub__('bad') is NotImplemented


def test_parameter_rsub_unsupported():
    p = Parameter('x', prior=Uniform(0, 1))
    assert p.__rsub__('bad') is NotImplemented


# ---------------------------------------------------------------------------
# _serialize_bound / _deserialize_bound
# ---------------------------------------------------------------------------


def test_serialize_bound_float():
    """_serialize_bound on a float returns the float unchanged."""
    assert _serialize_bound(3.14, None) == pytest.approx(3.14)


def test_serialize_bound_param_leaf():
    """_serialize_bound on _ParamLeaf returns a dict with 'ref' key."""
    tok = _make_tok('alpha')
    leaf = _ParamLeaf(tok)
    namer = {tok: 'alpha'}
    d = _serialize_bound(leaf, namer)
    assert isinstance(d, dict)
    assert d == {'ref': 'alpha'}


def test_serialize_bound_binop():
    """_serialize_bound on a _BinOpExpr returns nested op dict."""
    tok = _make_tok('alpha')
    expr = tok * 2.0
    namer = {tok: 'alpha'}
    d = _serialize_bound(expr, namer)
    assert d['op'] == '*'
    assert d['left'] == {'ref': 'alpha'}
    assert d['right'] == pytest.approx(2.0)


def test_serialize_bound_expr_no_namer_raises():
    """_serialize_bound with _Expr and no namer raises ValueError."""
    tok = _make_tok('alpha')
    expr = tok + 1
    with pytest.raises(ValueError, match='param_namer'):
        _serialize_bound(expr, None)


def test_deserialize_bound_float():
    """_deserialize_bound on a float returns float."""
    assert _deserialize_bound(3.14, None) == pytest.approx(3.14)


def test_deserialize_bound_ref_dict():
    """_deserialize_bound on a {'ref': ...} dict returns _ParamLeaf."""
    tok = _make_tok('alpha')
    registry = {'alpha': tok}
    result = _deserialize_bound({'ref': 'alpha'}, registry)
    assert isinstance(result, _ParamLeaf)
    assert result.param is tok


def test_deserialize_bound_binop_dict():
    """_deserialize_bound on a nested op dict returns _BinOpExpr."""
    tok = _make_tok('alpha')
    registry = {'alpha': tok}
    d = {'op': '*', 'left': {'ref': 'alpha'}, 'right': 2.0}
    result = _deserialize_bound(d, registry)
    assert isinstance(result, _BinOpExpr)
    assert result.resolve({tok: 3.0}) == pytest.approx(6.0)


def test_deserialize_bound_dict_no_registry_raises():
    """_deserialize_bound with dict and no registry raises ValueError."""
    with pytest.raises(ValueError, match='token_registry'):
        _deserialize_bound({'ref': 'alpha'}, None)


def test_serialize_deserialize_three_param_ratio():
    """Three-parameter ratio expression round-trips through serialize/deserialize."""
    from unite.line.config import Flux

    a = Flux('a', prior=Uniform(0, 10))
    b = Flux('b', prior=Uniform(0, 10))
    c = Flux('c', prior=Uniform(0.1, 10))
    expr = a * b / c
    namer = {a: 'a', b: 'b', c: 'c'}
    d = _serialize_bound(expr, namer)

    registry = {'a': a, 'b': b, 'c': c}
    expr2 = _deserialize_bound(d, registry)
    assert isinstance(expr2, _Expr)
    assert expr2.resolve({a: 2.0, b: 3.0, c: 6.0}) == pytest.approx(1.0)
