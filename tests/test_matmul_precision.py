"""Matrix products in the model path must not depend on JAX's matmul default.

On GPUs with TF32 tensor cores (A100, H100) the float32 default truncates matmul
operands to a 10-bit mantissa, which puts a noise floor in the likelihood and its
gradients.  Every product therefore goes through :func:`unite._utils.mm` (or
passes ``precision=HIGHEST`` itself).  CI has no GPU, so these tests check the
traced program and the source rather than the numerics.
"""

import ast
from pathlib import Path

import jax
import pytest
from numpyro import handlers
from test_static_calibration import _build, _params

import unite
from unite._utils import mm

_HIGHEST = jax.lax.Precision.HIGHEST

# NumPy-only code that runs after sampling and never touches the device.
_NUMPY_ONLY = {'results.py', 'continuum/fit.py'}
_PRECISION_CALLS = {'dot', 'matmul', 'einsum', 'tensordot', 'inner', 'vdot'}


def _dot_generals(jaxpr):
    """Yield every ``dot_general`` equation, descending into nested jaxprs."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == 'dot_general':
            yield eqn
        for val in eqn.params.values():
            for sub in val if isinstance(val, (tuple, list)) else (val,):
                inner = getattr(sub, 'jaxpr', sub)
                if hasattr(inner, 'eqns'):
                    yield from _dot_generals(inner)


def _assert_all_highest(jaxpr):
    eqns = list(_dot_generals(jaxpr))
    bad = [str(e) for e in eqns if e.params['precision'] != (_HIGHEST, _HIGHEST)]
    assert not bad, f'{len(bad)} dot_general without HIGHEST precision:\n' + '\n'.join(
        bad
    )
    return len(eqns)


def test_mm_is_highest():
    a = jax.numpy.ones((3, 4))
    b = jax.numpy.ones((4, 5))
    assert _assert_all_highest(jax.make_jaxpr(mm)(a, b)) == 1


@pytest.mark.parametrize('mode', ['analytic', 'convolution'])
def test_model_matmuls_are_highest(mode):
    """Likelihood and gradient (incl. absorbers, continuum) use HIGHEST only."""
    model_fn, args = _build(mode, fixed=False, gap=False)
    params = _params(model_fn, args)

    def loglik(p):
        tr = handlers.trace(handlers.substitute(model_fn, data=p)).get_trace(args)
        return tr['obs_s']['fn'].log_prob(tr['obs_s']['value']).sum()

    for fn in (loglik, jax.value_and_grad(loglik)):
        n = _assert_all_highest(jax.make_jaxpr(fn)(params))
        # Guard against the walk silently finding nothing.
        assert n > 0


def _source_files():
    root = Path(unite.__file__).parent
    for path in sorted(root.rglob('*.py')):
        rel = path.relative_to(root).as_posix()
        if rel not in _NUMPY_ONLY and rel != '_utils.py':
            yield rel, path


@pytest.mark.parametrize(
    ('rel', 'path'),
    list(_source_files()),
    ids=lambda v: v if isinstance(v, str) else '',
)
def test_no_unprotected_matmul_in_source(rel, path):
    """No bare ``@`` and no ``jnp.dot``-style call without ``precision=``."""
    tree = ast.parse(path.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            offenders.append(f'{rel}:{node.lineno}: bare @ (use unite._utils.mm)')
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _PRECISION_CALLS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'jnp'
            and not any(k.arg == 'precision' for k in node.keywords)
        ):
            offenders.append(
                f'{rel}:{node.lineno}: jnp.{node.func.attr} without precision='
            )
    assert not offenders, '\n'.join(offenders)
