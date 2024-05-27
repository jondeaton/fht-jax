from functools import partial, reduce

import jax
import jax.numpy as jnp

import jax._src.test_util as jtu
from jax import core, dtypes
from jax.interpreters import xla
from jax.lib import xla_client

from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call


import hadamard_transform_cuda

_fwht_p = core.Primitive("fwht")
_fwht_p.multiple_results = True
_fwht_p.def_impl(partial(xla.apply_primitive, _fwht_p))

for _name, _value in hadamard_transform_cuda.get_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")


def element_type_to_descriptor_type_mapping(element_type):
    _element_type_to_descriptor_type_mapping = {
        ir.BF16Type.get(): hadamard_transform_cuda.ElementType.BF16,
        ir.F16Type.get(): hadamard_transform_cuda.ElementType.F16,
        ir.F32Type.get(): hadamard_transform_cuda.ElementType.F32,
        ir.F64Type.get(): hadamard_transform_cuda.ElementType.F64,
    }
    return _element_type_to_descriptor_type_mapping.get(element_type)


def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


def _fwht_cuda_lowering(ctx, x):
    x_type = ir.RankedTensorType(x.type)
    x_shape = x_type.shape
    batch_size, n = x_type.shape

    opaque = hadamard_transform_cuda.create_hadamard_transform_descriptor(
        batch_size,
        n,
        element_type_to_descriptor_type_mapping(x_type.element_type),
    )
    out = custom_call(
        b"hadamard_transform",
        result_types=[
            ir.RankedTensorType.get(x_shape, x_type.element_type),
        ],
        operands=[x],
        backend_config=opaque,
        operand_layouts=default_layouts(x_shape),
        result_layouts=default_layouts(x_shape),
    ).results
    return out


mlir.register_lowering(
    _fwht_p,
    _fwht_cuda_lowering,
    platform="gpu",
)

# Abstract

from jax.core import ShapedArray


def _fwht_abstract(x):
    x_dtype = dtypes.canonicalize_dtype(x.dtype)
    return (ShapedArray(x.shape, x_dtype, named_shape=x.named_shape),)


_fwht_p.def_abstract_eval(_fwht_abstract)


# VJP


def _fwht_fwd(x):
    (out,) = _fwht_p.bind(x)
    return out, ()


def _fwht_bwd(res, g):
    del res # nothing
    (grad,) = _fwht_p.bind(g)
    return (grad,)


@jax.custom_vjp
def fwht(x):
    out, _ = _fwht_fwd(x)
    return out


fwht.defvjp(_fwht_fwd, _fwht_bwd)
