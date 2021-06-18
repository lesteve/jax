# cython: language_level=2
# distutils: language = c++

from cpython.pycapsule cimport PyCapsule_New
from libc.stdint cimport uint8_t
cimport numpy as np
np.import_array()

from collections import namedtuple
import numpy as np
from jaxlib import xla_client
_ops = xla_client.ops
Shape = xla_client.Shape


cdef register_cpu_custom_call_target(fn_name, void* fn):
  cdef const char* name = "xla._CUSTOM_CALL_TARGET"
  xla_client.register_custom_call_target(
      fn_name, PyCapsule_New(fn, name, NULL))

cdef void caller(void *out, const void **data) nogil:
  with gil:
    f = <object> (<void**> data[1])[0]
    args = tuple(
        np.asarray(<const np.uint8_t[:s.size]> data[i+2]).view(s.dtype).reshape(s.shape)
        for i, s in enumerate(f.arg_shapes))
    outs = f(*args)
    # TODO use outs
register_cpu_custom_call_target(b"caller", <void*>(caller))


ShapeDType = namedtuple('ShapeDType', ['size', 'dtype', 'shape'])

class PyCallback:
  def __init__(self, f, arg_shapes, out_shapes):
    self.f = f
    self.arg_shapes = arg_shapes
    self.out_shapes = out_shapes

  def __call__(self, *args):
    return self.f(*args)

def emit_callback(c, token, f, args, out_shapes):
  callback = PyCallback(f, [shape_dtype_spec(c, x) for x in args], out_shapes)
  persist_for_life_of_executable(callback)
  out = _ops.CustomCallWithLayout(
      c, b"caller",
      operand_shapes_with_layout=(
          Shape.token_shape(), Shape.array_shape(np.dtype(np.uint64), (), ()),
          *(c.get_shape(x) for x in args)),
      shape_with_layout=Shape.tuple_shape((Shape.token_shape(), *out_shapes)),
      operands=(token, _ops.Constant(c, np.uint64(id(callback))), *args),
      has_side_effect=True)
  token, *outs = [_ops.GetTupleElement(out, i) for i in range(1 + len(out_shapes))]
  return token, outs

def shape_dtype_spec(c, x):
  s = c.get_shape(x)
  shape, dtype = s.dimensions(), s.numpy_dtype()
  return ShapeDType(dtype.itemsize * int(np.prod(shape)), dtype, shape)

# TODO: jaxlib needs a way to attach object to executable
leaks = []
persist_for_life_of_executable = leaks.append
