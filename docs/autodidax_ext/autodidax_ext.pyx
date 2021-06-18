# cython: language_level=2
# distutils: language = c++

from cpython.pycapsule cimport PyCapsule_New
from libc.stdint cimport uint8_t
cimport numpy as np
import numpy as np
np.import_array()

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
    f(*args)
register_cpu_custom_call_target(b"caller", <void*>(caller))


class ShapeDType:
  def __init__(self, size, dtype, shape):
    self.size = size
    self.dtype = dtype
    self.shape = shape

class PyCallback:
  def __init__(self, f, arg_shapes):
    self.f = f
    self.arg_shapes = arg_shapes

  def __call__(self, *args):
    return self.f(*args)

def emit_callback(c, token, f, *args):
  callback = PyCallback(f, [shape_dtype_spec(c, x) for x in args])
  persist_for_life_of_executable(callback)
  return _ops.CustomCallWithLayout(
      c, b"caller", shape_with_layout=Shape.token_shape(),
      operands=(token, _ops.Constant(c, np.uint64(id(callback))), *args),
      operand_shapes_with_layout=(
          Shape.token_shape(), Shape.array_shape(np.dtype(np.uint64), (), ()),
          *(c.get_shape(x) for x in args)),
      has_side_effect=True)

def shape_dtype_spec(c, x):
  s = c.get_shape(x)
  shape, dtype = s.dimensions(), s.numpy_dtype()
  return ShapeDType(dtype.itemsize * int(np.prod(shape)), dtype, shape)

# TODO: jaxlib needs a way to attach object to executable
leaks = []
persist_for_life_of_executable = leaks.append
