import os.path

import theano
from theano import config
from theano import tensor as T
from theano.sandbox.cuda import GpuOp, as_cuda_ndarray_variable, device_properties
from theano.sandbox.cuda.basic_ops import (gpu_contiguous, GpuFromHost, HostFromGpu,
                                           gpu_from_host, host_from_gpu, GpuJoin)
from theano.sandbox.cuda.opt import register_opt, local_optimizer
from theano.sandbox.cuda.type import CudaNdarrayType


def strip_transfer(variable):
    """
    Forcefully strip off a GPU<->host transfer from the given variable.

    If `variable` is not directly the result of a GPU<->host transfer, this
    function returns `variable` unchanged.
    """

    if isinstance(variable.owner.op, (GpuFromHost, HostFromGpu)):
        return variable.owner.inputs[0]
    return variable


class AdvancedSubtensor1Floats(T.subtensor.AdvancedSubtensor1):
    """
    Dummy class which supports subtensor indexing with float indices.

    This allows us to do GPU subtensor indexing using indices drawn from a
    float32 GPU shared variable.
    """

    def __init__(self, tag=None):
        super(AdvancedSubtensor1Floats, self).__init__()
        self._tag = tag

    def make_node(self, x, ilist):
        # copy-paste of super.make_node, but without the int type constraint
        x_ = T.as_tensor_variable(x)
        ilist_ = T.as_tensor_variable(ilist)
        #if ilist_.type.dtype[:3] not in ('int', 'uin'):
        #    raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        bcast = (ilist_.broadcastable[0],) + x_.broadcastable[1:]
        return theano.gof.Apply(self, [x_, ilist_],
                                [T.TensorType(dtype=x.dtype, broadcastable=bcast)()])

    def grad(self, inputs, grads):
        x, ilist = inputs
        gz, = grads
        assert len(inputs) == 2
        if self.sparse_grad:
            raise RuntimeError("sparse grad not supported for AdvancedSubtensor1Floats")

        setinc, inpl = self.set_instead_of_inc, self.inplace
        inc_op = AdvancedIncSubtensor1Floats(set_instead_of_inc=setinc, inplace=inpl)
        rval1 = [inc_op(x.zeros_like(), gz, ilist)]
        return rval1 + [T.DisconnectedType()()] * (len(inputs) - 1)


class GpuAdvancedSubtensor1Floats(AdvancedSubtensor1Floats, GpuOp):

    def __init__(self, tag=None):
        self._tag = tag

    def __str__(self):
        return "GpuAdvancedSubtensor1Floats(%s)" % self._tag

    def make_node(self, x, ilist):
        x_ = as_cuda_ndarray_variable(x)
        ilist_ = gpu_contiguous(T.cast(ilist, dtype=config.floatX)) # T.as_tensor_variable(ilist)
        #if ilist_.type.dtype[:3] not in ('int', 'uin'):
        #    raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')

        # # c code suppose it is int64
        # if x.ndim in [1, 2, 3] and ilist_.dtype in [
        #     'int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']:
        #     ilist_ = tensor.cast(ilist_, 'int64')

        bcast = (ilist_.broadcastable[0],) + x_.broadcastable[1:]
        return theano.gof.Apply(self, [x_, ilist_],
                                [CudaNdarrayType(dtype=x.dtype,
                                                 broadcastable=bcast)()])

    def perform(self, node, inp, out):
        raise NotImplementedError("AdvancedSubtensor1FloatsGPU is GPU only")

    def c_code_cache_version(self):
        return 19

    def c_support_code(self):
        return """
/*
 * Defines `k_take_3` for the case where we have `indices` only as a float32
 * view.
 *
 * d0,... are the output dims
 * indices are a list of index to operate on
 *         They are int32 viewed as float32.
 * a is the output
 * b is the input
 * dB0, the source leading dimensions size
 */
template <int operator_num>
__global__ void k_take_3_float(const int d0, const int d1, const int d2,
                               const float* indices,
                               float* a,
                               const int sA0, const int sA1, const int sA2,
                               const float* b, const int dB0,
                               const int sB0, const int sB1, const int sB2,
                               int* err){
    for (int i0 = blockIdx.x; i0 < d0; i0 += gridDim.x){
        // Only difference from `k_take_3` -- cast from the float32 view
        int idx = (int) indices[i0];
        if (idx<0)
            idx += dB0; // To allow negative indexing.
        if ((idx < 0) || (idx >= dB0)){
            // Any value other the 0 probably work. But to be more safe, I want
            // to change all bits to prevent problem with concurrent write that
            // could cross cache line. But this should not happen with the
            // current code and driver.
            *err = 0xFFFF;
            continue;
        }
        for (int i1 = threadIdx.x; i1 < d1; i1 += blockDim.x){
            for (int i2 = threadIdx.y; i2 < d2; i2 += blockDim.y){
                int a_idx = i0*sA0 + i1*sA1 + i2*sA2;
                int b_idx = idx*sB0 + i1*sB1 + i2*sB2;
                a[a_idx] = b[b_idx];
            }
        }
    }
}


/**
 * Defines `CudaNdarray_TakeFrom` in the case where we have `indices` as a
 * float32 view.
 *
 * This is just a copy-paste of `CudaNdarray_TakeFrom` from Theano commit
 * 894d66655^ , modified to be accessible from a Theano op's C code (rather
 * than as a C->Python binding). See original at
 *
 *     https://github.com/Theano/Theano/blob/894d66655c5b54432bb1d26da910c3cce3f4b830%5E/theano/sandbox/cuda/cuda_ndarray.cu#L742
 */
CudaNdarray *
TakeFrom_Float(CudaNdarray *self, CudaNdarray *indices, long axis,
               CudaNdarray *out, const char *clipmode) {
    int verbose = 0;

    if (verbose) {
        printf("indices used on the gpu\\n");
        PyObject * used_indices = CudaNdarray_CreateArrayObj(indices);
        PyObject_Print(used_indices, stdout, 0);
        Py_DECREF(used_indices);
    }

    if (verbose) printf("after print of object\\n");
    if(!CudaNdarray_is_c_contiguous(indices) != 0) {
        PyErr_SetString(PyExc_NotImplementedError, "CudaNdarray_TakeFrom: The indices must be contiguous in memory.");
        Py_DECREF(indices);
        return NULL;
    }
    int nb_indices = CudaNdarray_SIZE((CudaNdarray *)indices);

    if (axis != 0) {
        PyErr_SetString(PyExc_NotImplementedError,"CudaNdarray_TakeFrom: only axis=0 is currently supported");
        Py_DECREF(indices);
        return NULL;
    }

    //Check argument out
    if (out && (out->nd != self->nd ||
                CudaNdarray_HOST_DIMS(out)[0] != nb_indices))
        out = NULL;
    int dims[self->nd];
    dims[0] = nb_indices;

    for (int i=1 ; i<self->nd ; i++) {
        dims[i] = CudaNdarray_HOST_DIMS(self)[i];
        if (out && CudaNdarray_HOST_DIMS(out)[i] != dims[i]) {
            out = NULL;
        }
    }
    if (!out) {
        out = (CudaNdarray*)CudaNdarray_New();
        if (!out){
            Py_DECREF(indices);
            return NULL;
        }
        if (CudaNdarray_alloc_contiguous(out, self->nd, dims)) {
            Py_DECREF(out);
            Py_DECREF(indices);
            return NULL;
        }
    }else {
        Py_INCREF(out);
    }

    //Check argument clipmode
    if (strcmp(clipmode, "raise") != 0) {
        PyErr_SetString(PyExc_NotImplementedError,"CudaNdarray_TakeFrom: only the raise mode is currently supported");
        Py_DECREF(indices);
        Py_DECREF(out);
        return NULL;
    }

    void (*k3)(const int, const int, const int,
               const float*,
               float*, const int, const int, const int,
               const float*, const int,
               const int, const int, const int,
               int*);
    k3 = k_take_3_float<CPY>;

    // Create the memory place that will store the error information.
    if (init_err_var() != 0) {
        Py_DECREF(indices);
        Py_DECREF(out);
        return NULL;
    }

    dim3 n_blocks(std::min(CudaNdarray_HOST_DIMS(out)[0],65535),1,1);
    switch (self->nd) {
        case 1:
            {
                dim3 n_threads(1, 1, 1);
                if (verbose)
                    printf("kernel config: (n_blocks.x=%d, n_blocks.y=%d,"
                           " n_threads.x=%i, n_threads.y=%i)\\n",
                           n_blocks.x, n_blocks.y, n_threads.x, n_threads.y);
                k3<<<n_blocks, n_threads>>>(
                        dims[0],
                        1,
                        1,
                        CudaNdarray_DEV_DATA(indices),
                        CudaNdarray_DEV_DATA(out),
                        CudaNdarray_HOST_STRIDES(out)[0], //strides
                        1,
                        1,
                        CudaNdarray_DEV_DATA(self),
                        CudaNdarray_HOST_DIMS(self)[0], //For indices check
                        CudaNdarray_HOST_STRIDES(self)[0], //strides
                        1,
                        1,
                        err_var);
            }
            break;
        case 2:
            {
                dim3 n_threads(std::min(CudaNdarray_HOST_DIMS(out)[1], 512), 1, 1);
                if (verbose)
                    printf("kernel config: (n_blocks.x=%d, n_blocks.y=%d,"
                           " n_threads.x=%i, n_threads.y=%i)\\n",
                           n_blocks.x, n_blocks.y, n_threads.x, n_threads.y);
                k3<<<n_blocks, n_threads>>>(
                        dims[0], //dimensions
                        dims[1],
                        1,
                        CudaNdarray_DEV_DATA(indices),
                        CudaNdarray_DEV_DATA(out),
                        CudaNdarray_HOST_STRIDES(out)[0], //strides
                        CudaNdarray_HOST_STRIDES(out)[1],
                        1,
                        CudaNdarray_DEV_DATA(self),
                        CudaNdarray_HOST_DIMS(self)[0], //For indices check
                        CudaNdarray_HOST_STRIDES(self)[0], //strides
                        CudaNdarray_HOST_STRIDES(self)[1],
                        1,
                        err_var);
            }
            break;
        case 3:
            {
                int ty = std::min(CudaNdarray_HOST_DIMS(out)[2], 512);
                int tx = std::min(CudaNdarray_HOST_DIMS(out)[1], 512 / ty);
                dim3 n_threads(tx, ty, 1);
                if (verbose)
                    printf("kernel config: (n_blocks.x=%d, n_blocks.y=%d,"
                           " n_threads.x=%i, n_threads.y=%i)\\n",
                           n_blocks.x, n_blocks.y, n_threads.x, n_threads.y);
                k3<<<n_blocks, n_threads>>>(
                        dims[0], //dimensions
                        dims[1],
                        dims[2],
                        CudaNdarray_DEV_DATA(indices),
                        CudaNdarray_DEV_DATA(out),
                        CudaNdarray_HOST_STRIDES(out)[0], //strides
                        CudaNdarray_HOST_STRIDES(out)[1],
                        CudaNdarray_HOST_STRIDES(out)[2],
                        CudaNdarray_DEV_DATA(self),
                        CudaNdarray_HOST_DIMS(self)[0], //For indices check
                        CudaNdarray_HOST_STRIDES(self)[0], //strides
                        CudaNdarray_HOST_STRIDES(self)[1],
                        CudaNdarray_HOST_STRIDES(self)[2],
                        err_var);
            }
            break;
    default:
        PyErr_SetString(PyExc_NotImplementedError,
                        "CudaNdarray_TakeFrom: only input with 1, 2 or 3"
                        " dimensions are currently supported");

    }
    CNDA_THREAD_SYNC;
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        PyErr_Format(PyExc_RuntimeError,
                     "Cuda error: %s: %s.\\n",
                     "CudaNdarray_TakeFrom",
                     cudaGetErrorString(err));
        Py_DECREF(indices);
        Py_DECREF(out);
        return NULL;
    }

    // Unsafe: don't copy back to CPU for error checking
    /*if (check_err_var() != 0) {
        Py_DECREF(indices);
        Py_DECREF(out);
        return NULL;
    }*/

    if (verbose) printf("TAKE SUCCEDED\\n");
    return out;
}
"""

    def c_code(self, node, name, inp, out_, sub):
        x, idx = inp
        out, = out_
        fail = sub["fail"]

        return """
CudaNdarray *out = TakeFrom_Float((CudaNdarray *)%(x)s, (CudaNdarray *)%(idx)s, 0,
                                  (CudaNdarray *)%(out)s, "raise");
if (out == NULL) {
    %(fail)s;
}
%(out)s = out;

if (cudaGetLastError() != cudaSuccess) {
    PyErr_Format(PyExc_RuntimeError, "Cuda error: k_take_3_float: %%s",
                 cudaGetErrorString(cudaGetLastError()));
    %(fail)s;
}
                                 """ % locals()


@register_opt("fast_compile")
@local_optimizer([gpu_from_host, AdvancedSubtensor1Floats])
def local_gpu_advanced_subtensor1_floats(node):
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and \
           host_input.owner.op.__class__ is AdvancedSubtensor1Floats:
            x = host_input.owner.inputs[0]
            coords = host_input.owner.inputs[1:]
            return [GpuAdvancedSubtensor1Floats(host_input.owner.op._tag)(as_cuda_ndarray_variable(x),
                                                  *coords)]
    if node.op.__class__ is AdvancedSubtensor1Floats:
        x = node.inputs[0]
        coords = node.inputs[1:]
        # print x.owner.op, x.type, node.op._tag # DEV
        if (x.owner and isinstance(x.owner.op, HostFromGpu) and
                x.dtype == "float32"):
            gpu_x, = x.owner.inputs
            return [host_from_gpu(GpuAdvancedSubtensor1Floats(node.op._tag)(gpu_x, *coords))]
    return False



class AdvancedIncSubtensor1Floats(T.subtensor.AdvancedIncSubtensor1):

    def make_node(self, x, y, ilist):
        x_ = T.as_tensor_variable(x)
        y_ = T.as_tensor_variable(y)
        ilist_ = T.as_tensor_variable(ilist)

        #if ilist_.type.dtype[:3] not in ('int', 'uin'):
        #    raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        if y_.type.ndim > x_.type.ndim:
            if self.set_instead_of_inc:
                opname = 'set'
            else:
                opname = 'increment'
            raise TypeError(
                'cannot %s x subtensor with ndim=%s'
                ' by y with ndim=%s' % (
                    opname, x_.type.ndim, y_.type.ndim))

        return theano.gof.Apply(self, [x_, y_, ilist_], [x_.type()])

    def grad(self, inputs, grads):
        g_output, = grads
        x, y, idx_list = inputs
        if x.dtype in theano.tensor.discrete_dtypes:
            # The output dtype is the same as x
            gx = x.zeros_like(dtype=theano.config.floatX)
            if y.dtype in theano.tensor.discrete_dtypes:
                gy = y.zeros_like(dtype=theano.config.floatX)
            else:
                gy = y.zeros_like()
        elif x.dtype in theano.tensor.complex_dtypes:
            raise NotImplementedError("No support for complex grad yet")
        else:
            if self.set_instead_of_inc:
                gx_op = AdvancedIncSubtensor1Floats(set_instead_of_inc=True,
                                                    inplace=self.inplace)
                gx = gx_op(g_output, y.zeros_like(), idx_list)
            else:
                gx = g_output
            gy = AdvancedSubtensor1Floats()(g_output, idx_list)
            gy = T.subtensor._sum_grad_over_bcasted_dims(y, gy)

        return [gx, gy] + [T.DisconnectedType()()]


class GpuAdvancedIncSubtensor1Floats_dev20(AdvancedIncSubtensor1Floats, GpuOp):

    """
    Modified form of `GpuAdvancedIncSubtensor1_dev20` which supports indices in
    float32 view.
    """

    def make_node(self, x, y, ilist):
        x_ = as_cuda_ndarray_variable(x)
        y_ = as_cuda_ndarray_variable(y)
        ilist_ = gpu_contiguous(T.cast(ilist, config.floatX))

        assert x_.type.dtype == y_.type.dtype
        assert x_.type.ndim >= y_.type.ndim

        #if ilist_.type.dtype[:3] not in ('int', 'uin'):
        #    raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        if y_.type.ndim > x_.type.ndim:
            if self.set_instead_of_inc:
                opname = 'set'
            else:
                opname = 'increment'
            raise TypeError(
                'cannot %s x subtensor with ndim=%s'
                ' by y with ndim=%s' % (
                    opname, x_.type.ndim, y_.type.ndim))

        return theano.gof.Apply(self, [x_, y_, ilist_], [x_.type()])

    def perform(self, node, inp, out):
        raise NotImplementedError("GpuAdvancedIncSubtensor1Floats_dev20 supports GPU only")

    def c_code_cache_version(self):
        return 7

    def c_code(self, node, name, inputs, outputs, sub):
        x, y, ind = inputs
        out, = outputs
        fail = sub['fail']
        inplace = int(self.inplace)
        set_instead_of_inc = int(self.set_instead_of_inc)
        return """
        Py_XDECREF(%(out)s);
        if (!%(inplace)s) {
            %(out)s = (CudaNdarray*)CudaNdarray_Copy(%(x)s);
        } else {
            %(out)s = %(x)s;
            Py_XINCREF(%(out)s);
        }
        if (CudaNdarray_Fvector_add_or_replace_fast(%(out)s, %(y)s, %(ind)s, %(set_instead_of_inc)s) != 0){
            %(fail)s
        }
        if (!%(out)s) {
            %(fail)s
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        return """
        __global__ void k_Fvector_add_or_replace_fast(int numRowsX,
                                           int numColsX,
                                           int stridesX0,
                                           int stridesX1,
                                           float *X,
                                           int numRowsY,
                                           int numColsY,
                                           int stridesY0,
                                           int stridesY1,
                                           float *Y ,
                                           float *d_indices_arr,
                                           int num,
                                           const int set_instead_of_inc,
                                           int* err)
        {
             for (int i = (blockIdx.x); i < num; i += gridDim.x)
             {
                  for(int j = (threadIdx.x); j < numColsX;j += blockDim.x)
                  {
                      int x_row = (int) d_indices_arr[i];
                      if(x_row < 0)
                          x_row += numRowsX;
                      int y_row = i;
                      if(x_row < numRowsX && x_row >= 0){
                        if(set_instead_of_inc){
                            // HACK: Unsafe (non-atomic) update.
                            X[(x_row * stridesX0) + (j * stridesX1)] = Y[(y_row * stridesY0) + (j * stridesY1)];
                        } else{
                            atomicAdd(&X[(x_row * stridesX0) + (j * stridesX1)],
                                  Y[(y_row * stridesY0) + (j * stridesY1)]);
                        }
                      } else {
                        *err = 1;
                      }
                  }
             }
             return;
        }
        int CudaNdarray_Fvector_add_or_replace_fast(CudaNdarray* py_self,
            CudaNdarray* py_other, CudaNdarray *py_indices,
            const int set_instead_of_inc)
        {
            if(init_err_var()!= 0) return -1;
            const int *shapeX = CudaNdarray_HOST_DIMS(py_self);
            const int *shapeY = CudaNdarray_HOST_DIMS(py_other);
            const int *strX   = CudaNdarray_HOST_STRIDES(py_self);
            const int *strY   = CudaNdarray_HOST_STRIDES(py_other);
            unsigned int size = (unsigned int)CudaNdarray_SIZE(py_indices);
            if(size == 0){
                return 0;
            }
            unsigned int numcolsX = shapeX[1];
            unsigned int num_threads_per_block = std::min(
                numcolsX, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
            unsigned int num_blocks = std::min(
                size, (unsigned int)NUM_VECTOR_OP_BLOCKS);
            dim3 n_blocks(num_blocks);
            dim3 n_threads(num_threads_per_block);
            cudaError_t err;

            k_Fvector_add_or_replace_fast<<<n_blocks, n_threads>>>(
                shapeX[0],
                shapeX[1],
                strX[0],
                strX[1],
                CudaNdarray_DEV_DATA(py_self),
                shapeY[0],
                shapeY[1],
                strY[0],
                strY[1],
                CudaNdarray_DEV_DATA(py_other),
                CudaNdarray_DEV_DATA(py_indices),
                size,
                set_instead_of_inc,
                err_var
            );

            // Unsafe: This induces a DtoH transfer. Only enable for dev and the like.
            //int index_err = check_err_var();
            //if(index_err != 0) return -1;

            err = cudaGetLastError();
            if(err != cudaSuccess){
                PyErr_Format(
                    PyExc_RuntimeError,
                    "GpuAdvancedIncSubtensor1Floats_dev20: cuda error: %%s",
                    cudaGetErrorString(err));
                return -1;
            }
            return 0;
        }
        """ % locals()



class GpuAdvancedIncSubtensor1Floats_scal_dev20(AdvancedIncSubtensor1Floats, GpuOp):

    """
    Modified form of `GpuAdvancedIncSubtensor1_dev20` which supports indices in
    float32 view and scalar set values.
    """

    def make_node(self, x, y, ilist):
        x_ = as_cuda_ndarray_variable(x)
        y_ = as_cuda_ndarray_variable(y)
        ilist_ = gpu_contiguous(T.cast(ilist, config.floatX))

        assert x_.type.dtype == y_.type.dtype
        assert x_.type.ndim >= y_.type.ndim

        #if ilist_.type.dtype[:3] not in ('int', 'uin'):
        #    raise TypeError('index must be integers')
        if ilist_.type.ndim != 1:
            raise TypeError('index must be vector')
        if x_.type.ndim == 0:
            raise TypeError('cannot index into a scalar')
        if y_.type.ndim > x_.type.ndim:
            if self.set_instead_of_inc:
                opname = 'set'
            else:
                opname = 'increment'
            raise TypeError(
                'cannot %s x subtensor with ndim=%s'
                ' by y with ndim=%s' % (
                    opname, x_.type.ndim, y_.type.ndim))

        return theano.gof.Apply(self, [x_, y_, ilist_], [x_.type()])

    def perform(self, node, inp, out):
        raise NotImplementedError("GpuAdvancedIncSubtensor1Floats_dev20 supports GPU only")

    def c_code_cache_version(self):
        return 6

    def c_code(self, node, name, inp, out, sub):
        x, y, ind = inp
        out, = out
        fail = sub["fail"]

        inplace = int(self.inplace)
        set_instead_of_inc = int(self.set_instead_of_inc)

        return """
        Py_XDECREF(%(out)s);
        if (!%(inplace)s) {
            %(out)s = (CudaNdarray*)CudaNdarray_Copy(%(x)s);
        } else {
            %(out)s = %(x)s;
            Py_XINCREF(%(out)s);
        }
        if (CudaNdarray_broadcast_inc_scalar(%(out)s, %(y)s, %(ind)s, %(set_instead_of_inc)s) != 0){
            %(fail)s
        }
        if (!%(out)s) {
            %(fail)s
        }
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        return """
        __global__ void k_broadcast_inc_scalar(int nX, int strX, float *d_X,
                                               const float *d_Y,
                                               const float *d_indices, int n,
                                               const int set_instead_of_inc, int *err)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n)
                return;
            idx = (int) d_indices[idx];
            if (idx >= nX)
                return;

            if (set_instead_of_inc) {
                d_X[idx * strX] = d_Y[0];
            } else {
                d_X[idx * strX] += d_Y[0];
            }

        }

        int CudaNdarray_broadcast_inc_scalar(CudaNdarray* py_self,
            CudaNdarray* py_other, CudaNdarray* py_indices,
            const int set_instead_of_inc)
        {
            if(init_err_var()!= 0) return -1;

            int size = CudaNdarray_SIZE(py_indices);
            if(size == 0){
                return 0;
            }

            unsigned int n_threads = std::min(size, NUM_VECTOR_OP_THREADS_PER_BLOCK);
            unsigned int n_blocks = std::min(NUM_VECTOR_OP_BLOCKS,
                (size + NUM_VECTOR_OP_THREADS_PER_BLOCK - 1) / NUM_VECTOR_OP_THREADS_PER_BLOCK);

            cudaError_t err;

            k_broadcast_inc_scalar<<<n_blocks, n_threads>>>(
                CudaNdarray_SIZE(py_self),
                CudaNdarray_HOST_STRIDES(py_self)[0],
                CudaNdarray_DEV_DATA(py_self),
                CudaNdarray_DEV_DATA(py_other),
                CudaNdarray_DEV_DATA(py_indices),
                size, set_instead_of_inc, err_var);

            // Unsafe: This induces a DtoH transfer. Only enable for dev and the like.
            /*int index_err = check_err_var();
            if(index_err != 0) return -1;*/

            err = cudaGetLastError();
            if(err != cudaSuccess){
                PyErr_Format(
                    PyExc_RuntimeError,
                    "GpuAdvancedIncSubtensor1Floats_scal_dev20: cuda error: %%s",
                    cudaGetErrorString(err));
                return -1;
            }
            return 0;
        }
        """ % locals()


@register_opt("fast_compile")
@local_optimizer([gpu_from_host, AdvancedIncSubtensor1Floats])
def local_gpu_advanced_incsubtensor1_scal_floats(node):
    supported_dims = {
            # x.ndim, y.ndim
            (1, 0): GpuAdvancedIncSubtensor1Floats_scal_dev20,
            (2, 2): GpuAdvancedIncSubtensor1Floats_dev20,
    }

    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        # Should not execute for GpuAdvancedIncSubtensor1
        if host_input.owner and \
           host_input.owner.op.__class__ is AdvancedIncSubtensor1Floats:
            x, y = host_input.owner.inputs[0:2]
            dims = (x.ndim, y.ndim)
            if dims not in supported_dims.keys():
                return False

            coords = host_input.owner.inputs[2:]
            set_instead_of_inc = host_input.owner.op.set_instead_of_inc
            inplace = host_input.owner.op.inplace

            gpu_op = supported_dims[dims](inplace=inplace,
                set_instead_of_inc=set_instead_of_inc)
            return [gpu_op(as_cuda_ndarray_variable(x),
                           as_cuda_ndarray_variable(y), *coords)]

    # Should not execute for GpuAdvancedIncSubtensor1
    if (node.op.__class__ is AdvancedIncSubtensor1Floats and
            node.inputs[0].dtype == "float32" and
            node.inputs[1].dtype == "float32" and
            node.inputs[2].dtype == "float32"):
        x, y = node.inputs[0:2]
        dims = (x.ndim, y.ndim)
        if dims not in supported_dims:
            return False

        coords = node.inputs[2:]
        go_gpu = False
        if x.owner and isinstance(x.owner.op, HostFromGpu):
            go_gpu = True
            gpu_x, = x.owner.inputs
        else:
            gpu_x = as_cuda_ndarray_variable(x)
        if y.owner and isinstance(y.owner.op, HostFromGpu):
            go_gpu = True
            gpu_y, = y.owner.inputs
        else:
            gpu_y = as_cuda_ndarray_variable(y)
        if go_gpu:
            set_instead_of_inc = node.op.set_instead_of_inc
            inplace = node.op.inplace

            gpu_op = supported_dims[dims](inplace=inplace,
                set_instead_of_inc=set_instead_of_inc)
            return [host_from_gpu(gpu_op(gpu_x, gpu_y, *coords))]
    return False


class JoinUnsafe(T.Join):
    pass


class GpuJoinUnsafe(GpuJoin):
    """Implements GPU-based join without error checking."""

    def c_code(self, node, name, inputs, out_, sub):
        nd = node.inputs[1].ndim
        if not all(i.ndim == nd for i in node.inputs[2:]):
            # all inputs ndarray need to have the same number of dimensions
            raise NotImplementedError()
        axis = inputs[0]
        n_cndas = len(inputs[1:])
        input_1 = inputs[1]
        fail = sub['fail']
        out = out_[0]

        # getting the shapes of all the involved tensors (input[0]+out)
        str = """
        int axis = PyInt_AsLong((PyObject*)%(axis)s);
        const int nd = %(nd)s;
        int shape_out[nd];
        int width_sum = 0;
        int errorcode;
        int sum = 0;
        PyObject *slice_tuple = NULL;
        PyObject *section_slice = NULL;
        PyObject *full_slice = NULL;
        PyObject *out_sub = NULL;
        PyObject *start, *stop;
        start = NULL;
        stop = NULL;
        """ % locals()

        # Test negative axis
        str += """
        if( axis < -nd ){
            PyErr_Format(PyExc_IndexError,
                         "Join axis %%d out of bounds [0, %%d)", axis, nd);
            %(fail)s
        }
        if( axis < 0 ){
            axis = axis + nd;
        }
        """ % locals()

        # getting the shapes of all the involved tensors (input[1:])
        # + check: all input tensors have same shape as final out
        # except for "axis" dimension
        # shape_%(cdna)s[nd] is initialized before, to prevent following
        # error: jump to label __label_9 crosses initialization of
        # shape_%(cdna)s[nd]
        for i, cdna in enumerate(theano.gof.utils.uniq(inputs[1:])):
            str += """
            int shape_%(cdna)s[nd];
            """ % locals()
        str += """
        if(-1 == axis && PyErr_Occurred()){
            %(fail)s;
        }
        full_slice = PySlice_New(NULL, NULL, NULL);
        if(full_slice == NULL){
            %(fail)s;
        }
        for(int i = 0; i<nd; i+=1)
        {
            shape_%(input_1)s[i] = CudaNdarray_HOST_DIMS(%(input_1)s)[i];
            shape_out[i] = shape_%(input_1)s[i];
        }
        """ % locals()
        for i, cdna in enumerate(theano.gof.utils.uniq(inputs[2:])):
            str += """
            for(int i = 0; i<nd; i+=1)
            {
                shape_%(cdna)s[i] = CudaNdarray_HOST_DIMS(%(cdna)s)[i];
                if((i!=axis) && (shape_%(cdna)s[i]!=shape_out[i]))
                {
                    PyErr_Format(
                        PyExc_ValueError,
                        "GpuJoin: Wrong inputs for input %%d related"
                        " to inputs 0.!",
                        i);
                    %(fail)s;
                }
            }
            """ % locals()

        # computing the new shape for the out tensors
        for i, cdna in enumerate(inputs[1:]):
            str += "\t\twidth_sum += CudaNdarray_HOST_DIMS(%(cdna)s)[axis];\n" % locals()
        str += "\t\tshape_out[axis] = width_sum;\n"

        # preparing the output array + init of the necessary variables
        # for the data transfer
        str += """
        if (CudaNdarray_prep_output(&%(out)s, nd, shape_out))
        {
            %(fail)s;
        }
        """ % locals()
        # start copying the data into the new out tensors
        for i, cdna in enumerate(inputs[1:]):
            str += """
            sum += shape_%(cdna)s[axis];
            stop = PyInt_FromLong(sum);
            slice_tuple = PyTuple_New(nd);
            if(slice_tuple == NULL){
                %(fail)s;
            }
            section_slice = PySlice_New(start, stop, NULL);
            if(section_slice == NULL){
                %(fail)s;
            }
            for(int i=0; i<nd; i++)
            {
                if(i!=axis)
                {
                    Py_INCREF(full_slice);
                    PyTuple_SetItem(slice_tuple, i, full_slice);
                }
                else
                {
                    Py_INCREF(section_slice);
                    PyTuple_SetItem(slice_tuple, i, section_slice);
                }
            }
            out_sub = CudaNdarray_Subscript((PyObject*)%(out)s, slice_tuple);
            if(out_sub == NULL){
                Py_XDECREF(start);
                Py_XDECREF(stop);
                Py_XDECREF(slice_tuple);
                Py_XDECREF(out_sub);
                Py_XDECREF(%(out)s);
                %(fail)s;
            }
            Py_CLEAR(slice_tuple);
            Py_CLEAR(section_slice);

            // Unsafe: skip error checking.
            /*errorcode = CudaNdarray_CopyFromCudaNdarray(
                (CudaNdarray*)out_sub, %(cdna)s);
            if(errorcode != 0)
            {
                Py_XDECREF(start);
                Py_XDECREF(stop);
                Py_XDECREF(out_sub);
                Py_XDECREF(%(out)s);
                %(fail)s;
            }*/

            Py_XDECREF(out_sub);
            Py_XDECREF(start);
            start = stop;
            stop = NULL;
            """ % locals()

        str += """
            Py_XDECREF(start);
            Py_XDECREF(stop);
        """
        return str

    def c_code_cache_version(self):
        return (1,)


@register_opt("fast_compile")
@local_optimizer([JoinUnsafe])
def local_gpu_join_unsafe(node):
    """
    Inspired by the opt for convop.
    Very loose notation follows.
    Subgraphs concerned first look like
        [array of HostTensor] -> HostToGpu -> GpuToHost
        -> Join -> HostToGpu -> GpuToHost
    First we apply this Opt:
    join(host_from_gpu) -> host_from_gpu(gpu_join)
    then, as an intermediate result, there should be
    host_from_gpu(gpu_join) -> HostToGpu -> GpuToHost
    this unnecessary GpuToHost -> HostToGpu should be removed
    by other opts, leaving us with
    host_from_gpu(gpu_join)
    For intermediate places in the graph not covered by the first opt, the
    following could be useful:
    gpu_from_host(join) -> gpu_join(gpu_from_host)
    not implemented yet.
    """
    if isinstance(node.op, JoinUnsafe):
        # optimizing this case:
        # join(host_from_gpu) -> host_from_gpu(gpu_join)

        axis_and_tensors = node.inputs

        matches = [t.dtype == 'float32' and
                   ((t.owner is not None and
                     isinstance(t.owner.op, HostFromGpu)) or
                    isinstance(t, theano.gof.Constant)) for t in axis_and_tensors[1:]]

        if all(matches):
            new_tensors = [as_cuda_ndarray_variable(t)
                           for t in axis_and_tensors[1:]]
            new_a_and_t = [axis_and_tensors[0]] + new_tensors

            replacement_node = host_from_gpu(GpuJoinUnsafe()(*new_a_and_t))

            return [replacement_node]
