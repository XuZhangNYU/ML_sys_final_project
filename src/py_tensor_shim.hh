#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <type_traits>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "utils/tensor.cuh"
#include "ops/op_elemwise.cuh"
#include "ops/op_mm.cuh"
#include "ops/op_reduction.cuh"
#include "ops/op_cross_entropy.cuh"

// --- NEW: Include your new kernels here ---

// ------------------------------------------

namespace py = pybind11;

// map C++ type name to a short string for Python repr, only float32 or uint32 for now
template <typename T> inline const char* dtype_name();
template <> inline const char* dtype_name<float>()  { return "float32"; }
template <> inline const char* dtype_name<uint32_t>()   { return "uint32"; }

// Wrapper from Tensor<T> to a Python object
template <typename T>
class PyTensor {
public:
  Tensor<T> t;


  // Existing 2D Constructor
  PyTensor(int h, int w, bool is_cuda) : t(h, w, is_cuda) {}  

  // NEW 4D Constructor Wrapper
  PyTensor(int b, int d, int h, int w, bool is_cuda) 
  : t(b, d, h, w, is_cuda) {}

  explicit PyTensor(const Tensor<T>& existing) : t(existing) {}          // copy view
  explicit PyTensor(Tensor<T>&& existing) : t(std::move(existing)) {}     // move view


  std::pair<int,int> shape() const { return {t.h, t.w}; }

  bool is_cuda() const { return t.on_device; }

  // --- UPDATED: COPY FROM NUMPY ---
  // Accepts any shape (2D, 3D, 4D) as long as total elements match
  void copy_from_numpy(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
    // 1. Check Total Size instead of Dimensions
    ssize_t input_size = arr.size(); // Total elements in numpy array
    ssize_t my_size = (ssize_t)t.h * (ssize_t)t.w; // Total elements in tensor

    if (input_size != my_size) {
        throw std::runtime_error("Size mismatch. Tensor has " + 
            std::to_string(my_size) + " elements, but Numpy array has " + 
            std::to_string(input_size));
    }

    // 2. Create a temporary host tensor (treated as flat 2D for copying)
    Tensor<T> host_t(t.h, t.w, /*on_device=*/false);
    
    // 3. Memcpy
    // Since both are row-major contiguous, we can copy bytes directly regardless of the shape (4D vs 2D).
    std::memcpy(host_t.rawp, arr.data(), sizeof(T) * my_size);
    if (t.on_device) {
      host_t.toDevice(t);
    } else {
      std::memcpy(t.rawp, host_t.rawp, sizeof(T) * my_size);
    }
  }

  // --- UPDATED: TO NUMPY ---
  // Returns 4D array if internal metadata exists, otherwise 2D
  py::array to_numpy() const {
      // ... (Existing copy to host logic) ...
      Tensor<T> host_t(t.h, t.w, /*on_device=*/false);
      if (t.on_device) { t.toHost(host_t); } 
      else { std::memcpy(host_t.rawp, t.rawp, sizeof(T) * t.h * t.w); }

      // Logic Switch
      if (t.is_3d) {
          // CASE 3D: Return [Batch, Seq, Dim]
          // Internal structure is [B, 1, S, D]
          py::array_t<T> out({t.b, t.true_h, t.true_w}); // Skip 'd'
          std::memcpy(out.mutable_data(), host_t.rawp, sizeof(T) * t.h * t.w);
          return out;
      } 
      else if (t.b > 1 || t.d > 1) {
          // CASE 4D: Return [B, D, S, D]
          py::array_t<T> out({t.b, t.d, t.true_h, t.true_w});
          std::memcpy(out.mutable_data(), host_t.rawp, sizeof(T) * t.h * t.w);
          return out;
      } else {
          // CASE 2D
          py::array_t<T> out({t.h, t.w});
          std::memcpy(out.mutable_data(), host_t.rawp, sizeof(T) * t.h * t.w);
          return out;
      }
    }

  void fill(T v) {
    op_const_fill(t, v);
  }

  PyTensor<T> transpose() const {
    return PyTensor<T>(t.transpose());
  }

  PyTensor<T> slice_view(int start_h, int end_h, int start_w, int end_w) const {
    return PyTensor<T>(t.slice(start_h, end_h, start_w, end_w));
  }
 
  // Interpret Python indexing syntax and return a sliced tensor view.
  // PyTensor<T> getitem(py::object index) const {
  //   auto parse_dim = [&](py::handle obj, int dim_size) -> std::pair<int, int> {
  //     if (obj.is(py::ellipsis()) || obj.is_none()) {
  //       return {0, dim_size};
  //     }

  //     if (py::isinstance<py::slice>(obj)) {
  //       Py_ssize_t start = 0;
  //       Py_ssize_t stop = 0;
  //       Py_ssize_t step = 0;
  //       Py_ssize_t slicelength = 0;
  //       auto slice = obj.cast<py::slice>();
  //       if (!slice.compute(dim_size, &start, &stop, &step, &slicelength)) {
  //         throw py::error_already_set();
  //       }
  //       if (step != 1) {
  //         throw py::index_error("Tensor slicing only supports step=1");
  //       }
  //       return {static_cast<int>(start), static_cast<int>(stop)};
  //     }

  //     if (py::isinstance<py::int_>(obj)) {
  //       int idx = obj.cast<int>();
  //       if (idx < 0) {
  //         idx += dim_size;
  //       }
  //       if (idx < 0 || idx >= dim_size) {
  //         throw py::index_error("Index out of range");
  //       }
  //       return {idx, idx + 1};
  //     }

  //     throw py::type_error("Tensor indices must be integers, slices, or None");
  //   };

  //   py::tuple idx_tuple;
  //   if (py::isinstance<py::tuple>(index)) {
  //     idx_tuple = index.cast<py::tuple>();
  //   } else {
  //     idx_tuple = py::make_tuple(index);
  //   }

  //   if (idx_tuple.size() > 2) {
  //     throw py::index_error("Tensor only supports 2D indexing");
  //   }

  //   py::object first_index = idx_tuple.size() > 0
  //     ? py::reinterpret_borrow<py::object>(idx_tuple[0])
  //     : py::ellipsis();
  //   py::object second_index = idx_tuple.size() > 1
  //     ? py::reinterpret_borrow<py::object>(idx_tuple[1])
  //     : py::ellipsis();

  //   auto [start_h, end_h] = parse_dim(first_index, t.h);
  //   auto [start_w, end_w] = parse_dim(second_index, t.w);

  //   return slice_view(start_h, end_h, start_w, end_w);
  // }
// Change return type to py::object to support returning Scalars OR Tensors
  py::object getitem(py::object index) const {
    
    // 1. Unpack Index into a Tuple
    py::tuple idx_tuple;
    if (py::isinstance<py::tuple>(index)) {
      idx_tuple = index.cast<py::tuple>();
    } else {
      idx_tuple = py::make_tuple(index);
    }

    // 4D SCAsdlar ACcesS [b, h, s, d]
    if (idx_tuple.size() == 4) {
        // Ensure the tensor thinks it is 4D
        if (t.b == 1 && t.d == 1 && idx_tuple.size() > 2) {

        }

        try {
            int b_idx = idx_tuple[0].cast<int>();
            int d_idx = idx_tuple[1].cast<int>();
            int h_idx = idx_tuple[2].cast<int>();
            int w_idx = idx_tuple[3].cast<int>();
            // Handle negative indices
            if (b_idx < 0) b_idx += t.b;
            if (d_idx < 0) d_idx += t.d;
            if (h_idx < 0) h_idx += t.true_h;
            if (w_idx < 0) w_idx += t.true_w;




            // Bounds Check
            if (b_idx < 0 || b_idx >= t.b || 
                d_idx < 0 || d_idx >= t.d || 
                h_idx < 0 || h_idx >= t.true_h || 
                w_idx < 0 || w_idx >= t.true_w) {
                throw py::index_error("Index out of bounds");
            }
            
            T val;
            
            T* ptr = t.rawp + t.offset + 
                     (b_idx * t.stride_b) + 
                     (d_idx * t.stride_d) + 
                     (h_idx * t.stride_h) + 
                     (w_idx * t.stride_w);

            if (t.on_device) {
                cudaMemcpy(&val, ptr, sizeof(T), cudaMemcpyDeviceToHost);
            } else {
                val = *ptr;
            }

            // Return as standard Python object (float or int)
            return py::cast(val);

        } catch (py::cast_error&) {
             throw py::type_error("4D Slicing not supported yet. Only 4D Scalar access (integers).");
        }
    }

    // legacy 2d
    
    auto parse_dim = [&](py::handle obj, int dim_size) -> std::pair<int, int> {
      if (obj.is(py::ellipsis()) || obj.is_none()) return {0, dim_size};
      if (py::isinstance<py::slice>(obj)) {
        Py_ssize_t start, stop, step, slicelength;
        if (!obj.cast<py::slice>().compute(dim_size, &start, &stop, &step, &slicelength)) throw py::error_already_set();
        if (step != 1) throw py::index_error("Tensor slicing only supports step=1");
        return {static_cast<int>(start), static_cast<int>(stop)};
      }
      if (py::isinstance<py::int_>(obj)) {
        int idx = obj.cast<int>();
        if (idx < 0) idx += dim_size;
        if (idx < 0 || idx >= dim_size) throw py::index_error("Index out of range");
        return {idx, idx + 1};
      }
      throw py::type_error("Indices must be integers, slices, or None");
    };

    if (idx_tuple.size() > 2) {
      throw py::index_error("Invalid index dimensions. Use 2 (Scroll View) or 4 (Scalar 4D).");
    }

    py::object first_index = idx_tuple.size() > 0 ? py::reinterpret_borrow<py::object>(idx_tuple[0]) : py::ellipsis();
    py::object second_index = idx_tuple.size() > 1 ? py::reinterpret_borrow<py::object>(idx_tuple[1]) : py::ellipsis();

    auto [start_h, end_h] = parse_dim(first_index, t.h);
    auto [start_w, end_w] = parse_dim(second_index, t.w);

    // Return a PyTensor Object (View)
    return py::cast(slice_view(start_h, end_h, start_w, end_w));
  }

  

  std::string repr() const {
    return "dtype=" + std::string(dtype_name<T>()) + " " + t.repr();
  }

};

template <typename T>
void bind_tensor_type(py::module_ &m, const char* pyname) {
  using Self = PyTensor<T>;
  py::class_<Self>(m, pyname)
    //!!! Existing 2D init
    .def(py::init<int,int,bool>(), py::arg("h"), py::arg("w"), py::arg("is_cuda")=true,
         "Create a tensor with shape (h, w) on CPU or CUDA.")
    // Add NEW 4D init !!!!!!!!!!!!!
    .def(py::init<int,int,int,int,bool>(), 
         py::arg("b"), py::arg("d"), py::arg("h"), py::arg("w"), py::arg("is_cuda")=true,
         "Create a 4D Tensor [Batch, Head, Seq, Dim]")
    .def_property_readonly("shape", [](const Self &me) {
      if (me.t.is_3d) {
          return py::make_tuple(me.t.b, me.t.true_h, me.t.true_w);
      } else if (me.t.b > 1 || me.t.d > 1) {
          return py::make_tuple(me.t.b, me.t.d, me.t.true_h, me.t.true_w);
      } else {
          return py::make_tuple(me.t.h, me.t.w);
      }
    })
    .def_property_readonly("is_cuda", &Self::is_cuda)
    .def("to_numpy", &Self::to_numpy, "Copy the tensor to a NumPy array (host).")
    .def("copy_from_numpy", &Self::copy_from_numpy,
         "Copy data from a NumPy array (must be same shape).")
    .def("fill", &Self::fill, py::arg("value"))
    .def("transpose", &Self::transpose, "Return a transposed view (no copy).")
    .def("slice", &Self::slice_view, py::arg("start_h"), py::arg("end_h"),
         py::arg("start_w"), py::arg("end_w"), "Explicit slice view of the tensor.")
    .def_property_readonly("T", &Self::transpose)
    .def("__getitem__", &Self::getitem)
    .def("__repr__", &Self::repr)
    // Handled 4 d case by out.t.true_h = me.t.true_h; out.t.true_w = me.t.true_w;
    .def("__add__", [](const Self &me, const Self &other) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      // Copy metadata from 'me' (assuming shapes match)
      out.t.b = me.t.b; out.t.d = me.t.d;
      out.t.true_h = me.t.true_h; out.t.true_w = me.t.true_w;
      out.t.stride_b = me.t.stride_b; out.t.stride_d = me.t.stride_d;
      
      op_add<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("__add__", [](const Self &me, T scalar) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      out.t.b = me.t.b; out.t.d = me.t.d;
      out.t.true_h = me.t.true_h; out.t.true_w = me.t.true_w;
      out.t.stride_b = me.t.stride_b; out.t.stride_d = me.t.stride_d;
      op_add<T>(me.t, scalar, out.t);
      return out;
    }, py::arg("scalar"))
    .def("__sub__", [](const Self &me, const Self &other) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      op_sub<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("__mul__", [](const Self &me, const Self &other) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      out.t.b = me.t.b; out.t.d = me.t.d;
      out.t.true_h = me.t.true_h; out.t.true_w = me.t.true_w;
      out.t.stride_b = me.t.stride_b; out.t.stride_d = me.t.stride_d;
      op_multiply<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("__mul__", [](const Self &me, T scalar) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      out.t.b = me.t.b; out.t.d = me.t.d;
      out.t.true_h = me.t.true_h; out.t.true_w = me.t.true_w;
      out.t.stride_b = me.t.stride_b; out.t.stride_d = me.t.stride_d;
      op_multiply<T>(me.t, scalar, out.t);
      return out;
    }, py::arg("scalar"))
    .def("__eq__", [](const Self &me, const Self &other) {
      PyTensor<uint32_t> out(me.t.h, me.t.w, me.t.on_device);
      out.t.b = me.t.b; out.t.d = me.t.d;
      out.t.true_h = me.t.true_h; out.t.true_w = me.t.true_w;
      out.t.stride_b = me.t.stride_b; out.t.stride_d = me.t.stride_d;
      op_equal<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("relu", [](const Self &me) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      op_relu<T>(me.t, out.t);
      return out;
    })
    .def("gelu", [](const Self &me) {
        PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
        // Copy metadata for 4D support
        out.t.b = me.t.b; out.t.d = me.t.d; 
        out.t.true_h = me.t.true_h; out.t.true_w = me.t.true_w;
        
        op_gelu<T>(me.t, out.t);
        return out;
    })
    .def("relu_back", [](const Self &me, const Self &dout) {
      PyTensor<T> din(me.t.h, me.t.w, me.t.on_device);
      op_relu_back<T>(me.t, dout.t, din.t);
      return din;
    }, py::arg("DOut"))
    .def("sum", [](const Self &me, int axis=0) {
      int out_h = me.t.h;
      int out_w = me.t.w;
      if (axis == 0) {
        out_h = 1;
      } else if (axis == 1) {
        out_w = 1;
      } else {
        throw std::runtime_error("Invalid axis, must be 0 or 1");
      }
      PyTensor<T> out(out_h, out_w, me.t.on_device);
      op_sum<T>(me.t, out.t);
      return out;
    }, py::arg("axis"))
    .def("argmax", [](const Self &me) {
      PyTensor<uint32_t> out(me.t.h, 1, me.t.on_device);
      op_argmax<T>(me.t, out.t);
      return out;
    })
    .def("__matmul__", [](const Self &me, const Self &other) {
      PyTensor<T> out(me.t.h, other.t.w, me.t.on_device);
      op_mm<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("cross_entropy_loss", [](const Self &logits, const PyTensor<uint32_t> &labels, PyTensor<T> &d_logits) {
      return op_cross_entropy_loss<T,uint32_t>(logits.t, labels.t, d_logits.t);
    }, py::arg("labels"), py::arg("d_logits"), "Compute cross-entropy loss and its gradient")
    
    // --- NEW BINDINGS FOR ATTENTION ---
    
    // Softmax (Row-wise only for now)
    .def("softmax", [](const Self &me) {
        // 1. Create Output (Allocates flattened2D memory)
        PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
        
        // 2. aaa !!! important: :Copy 4D Metadata from Input
        out.t.b = me.t.b;
        out.t.d = me.t.d;
        out.t.true_h = me.t.true_h;
        out.t.true_w = me.t.true_w;
        
        out.t.stride_b = me.t.stride_b;
        out.t.stride_d = me.t.stride_d;

        op_softmax<T>(me.t, out.t);
        return out;
    }, "Computes row-wise softmax (Preserves 4D Shape).")
    .def("view", [](const Self &me, int b, int d, int h, int w) {
        // Validation: Ensure total elements match
        size_t new_size = (size_t)b * d * h * w;
        size_t old_size = (size_t)me.t.b * me.t.d * me.t.true_h * me.t.true_w;
        // Adjust old size calculation if it was 2D
        if (me.t.b == 1 && me.t.d == 1) old_size = (size_t)me.t.h * me.t.w;

        if (new_size != old_size) {
             throw std::runtime_error("View size mismatch.");
        }

        PyTensor<T> out(b, d, h, w, me.t.on_device);
        if (me.t.on_device) {
             cudaMemcpy(out.t.rawp, me.t.rawp, new_size * sizeof(T), cudaMemcpyDeviceToDevice);
        } else {
             std::memcpy(out.t.rawp, me.t.rawp, new_size * sizeof(T));
        }
        return out;
    }, "Reshape/View tensor (Returns a Copy with new shape)")
    .def("view_3d", [](const Self &me, int b, int s, int d) {
      // Validate size
      size_t new_size = (size_t)b * s * d;
      size_t old_size = (size_t)me.t.b * me.t.d * me.t.true_h * me.t.true_w;
      if (new_size != old_size) throw std::runtime_error("View size mismatch");

      // Create a 4D tensor with Heads=1
      PyTensor<T> out(b, 1, s, d, me.t.on_device);
      
      // Mark as 3D
      out.t.is_3d = true;

      // Copy data
      if (me.t.on_device) {
           cudaMemcpy(out.t.rawp, me.t.rawp, new_size * sizeof(T), cudaMemcpyDeviceToDevice);
      } else {
           std::memcpy(out.t.rawp, me.t.rawp, new_size * sizeof(T));
      }
      return out;
      }, py::arg("b"), py::arg("s"), py::arg("d"), "View as 3D tensor [Batch, Seq, Dim]")
    // Batched Matrix Multiplication
    // Usage: A.bmm(B, batch_count, m, n, k)
    // We assume the user has flattened the tensors to 2D before calling this

    // --- NEW BINDING (Use this) ---
    .def("bmm", [](const Self &me, const Self &other) {
        // 1. Auto-Calculate Output Shape
        // A: [B, D, M, K]
        // B: [B, D, K, N] 
        // Out: [B, D, M, N]
        
        int out_b = me.t.b;
        int out_d = me.t.d;
        int out_h = me.t.true_h; // M
        int out_w = other.t.true_w; // N
        
        // Create Output Tensor (Allocates GPU memory)
        PyTensor<T> out(out_b, out_d, out_h, out_w, me.t.on_device);
        
        // Call C++ Op (No extra args needed!)
        op_bmm<T>(me.t, other.t, out.t);
        
        return out;
    }, py::arg("other"), "Batched Matrix Multiplication (Auto-inferred shapes)")
    .def("permute_0213", [](const Self &me) {
        // Input: [B, Seq, Heads, Dim] (Logically)
        // Output: [B, Heads, Seq, Dim]
        
        PyTensor<T> out(me.t.b, me.t.true_h, me.t.d, me.t.true_w, me.t.on_device);
        // Note the swap: out.d = me.h, out.h = me.d
        
        op_permute_0213<T>(me.t, out.t);
        return out;
    })
    .def("causal_mask", [](const Self &me, float scale) {
        // In-place modification is fine for masking usually, but let's copy to be safe/functional
        PyTensor<T> out(me.t.b, me.t.d, me.t.true_h, me.t.true_w, me.t.on_device);
        cudaMemcpy(out.t.rawp, me.t.rawp, sizeof(T)*me.t.h*me.t.w, cudaMemcpyDeviceToDevice);
        
        op_causal_mask<T>(out.t, scale);
        return out;
    }, py::arg("scale"))


    // 1. Split QKV Binding
    .def("split_qkv", [](const Self &me, int n_head, int head_dim) {
        // Input: [Batch*Seq, 3*Hidden]
        int rows = me.t.h;
        int hidden_dim = n_head * head_dim;
        
        // Create 3 output tensors: [Batch*Seq, Hidden]
        PyTensor<T> q(rows, hidden_dim, me.t.on_device);
        PyTensor<T> k(rows, hidden_dim, me.t.on_device);
        PyTensor<T> v(rows, hidden_dim, me.t.on_device);
        
        op_split_qkv(me.t, q.t, k.t, v.t);
        
        return py::make_tuple(q, k, v);
    }, py::arg("n_head"), py::arg("head_dim"))

    // 2. Permute 0231 Binding (For K)
    .def("permute_0231", [](const Self &me) {
        // Input Metadata: b=B, d=Seq, true_h=Heads, true_w=Dim
        
        // Output Metadata: [B, Heads, Dim, Seq]
        int out_b = me.t.b;
        int out_d = me.t.true_h; // Heads
        int out_h = me.t.true_w; // Dim
        int out_w = me.t.d;      // Seq
        
        PyTensor<T> out(out_b, out_d, out_h, out_w, me.t.on_device);
        op_permute_0231<T>(me.t, out.t);
        return out;
    })

    .def("layernorm", [](const Self &me,
                         const PyTensor<T> &gamma,
                         const PyTensor<T> &beta,
                         float eps) {
        // 1. Allocate flat buffer
        PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);

        // 2. Copy 4D metadata
        out.t.b        = me.t.b;
        out.t.d        = me.t.d;
        out.t.true_h   = me.t.true_h;
        out.t.true_w   = me.t.true_w;
        out.t.stride_b = me.t.stride_b;
        out.t.stride_d = me.t.stride_d;

        // 3. Call CUDA kernel wrapper
        op_layernorm<T>(me.t, gamma.t, beta.t, out.t, eps);

        return out;
    }, py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f);
  
}