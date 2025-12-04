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

  // void copy_from_numpy(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
  //   if (arr.ndim() != 2) throw std::runtime_error("Expected 2D NumPy array");
  //   auto h = static_cast<int>(arr.shape(0));
  //   auto w = static_cast<int>(arr.shape(1));
  //   if (h != t.h || w != t.w) throw std::runtime_error("Shape mismatch");

  //   // allocate a host tensor
  //   Tensor<T> host_t(h, w, /*on_device=*/false);
    
  //   // memcpy from NumPy to host tensor's raw pointer
  //   std::memcpy(host_t.rawp, arr.data(), sizeof(T) * h * w);

  //   // Move host -> device or host based on whether it's cuda tensor or not
  //   if (t.on_device) {
  //     host_t.toDevice(t);
  //   } else {
  //     std::memcpy(t.rawp, host_t.rawp, sizeof(T) * h * w);
  //   }
  // }

  // // Copy to NumPy (this→host→NumPy)
  // py::array to_numpy() const {
  //   Tensor<T> host_t(t.h, t.w, /*on_device=*/false);
  //   if (t.on_device) {
  //     t.toHost(host_t);
  //   } else {
  //     std::memcpy(host_t.rawp, t.rawp, sizeof(T) * t.h * t.w);
  //   }

  //   // Create NumPy array that owns its own memory (copy)
  //   py::array_t<T> out({t.h, t.w});
  //   std::memcpy(out.mutable_data(), host_t.rawp, sizeof(T) * t.h * t.w);
  //   return out;
  // }
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
    // Note: We use the Flattened Height (t.h) and Width (t.w) for the buffer
    Tensor<T> host_t(t.h, t.w, /*on_device=*/false);
    
    // 3. Memcpy
    // Since both are row-major contiguous, we can copy bytes directly 
    // regardless of the shape (4D vs 2D).
    std::memcpy(host_t.rawp, arr.data(), sizeof(T) * my_size);

    // 4. Send to Device
    if (t.on_device) {
      host_t.toDevice(t);
    } else {
      std::memcpy(t.rawp, host_t.rawp, sizeof(T) * my_size);
    }
  }

  // --- UPDATED: TO NUMPY ---
  // Returns 4D array if internal metadata exists, otherwise 2D
  py::array to_numpy() const {
    // 1. Pull data to Host
    Tensor<T> host_t(t.h, t.w, /*on_device=*/false);
    if (t.on_device) {
      t.toHost(host_t);
    } else {
      std::memcpy(host_t.rawp, t.rawp, sizeof(T) * t.h * t.w);
    }

    // 2. Determine Output Shape
    // If we have 4D metadata, we return a 4D numpy array
    if (t.b > 1 || t.d > 1) {
        py::array_t<T> out({t.b, t.d, t.true_h, t.true_w});
        std::memcpy(out.mutable_data(), host_t.rawp, sizeof(T) * t.h * t.w);
        return out;
    } else {
        // Otherwise return standard 2D view
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

            // COPY SCALAR TO HOST
            
            T val;
            
            // Calculate pointer location using your new strides
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
    
    // ... ( existing Helper Lambda) ...
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
        if (me.t.b > 1 || me.t.d > 1) {
            // If internal 4D metadata exists, return the 4D tuple
            return py::make_tuple(me.t.b, me.t.d, me.t.true_h, me.t.true_w);
        } else {
            // Otherwise, return the standard 2D tuple (Flattened view)
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
    .def("__add__", [](const Self &me, const Self &other) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      op_add<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("__add__", [](const Self &me, T scalar) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
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
      op_multiply<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("__mul__", [](const Self &me, T scalar) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      op_multiply<T>(me.t, scalar, out.t);
      return out;
    }, py::arg("scalar"))
    .def("__eq__", [](const Self &me, const Self &other) {
      PyTensor<uint32_t> out(me.t.h, me.t.w, me.t.on_device);
      op_equal<T>(me.t, other.t, out.t);
      return out;
    }, py::arg("other"))
    .def("relu", [](const Self &me) {
      PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
      op_relu<T>(me.t, out.t);
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
        PyTensor<T> out(me.t.h, me.t.w, me.t.on_device);
        op_softmax<T>(me.t, out.t);
        return out;
    }, "Computes row-wise softmax.")

    // Batched Matrix Multiplication
    // Usage: A.bmm(B, batch_count, m, n, k)
    // We assume the user has flattened the tensors to 2D before calling this
    // --- OLD BINDING (Delete this) ---
    // .def("bmm", [](const Self &me, const Self &other, int batch_count, int m, int n, int k) ...

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
    }, py::arg("other"), "Batched Matrix Multiplication (Auto-inferred shapes)");

}