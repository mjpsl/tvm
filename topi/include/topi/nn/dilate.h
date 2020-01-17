/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Dilate op constructions
 * \file nn/dilate.h
 */
#ifndef TOPI_NN_DILATE_H_
#define TOPI_NN_DILATE_H_

#include <string>

#include "tvm/operation.h"
#include "tvm/ir_pass.h"
#include "topi/tags.h"

namespace topi {
namespace nn {
using namespace tvm;

/*!
* \brief Create a new expression of the logical and of all
* conditions in the arguments.
*
* \param args The arguments to find the logical conjunction of
*
* \return The logical conjunction expression
*/
Expr all(Array<Expr> args) {
  CHECK_GT(args.size(), 0) << "all requires at least one argument";

  Expr ret = args[0];
  for (size_t i = 1; i < args.size(); ++i) {
    ret = ret && args[i];
  }
  return ret;
}

/*!
* \brief Dilate data with zeros
*
* \param x The input tensor, this can have any number of
* dimensions and any layout.
* \param strides Dilation stride for each dimension. Stride 1
* means no dilation.
* \param name The name of the operation
* \param tag The tag to mark the operation
*
* \return The output tensor.
*/
inline Tensor dilate(const Tensor& x,
                     Array<Expr> strides,
                     std::string name = "tensor",
                     std::string tag = kInjective) {
  auto n = x->shape.size();
  CHECK_EQ(n, strides.size())
    << "strides size (" << strides.size()
    << ") must match dimension of x (" << n << ")";

  Array<Expr> out_shape;
  for (size_t i = 0; i < n; ++i) {
    out_shape.push_back(tvm::ir::Simplify(
      (x->shape[i] - 1) * cast(DataType::Int(32), strides[i] + 1)));
  }

  return tvm::compute(
    out_shape,
    [&](const Array<Var>& indices) {
      Array<Expr> not_zero;
      Array<Expr> index_tuple;
      for (size_t i = 0; i < n; ++i) {
        if (IsConstInt(strides[i]) && GetConstInt(strides[i]) == 1) {
          index_tuple.push_back(indices[i]);
        } else {
          index_tuple.push_back(indexdiv(indices[i], strides[i]));
          not_zero.push_back((indexmod(indices[i], strides[i])) == 0);
        }
      }
      if (not_zero.size() > 0) {
        auto all_not_zero = all(not_zero);
        return tvm::if_then_else(
            all_not_zero, x(index_tuple), make_const(x->dtype, 0));
      }
      return x(index_tuple);
    }, name, tag);
}

/*!
* \brief Perform Dilation2 on height,width and channel dimension of data.
*
* \param x The input tensor
* \param kernel_size Vector of three ints: {kernel_height, kernel_width, kernel_channel}
* \param stride_size Vector of two ints: {stride_height, stride_width}
* \param padding_size Vector of two ints: {padding_height, padding_width}
* \param depth_axis index of the depth dimension
* \param height_axis index of the height dimension
* \param width_axis index of the width dimension
* \param count_include_pad Whether include padding in the calculation
*
* \return The output tensor in same layout order
*/
/*
inline Tensor dilation2dImpl(const Tensor& x,
                        const Array<Expr>& kernel_size,
                        const Array<Expr>& stride_size,
                        const Array<Expr>& padding_size,
                        const size_t depth_axis,
                        const size_t height_axis,
                        const size_t width_axis,
                        bool count_include_pad) {
  CHECK(x->shape.size() >= 3) << "Dilation input must >= 3-D (H, W, C)";
  CHECK_EQ(kernel_size.size(), 3) << "Pooling kernel_size must have 2 elements";
  CHECK_EQ(stride_size.size(), 3) << "Pooling stride_size must have 2 elements";
  CHECK_EQ(padding_size.size(), 6) << "Pooling padding_size must have 4 elements";

  auto kernel_depth = cast(Int(32), kernel_size[0]);
  auto kernel_height = cast(Int(32), kernel_size[1]);
  auto kernel_width = cast(Int(32), kernel_size[2]);
  auto stride_depth = cast(Int(32), stride_size[0]);
  auto stride_height = cast(Int(32), stride_size[1]);
  auto stride_width = cast(Int(32), stride_size[2]);

  auto depth = x->shape[depth_axis];
  auto height = x->shape[height_axis];
  auto width = x->shape[width_axis];

  auto pad_backward = cast(Int(32), padding_size[0]);
  auto pad_top = cast(Int(32), padding_size[1]);
  auto pad_left = cast(Int(32), padding_size[2]);
  auto pad_forward = cast(Int(32), padding_size[3]);
  auto pad_bottom = cast(Int(32), padding_size[4]);
  auto pad_right = cast(Int(32), padding_size[5]);

  Array<Expr> pad_before(std::vector<Expr>(x->shape.size(), 0));
  pad_before.Set(height_axis, pad_top);
  pad_before.Set(width_axis, pad_left);

  Array<Expr> pad_after(std::vector<Expr>(x->shape.size(), 0));
  pad_after.Set(height_axis, pad_bottom);
  pad_after.Set(width_axis, pad_right);

  auto out_height = tvm::ir::Simplify(
      indexdiv(height - kernel_height + pad_top + pad_bottom, stride_height) + 1);
  auto out_width = tvm::ir::Simplify(
      indexdiv(width - kernel_width + pad_left + pad_right, stride_width) + 1);

  auto dheight = tvm::reduce_axis(Range(0, kernel_height));
  auto dwidth = tvm::reduce_axis(Range(0, kernel_width));

  Array<Expr> out_shape = x->shape;
  out_shape.Set(height_axis, out_height);
  out_shape.Set(width_axis, out_width);

  const int64_t *padding_h0 = as_const_int(pad_top);
  const int64_t *padding_w0 = as_const_int(pad_left);
  const int64_t *padding_h1 = as_const_int(pad_bottom);
  const int64_t *padding_w1 = as_const_int(pad_right);
  
  const bool do_pad = ( (padding_h0 && *padding_h0) || (padding_w0 && *padding_w0)) ||
                      ( (padding_h1 && *padding_h1) || (padding_w1 && *padding_w1));


auto temp = do_pad ? pad(x, pad_before, pad_after, x->dtype.min(), "pad_temp") : x;

return tvm::compute(out_shape, [&](const Array<Var>& output) {
  Array<Expr> indices;
  for (const Var& var : output) indices.push_back(var);
  indices.Set(height_axis, output[height_axis] * stride_height + dheight);
  indices.Set(width_axis, output[width_axis] * stride_width + dwidth);
  return tvm::max(temp(indices), { dheight, dwidth });
    }, "tensor", "dilation2d");
}
*/
}  // namespace nn
}  // namespace topi
#endif  // TOPI_NN_DILATE_H_
