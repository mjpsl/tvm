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
 *  Copyright (c) 2018 by Contributors
 * \file dilation.cc
 * \brief Dilation operators
 */
#include <tvm/data_layout.h>
#include <tvm/ir_pass.h>
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/nn.h>
#include <vector>

#include "../../pass/alter_op_layout.h"
//#include "topi/nn/dilate.h"
#include "dilation.h"


namespace tvm {
namespace relay {

// relay.nn.dilation2d
TVM_REGISTER_NODE_TYPE(Dilation2DAttrs);

template<typename T>
Array<Array<Layout> > Dilation2DInferCorrectLayout(
    const Attrs& attrs,
    const Array<Layout>& new_in_layouts,
    const Array<Layout>& old_in_layouts,
    const Array<Array<IndexExpr>> &old_in_shapes) {
  const T* params = attrs.as<T>();

  // We always make other operators to fit the layouts of convolution layers
  // So this inference ignores all inputs
  return Array<Array<Layout> >{{"NHWC", "HWC"},
                               {"NHWC"}};
}
/*
template<typename AttrType>
Array<Tensor> Dilation2DCompute(const Attrs& attrs,
                            const Array<Tensor>& inputs,
                            const Type& out_type
                            //const Target& target
                                ) {
  static const Layout kNHWC("NHWC");
  const auto* param = attrs.as<AttrType>();
  CHECK(param != nullptr);
  auto pool_size = param->pool_size;
  auto strides = param->strides;
  auto padding = param->padding;
//  Layout layout(param->layout);
//
//  CHECK(BijectiveLayoutNode::make(layout, kNCHW).defined())
//      << "max_pool2d currently only supports layouts that are convertible from NCHW";
//  CHECK_EQ(layout.IndexOf(LayoutAxis::Get('h')), -1)
//      << "max_pool2d does not support input split on height";
//  CHECK_EQ(layout.IndexOf(LayoutAxis::Get('w')), -1)
//      << "max_pool2d does not support input split on width";
//
//  CHECK(inputs[0].ndim() == 4U ||
//        inputs[0].ndim() == 5U ||
//        inputs[0].ndim() == 6U)
//      << "Pool2D only support 4-D input (e.g., NCHW)"
//      << " or 5-D input (e.g. NCHWc on for vector instructions)"
//      << " or 6-D input (e.g. NCHWnc for tensor accelerators)";

//  if (param->padding.size() == 1) {
//    padding.push_back(padding[0]);
//    padding.push_back(padding[0]);
//    padding.push_back(padding[0]);
//  } else if (param->padding.size() == 2) {
//    padding.push_back(padding[0]);
//    padding.push_back(padding[1]);
//  }
  return Array<Tensor>{
      topi::nn::dilation2d(inputs[0], pool_size, strides, padding)
            };

}
*/

// Positional relay function to create dilation2d operator
// used by frontend FFI.
Expr MakeDilation2D(Expr data,
                Expr weight,
                Array<IndexExpr> strides,
                Array<IndexExpr> rate,
                Array<IndexExpr> padding
                
        ) {
  auto attrs = make_node<Dilation2DAttrs>();
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->rate = std::move(rate);
  static const Op& op = Op::Get("nn.dilation2d");
  std::cout<<"+++++++++Cpp call MakeDilation2D"<<std::endl;
  return CallNode::make(op, {data, weight}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.nn._make.dilation2d")
.set_body_typed(MakeDilation2D);


RELAY_REGISTER_OP("nn.dilation2d")
.describe(R"code(2D dilation layer (e.g. dilation over 2D image data,
This layer creates a dilated kernel that is convolved
with the layer input to produce a tensor of outputs.

 **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
 **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
 **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.
)code"  TVM_ADD_FILELINE)
.set_attrs_type<Dilation2DAttrs>()
.set_num_inputs(2)
.add_argument("data", "Tensor", "The input tensor.")
.add_argument("weight", "Tensor", "The weight tensor.")
.set_support_level(2)
//.set_attr<FTVMCompute>("FTVMCompute", Dilation2DCompute<Dilation2DAttrs>)
.add_type_rel("Dilation2D", Dilation2DRel<Dilation2DAttrs>);
//.set_attr<FInferCorrectLayout>("FInferCorrectLayout", Dilation2DInferCorrectLayout<Dilation2DAttrs>);
//.set_attr<FTVMCompute>("FTVMCompute", Pool2DCompute<MaxPool2DAttrs, topi::nn::kMaxPool>);


}  // namespace relay
}  // namespace tvm
