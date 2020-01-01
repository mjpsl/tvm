# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Example code to do convolution."""
import os
import numpy as np
import tvm
import topi
import topi.testing
from tvm.contrib.pickle_memoize import memoize
from topi.util import get_const_tuple

#Changes made
# 1. removed num_filter
# what is the memoize function part?
# 2. appending to each function name 'dilation2D'

def verify_dilation2D(batch,  in_channel,in_size, kernel, stride, rate, padding):
    in_height = in_width = in_size

    A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel), name='W')
    B = topi.nn.dilation2d(A, W, stride,rate, padding)

    a_shape = get_const_tuple(A.shape)
    w_shape = get_const_tuple(W.shape)
    dtype = A.dtype

    @memoize("topi.tests.test_topi_dilation2D.verify_dilate")
    def get_ref_data():
        a_np = np.random.uniform(size=a_shape).astype(dtype)
        w_np = np.random.uniform(size=w_shape).astype(dtype)
        dw_np = topi.testing.dilate_python(w_np, (1,rate, rate, 1))
        b_np = topi.testing.dilation2D_python(a_np, dw_np, stride, padding)
        return a_np, w_np, b_np
    a_np, w_np, b_np = get_ref_data()

    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_dilation2D([B])
        ctx = tvm.context(device, 0)
        a = tvm.nd.array(a_np, ctx)
        w = tvm.nd.array(w_np, ctx)
        b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        func = tvm.build(s, [A, W, B], device)
        func(a, w, b)
        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-5)

    for device in ['llvm']:
        check_device(device)


def test_dilation2D():
    #(batch,  in_channel,in_size, kernel, stride, rate, padding)
    verify_dilation2D(1, 3, 5, 3, [1,1,1,1], [1,1,1,1], "VALID")
    verify_dilation2D(1, 3, 5, 3, [1,1,1,1], [1,1,1,1], "SAME")
    verify_dilation2D(1, 3, 5, 3, [1,1,1,1], [1,2,2,1], "VALID")
    verify_dilation2D(1, 3, 28, 5, [1,2,2,1], [1,1,1,1], "VALID")
    verify_dilation2D(4, 3, 5, 3, [1,1,1,1], [1,1,1,1], "VALID")


if __name__ == "__main__":
    test_dilation2D()
