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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""Dilation2D operators"""
from __future__ import absolute_import as _abs
from collections import namedtuple
import tvm

from .pad import pad
from .util import get_pad_tuple
from ..util import simplify, get_const_tuple
from .winograd_util import winograd_transform_matrices


@tvm.target.generic_func
def dilation2d(input, filter, strides, padding, rate, layout='NHWC', out_dtype=None):
    """Dilation2D operator.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    strides : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    rate: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    # search platform specific declaration first
    # default declaration
    print('topi/python/topi/nn/dialtion2d function will call dilation2d_nhwc')
    if layout == 'NHWC':
        return dilation2d_nhwc(input, filter, strides, padding, rate, out_dtype)
    raise ValueError("not support this layout {} yet".format(layout))


@tvm.target.generic_func
def dilation2d_legalize(attrs, inputs, types):
    """Legalizes Dilation2D op.

    Parameters
    ----------
    attrs : tvm.attrs.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    # not to change by default
    return None


def _get_workload(data, kernel, stride, padding, out_dtype, data_layout='NHWC'):
    """ Get the workload structure. """
    if data_layout == 'NCHW':
        _, CI, IH, IW = [x.value for x in data.shape]
    elif data_layout == 'NHWC':
        _, IH, IW, CI = [x.value for x in data.shape]
    elif data_layout == 'HWCN':
        IH, IW, CI, _ = [x.value for x in data.shape]
    else:
        raise ValueError("not support this layout {} yet".format(data_layout))

    if data_layout == 'NCHW':
        CO, CIG, KH, KW = [x.value for x in kernel.shape]
    else:
        KH, KW, CIG, CO = [x.value for x in kernel.shape]

    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    GRPS = CI // CIG
    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride
    assert (data.dtype == kernel.dtype) or (data.dtype == 'uint8' and kernel.dtype == 'int8'), \
        "Do not support inputs with different data types now. ' \
        '{} vs. {}".format(data.dtype, kernel.dtype)
    return Workload(data.dtype, out_dtype, IH, IW, CI, GRPS, CO, KH, KW, HPAD, WPAD, HSTR, WSTR)



@tvm.target.generic_func
def dilation2d_nhwc(Input, Filter, stride, padding, rate, out_dtype='float32'):
    """Depthwise convolution nhwc forward operator.

    Parameters
    ----------
    Input : tvm.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.Tensor
        3-D with shape [filter_height, filter_width, in_channel]

    stride : tuple of two ints
        The spatial stride along height and width

    padding : int or str
        Padding size, or ['VALID', 'SAME']

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str, optional
        Output data type

    Returns
    -------
    Output : tvm.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    print('Inside dilation2d implementation')
    out_dtype = Input.dtype if out_dtype is None else out_dtype

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(rate, int):
        dilation_h = dilation_w = rate
    else:
        dilation_h, dilation_w = rate

    batch, in_height, in_width, in_channel = Input.shape
    # shape of dilated kernel
    filter_height, filter_width, filter_channel = Filter.shape

    dilated_kernel_h = (filter_height - 1) * dilation_h + 1
    dilated_kernel_w = (filter_width - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w))
    out_channel = in_channel
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # padding stage
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    # depthconv stage
    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod

    di = tvm.reduce_axis((0, filter_height), name='di')
    dj = tvm.reduce_axis((0, filter_width), name='dj')
    Output = tvm.compute(
        (batch, out_height, out_width, out_channel),
        lambda b, i, j, c: tvm.max(
            (PaddedInput[b, i * stride_h + di * dilation_h, j * stride_w + dj * dilation_w,
                         c,].astype(out_dtype) +
             Filter[di, dj, c].astype(out_dtype)),
            axis=[di, dj]),
        name='dilation2d', tag="dilation2d")
    return Output
