# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from maskrcnn_benchmark import _C
from maskrcnn_benchmark.layers.tensor_roi_align import tensor_roi_align


class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        print(f'my input device type is {input.device.type}')
        #print(f'input type is {input.dtype}')
        #print(f'rois type is {rois.dtype}')
        rois = rois.to(torch.float32)
        #print(f'input type is {input.dtype}')
        print(f'input shape is {input.shape}')
        print(f'rois shape is {rois.shape}')
        if self.sampling_ratio > 0 and input.device.type == 'xla':
            print('tensor-impl of ROI_align')
            batch_size = input.size(0)
            num_rois = rois.size(0)
            # rois shape [num, 5]
            if num_rois != 0:
                rois = rois[:, 1:].reshape(batch_size, num_rois, -1) # ignore indexs
                ret = tensor_roi_align(
                input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
            )
            else:
                ret = torch.zeros(0, input.shape[1], self.output_size[0], self.output_size[0], device = input.device) 
            return ret

        print('CPU imp of ROI_align')
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )   


    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr
