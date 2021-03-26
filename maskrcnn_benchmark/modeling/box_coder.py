# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """

        TO_REMOVE = 1  # TODO remove
        ex_widths = proposals[:, 2] - proposals[:, 0] + TO_REMOVE
        ex_heights = proposals[:, 3] - proposals[:, 1] + TO_REMOVE
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + TO_REMOVE
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """
        print('decoding...')
        print(rel_codes.dtype) # this is torch.float32
        # change to torch.float64
        boxes = boxes.to(rel_codes.dtype)
        print(boxes)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)
        print(f'transferred...')
        print(f'dx = {dx.shape}')
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]



        pred_boxes = torch.zeros_like(rel_codes)
        #print(f'rel_codes: {rel_codes}, type: {rel_codes.dtype}')
        #print(f'pre_boxes: {pred_boxes}, type: {pred_boxes.dtype}, shape: {pred_boxes.shape}')
        '''
        print(f'shift calculations... ')
        print(f'pred_ctr_x: {pred_ctr_x}')
        print(f'pred_ctr_y: {pred_ctr_y}')
        print(f'pred_w: {pred_w}')
        print(f'pred_h: {pred_h}')
        '''

        print(f'pred_boxes: {pred_boxes.shape}')
        print(f'pred_h: {pred_h.shape}')
        print(f'pred_w: {pred_w.shape}')
        print(f'pred_boxes dtype: {pred_boxes.dtype}')
        print(f'pred_h dtype: {pred_h.dtype}')
        print(f'pred_w dtype: {pred_w.dtype}')

        # how to deal with it when pred_boxes [n, ]
        dev = pred_boxes.device
        pred_boxes = pred_boxes.to(torch.float32)
        pred_ctr_x = pred_ctr_x.to(torch.float32)
        pred_ctr_y = pred_ctr_y.to(torch.float32)
        pred_w = pred_w.to(torch.float64)
        pred_h = pred_h.to(torch.float64)
        
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - torch.tensor(0.5, dtype = torch.float32, device = dev) * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - torch.tensor(0.5, dtype = torch.float32, device = dev) * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 2::4] = pred_ctr_x + torch.tensor(0.5, dtype = torch.float32, device = dev) * pred_w - torch.tensor(1.0, dtype = torch.float32, device = dev)
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes[:, 3::4] = pred_ctr_y + torch.tensor(0.5, dtype = torch.float32, device = dev) * pred_h - torch.tensor(1.0, dtype = torch.float32, device = dev)
 
        # need to verify the tensor values!!
        #pred_boxes = torch.cat((pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h, pred_ctr_x + 0.5 * pred_w - 1, pred_ctr_y + 0.5 * pred_h - 1), dim = 1)
        
        #print(f'finisehd shift calculations... ')
        print(f'pred_boxes: {pred_boxes}')
        print(f'pred_boxes shape: {pred_boxes.shape}')
        print(f'pred_boxes dtype: {pred_boxes.dtype}')
        return pred_boxes
