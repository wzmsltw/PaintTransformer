import render_utils
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import math

def crop(img, h, w):
    H, W = img.shape[-2:]
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    remainder_h = (H - h) % 2
    remainder_w = (W - w) % 2
    img = img[:, :, pad_h:H - pad_h - remainder_h, pad_w:W - pad_w - remainder_w]
    return img

def stroke_net_predict(img_patch, result_patch, patch_size, net_g, stroke_num, patch_num):
    """
    stroke_net_predict
    """
    img_patch = img_patch.transpose([0, 2, 1]).reshape([-1, 3, patch_size, patch_size])
    result_patch = result_patch.transpose([0, 2, 1]).reshape([-1, 3, patch_size, patch_size])
    #*----- Stroke Predictor -----*#
    shape_param, stroke_decision = net_g(img_patch, result_patch)
    stroke_decision = (stroke_decision > 0).astype('float32')
    #*----- sampling color -----*#
    grid = shape_param[:, :, :2].reshape([img_patch.shape[0] * stroke_num, 1, 1, 2])
    img_temp = img_patch.unsqueeze(1).tile([1, stroke_num, 1, 1, 1]).reshape([
        img_patch.shape[0] * stroke_num, 3, patch_size, patch_size])
    color = nn.functional.grid_sample(img_temp, 2 * grid - 1, align_corners=False).reshape([
        img_patch.shape[0], stroke_num, 3])
    param = paddle.concat([shape_param, color], axis=-1)

    param = param.reshape([-1, 8])
    param[:, :2] = param[:, :2] / 2 + 0.25
    param[:, 2:4] = param[:, 2:4] / 2
    param = param.reshape([1, patch_num, patch_num, stroke_num, 8])
    decision = stroke_decision.reshape([1, patch_num, patch_num, stroke_num])#.astype('bool')
    return param, decision


def param2img_parallel(param, decision, meta_brushes, cur_canvas, stroke_num=8):
    """
        Input stroke parameters and decisions for each patch, meta brushes, current canvas, frame directory,
        and whether there is a border (if intermediate painting results are required).
        Output the painting results of adding the corresponding strokes on the current canvas.
        Args:
            param: a tensor with shape batch size x patch along height dimension x patch along width dimension
             x n_stroke_per_patch x n_param_per_stroke
            decision: a 01 tensor with shape batch size x patch along height dimension x patch along width dimension
             x n_stroke_per_patch
            meta_brushes: a tensor with shape 2 x 3 x meta_brush_height x meta_brush_width.
            The first slice on the batch dimension denotes vertical brush and the second one denotes horizontal brush.
            cur_canvas: a tensor with shape batch size x 3 x H x W,
             where H and W denote height and width of padded results of original images.

        Returns:
            cur_canvas: a tensor with shape batch size x 3 x H x W, denoting painting results.
        """
    # param: b, h, w, stroke_per_patch, param_per_stroke
    # decision: b, h, w, stroke_per_patch
    b, h, w, s, p = param.shape
    h, w = int(h), int(w)
    param = param.reshape([-1, 8])
    decision = decision.reshape([-1, 8])
    
    H, W = cur_canvas.shape[-2:]
    is_odd_y = h % 2 == 1
    is_odd_x = w % 2 == 1
    render_size_y = 2 * H // h
    render_size_x = 2 * W // w

    even_idx_y = paddle.arange(0, h, 2)
    even_idx_x = paddle.arange(0, w, 2)    
    if h > 1:
        odd_idx_y = paddle.arange(1, h, 2)
    if w > 1:
        odd_idx_x = paddle.arange(1, w, 2)

    cur_canvas = F.pad(cur_canvas, [render_size_x // 4, render_size_x // 4,
                                    render_size_y // 4, render_size_y // 4])

    valid_foregrounds = render_utils.param2stroke(param, render_size_y, render_size_x, meta_brushes)
    
    #* ----- load dilation/erosion ---- *#
    dilation = render_utils.Dilation2d(m=1)
    erosion = render_utils.Erosion2d(m=1)

    #* ----- generate alphas ----- *#
    valid_alphas = (valid_foregrounds > 0).astype('float32')
    valid_foregrounds = valid_foregrounds.reshape([-1, stroke_num, 1, render_size_y, render_size_x])
    valid_alphas = valid_alphas.reshape([-1, stroke_num, 1, render_size_y, render_size_x])

    temp = [dilation(valid_foregrounds[:, i, :, :, :]) for i in range(stroke_num)]
    valid_foregrounds = paddle.stack(temp, axis=1)
    valid_foregrounds = valid_foregrounds.reshape([-1, 1, render_size_y, render_size_x])

    temp = [erosion(valid_alphas[:, i, :, :, :]) for i in range(stroke_num)]
    valid_alphas = paddle.stack(temp, axis=1)
    valid_alphas = valid_alphas.reshape([-1, 1, render_size_y, render_size_x])

    foregrounds = valid_foregrounds.reshape([-1, h, w, stroke_num, 1, render_size_y, render_size_x])
    alphas = valid_alphas.reshape([-1, h, w, stroke_num, 1, render_size_y, render_size_x])
    decision = decision.reshape([-1, h, w, stroke_num, 1, 1, 1])
    param = param.reshape([-1, h, w, stroke_num, 8])
    
    def partial_render(this_canvas, patch_coord_y, patch_coord_x):
        canvas_patch = F.unfold(this_canvas, [render_size_y, render_size_x], strides=[render_size_y // 2, render_size_x // 2])
        # canvas_patch: b, 3 * py * px, h * w
        canvas_patch = canvas_patch.reshape([b, 3, render_size_y, render_size_x, h, w])
        canvas_patch = canvas_patch.transpose([0, 4, 5, 1, 2, 3])
        selected_canvas_patch = paddle.gather(canvas_patch, patch_coord_y, 1)
        selected_canvas_patch = paddle.gather(selected_canvas_patch, patch_coord_x, 2)
        selected_canvas_patch = selected_canvas_patch.reshape([0, 0, 0, 1, 3, render_size_y, render_size_x])
        selected_foregrounds = paddle.gather(foregrounds, patch_coord_y, 1)
        selected_foregrounds = paddle.gather(selected_foregrounds, patch_coord_x, 2)
        selected_alphas = paddle.gather(alphas, patch_coord_y, 1)
        selected_alphas = paddle.gather(selected_alphas, patch_coord_x, 2)
        selected_decisions = paddle.gather(decision, patch_coord_y, 1)
        selected_decisions = paddle.gather(selected_decisions, patch_coord_x, 2)
        selected_color = paddle.gather(param, patch_coord_y, 1)
        selected_color = paddle.gather(selected_color, patch_coord_x, 2)
        selected_color = paddle.gather(selected_color, paddle.to_tensor([5,6,7]), 4)
        selected_color = selected_color.reshape([0, 0, 0, stroke_num, 3, 1, 1])

        for i in range(stroke_num):
            i = paddle.to_tensor(i)

            cur_foreground = paddle.gather(selected_foregrounds, i, 3)
            cur_alpha = paddle.gather(selected_alphas, i, 3)
            cur_decision = paddle.gather(selected_decisions, i, 3)
            cur_color = paddle.gather(selected_color, i, 3)
            cur_foreground = cur_foreground * cur_color
            selected_canvas_patch = cur_foreground * cur_alpha * cur_decision + selected_canvas_patch * (1 - cur_alpha * cur_decision)

        selected_canvas_patch = selected_canvas_patch.reshape([0, 0, 0, 3, render_size_y, render_size_x])
        this_canvas = selected_canvas_patch.transpose([0, 3, 1, 4, 2, 5])
        
        # this_canvas: b, 3, h_half, py, w_half, px
        h_half = this_canvas.shape[2]
        w_half = this_canvas.shape[4]
        this_canvas = this_canvas.reshape([b, 3, h_half * render_size_y, w_half * render_size_x])
        # this_canvas: b, 3, h_half * py, w_half * px
        return this_canvas

    # even - even area
    # 1 | 0
    # 0 | 0
    canvas = partial_render(cur_canvas, even_idx_y, even_idx_x)
    if not is_odd_y:
        canvas = paddle.concat([canvas, cur_canvas[:, :, -render_size_y // 2:, :canvas.shape[3]]], axis=2)
    if not is_odd_x:
        canvas = paddle.concat([canvas, cur_canvas[:, :, :canvas.shape[2], -render_size_x // 2:]], axis=3)
    cur_canvas = canvas

    # odd - odd area
    # 0 | 0
    # 0 | 1
    if h > 1 and w > 1:
        canvas = partial_render(cur_canvas, odd_idx_y, odd_idx_x)
        canvas = paddle.concat([cur_canvas[:, :, :render_size_y // 2, -canvas.shape[3]:], canvas], axis=2)
        canvas = paddle.concat([cur_canvas[:, :, -canvas.shape[2]:, :render_size_x // 2], canvas], axis=3)
        if is_odd_y:
            canvas = paddle.concat([canvas, cur_canvas[:, :, -render_size_y // 2:, :canvas.shape[3]]], axis=2)
        if is_odd_x:
            canvas = paddle.concat([canvas, cur_canvas[:, :, :canvas.shape[2], -render_size_x // 2:]], axis=3)
        cur_canvas = canvas
    
    # odd - even area
    # 0 | 0
    # 1 | 0
    if h > 1:
        canvas = partial_render(cur_canvas, odd_idx_y, even_idx_x)
        canvas = paddle.concat([cur_canvas[:, :, :render_size_y // 2, :canvas.shape[3]], canvas], axis=2)
        if is_odd_y:
            canvas = paddle.concat([canvas, cur_canvas[:, :, -render_size_y // 2:, :canvas.shape[3]]], axis=2)
        if not is_odd_x:
            canvas = paddle.concat([canvas, cur_canvas[:, :, :canvas.shape[2], -render_size_x // 2:]], axis=3)
        cur_canvas = canvas

    # odd - even area
    # 0 | 1
    # 0 | 0
    if w > 1:
        canvas = partial_render(cur_canvas, even_idx_y, odd_idx_x)
        canvas = paddle.concat([cur_canvas[:, :, :canvas.shape[2], :render_size_x // 2], canvas], axis=3)
        if not is_odd_y:
            canvas = paddle.concat([canvas, cur_canvas[:, :, -render_size_y // 2:, -canvas.shape[3]:]], axis=2)
        if is_odd_x:
            canvas = paddle.concat([canvas, cur_canvas[:, :, :canvas.shape[2], -render_size_x // 2:]], axis=3)
        cur_canvas = canvas

    cur_canvas = cur_canvas[:, :, render_size_y // 4:-render_size_y // 4, render_size_x // 4:-render_size_x // 4]

    return cur_canvas



def render_parallel(original_img, net_g, meta_brushes):

    patch_size = 32
    stroke_num = 8

    with paddle.no_grad():
        
        original_h, original_w = original_img.shape[-2:]
        K = max(math.ceil(math.log2(max(original_h, original_w) / patch_size)), 0)
        original_img_pad_size = patch_size * (2 ** K)
        original_img_pad = render_utils.pad(original_img, original_img_pad_size, original_img_pad_size)
        final_result = paddle.zeros_like(original_img)

        for layer in range(0, K + 1):
            layer_size = patch_size * (2 ** layer)

            img = F.interpolate(original_img_pad, (layer_size, layer_size))
            result = F.interpolate(final_result, (layer_size, layer_size))
            img_patch = F.unfold(img, [patch_size, patch_size], strides=[patch_size, patch_size])
            result_patch = F.unfold(result, [patch_size, patch_size], strides=[patch_size, patch_size])

            # There are patch_num * patch_num patches in total
            patch_num = (layer_size - patch_size) // patch_size + 1
            param, decision = stroke_net_predict(img_patch, result_patch, patch_size, net_g, stroke_num, patch_num)

            #print(param.shape, decision.shape)
            final_result = param2img_parallel(param, decision, meta_brushes, final_result)
        
        # paint another time for last layer
        border_size = original_img_pad_size // (2 * patch_num)
        img = F.interpolate(original_img_pad, (layer_size, layer_size))
        result = F.interpolate(final_result, (layer_size, layer_size))
        img = F.pad(img, [patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2])
        result = F.pad(result, [patch_size // 2, patch_size // 2, patch_size // 2, patch_size // 2])
        img_patch = F.unfold(img, [patch_size, patch_size], strides=[patch_size, patch_size])
        result_patch = F.unfold(result, [patch_size, patch_size], strides=[patch_size, patch_size])
        final_result = F.pad(final_result, [border_size, border_size, border_size, border_size])
        patch_num = (img.shape[2] - patch_size) // patch_size + 1
        #w = (img.shape[3] - patch_size) // patch_size + 1

        param, decision = stroke_net_predict(img_patch, result_patch, patch_size, net_g, stroke_num, patch_num)

        final_result = param2img_parallel(param, decision, meta_brushes, final_result)

        final_result = final_result[:, :, border_size:-border_size, border_size:-border_size]
        final_result = (final_result.numpy().squeeze().transpose([1,2,0])[:,:,::-1] * 255).astype(np.uint8)
        return final_result