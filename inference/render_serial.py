# !/usr/bin/env python3
"""
codes for oilpainting style transfer.
"""
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from PIL import Image
import math
import cv2
import render_utils
import time


def get_single_layer_lists(param, decision, ori_img, render_size_x, render_size_y, h, w, meta_brushes, dilation, erosion, stroke_num):
    """
    get_single_layer_lists
    """
    valid_foregrounds = render_utils.param2stroke(param[:, :], render_size_y, render_size_x, meta_brushes)

    valid_alphas = (valid_foregrounds > 0).astype('float32')
    valid_foregrounds = valid_foregrounds.reshape([-1, stroke_num, 1, render_size_y, render_size_x])
    valid_alphas = valid_alphas.reshape([-1, stroke_num, 1, render_size_y, render_size_x])

    temp = [dilation(valid_foregrounds[:, i, :, :, :]) for i in range(stroke_num)]
    valid_foregrounds = paddle.stack(temp, axis=1)
    valid_foregrounds = valid_foregrounds.reshape([-1, 1, render_size_y, render_size_x])

    temp = [erosion(valid_alphas[:, i, :, :, :]) for i in range(stroke_num)]
    valid_alphas = paddle.stack(temp, axis=1)
    valid_alphas = valid_alphas.reshape([-1, 1, render_size_y, render_size_x])

    patch_y = 4 * render_size_y // 5
    patch_x = 4 * render_size_x // 5

    img_patch = ori_img.reshape([1, 3, h, ori_img.shape[2]//h, w, ori_img.shape[3]//w])
    img_patch = img_patch.transpose([0, 2, 4, 1, 3, 5])[0]

    xid_list = []
    yid_list = []
    error_list = []

    for flag_idx, flag in enumerate(decision.cpu().numpy()):
        if flag:
            flag_idx = flag_idx // stroke_num
            x_id = flag_idx % w
            flag_idx = flag_idx // w
            y_id = flag_idx % h
            xid_list.append(x_id)
            yid_list.append(y_id)
    
    inner_fores = valid_foregrounds[:, :, render_size_y // 10:9 * render_size_y // 10, 
                                    render_size_x // 10:9 * render_size_x // 10]
    inner_alpha = valid_alphas[:, :, render_size_y // 10:9 * render_size_y // 10, 
                                    render_size_x // 10:9 * render_size_x // 10]
    inner_fores = inner_fores.reshape([h * w, stroke_num, 1, patch_y, patch_x])
    inner_alpha = inner_alpha.reshape([h * w, stroke_num, 1, patch_y, patch_x])
    inner_real = img_patch.reshape([h * w, 3, patch_y, patch_x]).unsqueeze(1)

    R = param[:, 5]
    G = param[:, 6]
    B = param[:, 7]#, G, B = param[5:]
    R = R.reshape([-1, stroke_num]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    G = G.reshape([-1, stroke_num]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    B = B.reshape([-1, stroke_num]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    error_R = R * inner_fores - inner_real[:, :, 0:1, :, :]
    error_G = G * inner_fores - inner_real[:, :, 1:2, :, :]
    error_B = B * inner_fores - inner_real[:, :, 2:3, :, :]
    error = paddle.abs(error_R) + paddle.abs(error_G)+ paddle.abs(error_B)

    error = error * inner_alpha
    error = paddle.sum(error, axis=(2, 3, 4)) / paddle.sum(inner_alpha, axis=(2, 3, 4))
    error_list = error.reshape([-1]).numpy()[decision.numpy()]
    error_list = list(error_list)

    valid_foregrounds = paddle.to_tensor(valid_foregrounds.numpy()[decision.numpy()])
    valid_alphas = paddle.to_tensor(valid_alphas.numpy()[decision.numpy()])
    
    selected_param = paddle.to_tensor(param.numpy()[decision.numpy()])
    return xid_list, yid_list, valid_foregrounds, valid_alphas, error_list, selected_param


def get_single_stroke_on_full_image_A(x_id, y_id, valid_foregrounds, valid_alphas, param, original_img, 
                                        render_size_x, render_size_y, patch_x, patch_y):
    """
    get_single_stroke_on_full_image_A
    """
    tmp_foreground = paddle.zeros_like(original_img)

    patch_y_num = original_img.shape[2] // patch_y
    patch_x_num = original_img.shape[3] // patch_x

    brush = valid_foregrounds.unsqueeze(0)
    color_map = param[5:]
    brush = brush.tile([1, 3, 1, 1])
    color_map = color_map.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)#.repeat(1, 1, H, W)
    brush = brush * color_map

    pad_l = x_id * patch_x
    pad_r = (patch_x_num - x_id - 1) * patch_x
    pad_t = y_id * patch_y
    pad_b = (patch_y_num - y_id - 1) * patch_y
    tmp_foreground = nn.functional.pad(brush, [pad_l, pad_r, pad_t, pad_b])
    tmp_foreground = tmp_foreground[:, :, render_size_y // 10:-render_size_y // 10, 
                                    render_size_x // 10:-render_size_x // 10]
   
    tmp_alpha = nn.functional.pad(valid_alphas.unsqueeze(0), [pad_l, pad_r, pad_t, pad_b])
    tmp_alpha = tmp_alpha[:, :, render_size_y // 10:-render_size_y // 10, render_size_x // 10:-render_size_x // 10]
    return tmp_foreground, tmp_alpha

def get_single_stroke_on_full_image_B(x_id, y_id, valid_foregrounds, valid_alphas, param, 
                                        original_img, render_size_x, render_size_y, patch_x, patch_y):
    """
    get_single_stroke_on_full_image_B
    """
    x_expand = patch_x // 2 + render_size_x // 10
    y_expand = patch_y // 2 + render_size_y // 10

    pad_l = x_id * patch_x
    pad_r = original_img.shape[3] + 2 * x_expand - (x_id * patch_x + render_size_x)
    pad_t = y_id * patch_y
    pad_b = original_img.shape[2] + 2 * y_expand - (y_id * patch_y + render_size_y)

    brush = valid_foregrounds.unsqueeze(0)
    color_map = param[5:]
    brush = brush.tile([1, 3, 1, 1])
    color_map = color_map.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)#.repeat(1, 1, H, W)
    brush = brush * color_map

    tmp_foreground = nn.functional.pad(brush, [pad_l, pad_r, pad_t, pad_b])

    tmp_foreground = tmp_foreground[:, :, y_expand:- y_expand, x_expand:-x_expand]
    tmp_alpha = nn.functional.pad(valid_alphas.unsqueeze(0), [pad_l, pad_r, pad_t, pad_b])
    tmp_alpha = tmp_alpha[:, :, y_expand:- y_expand, x_expand:-x_expand]
    return tmp_foreground, tmp_alpha

def stroke_net_predict(img_patch, result_patch, patch_size, net_g, stroke_num):
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
    stroke_param = paddle.concat([shape_param, color], axis=-1)

    param = stroke_param.reshape([-1, 8])
    decision = stroke_decision.reshape([-1]).astype('bool')
    param[:, :2] = param[:, :2] / 1.25 + 0.1
    param[:, 2:4] = param[:, 2:4] / 1.25
    return param, decision


def sort_strokes(params, decision, scores):
    """
    sort_strokes
    """
    sorted_scores, sorted_index = paddle.sort(scores, axis=1, descending=False)
    sorted_params = []
    for idx in range(8):
        tmp_pick_params = paddle.gather(params[:, :, idx], axis=1, index=sorted_index)
        sorted_params.append(tmp_pick_params)
    sorted_params = paddle.stack(sorted_params, axis=2)
    sorted_decison = paddle.gather(decision.squeeze(2), axis=1, index=sorted_index)
    return sorted_params, sorted_decison


def render_serial(original_img, net_g, meta_brushes):

    patch_size = 32
    stroke_num = 8
    H, W = original_img.shape[-2:]
    K = max(math.ceil(math.log2(max(H, W) / patch_size)), 0)

    dilation = render_utils.Dilation2d(m=1)
    erosion = render_utils.Erosion2d(m=1)
    frames_per_layer = [20, 20, 30, 40, 60]
    final_frame_list = []

    with paddle.no_grad():
        #* ----- read in image and init canvas ----- *#
        final_result = paddle.zeros_like(original_img)

        for layer in range(0, K + 1):
            t0 = time.time()
            layer_size = patch_size * (2 ** layer)

            img = nn.functional.interpolate(original_img, (layer_size, layer_size))
            result = nn.functional.interpolate(final_result, (layer_size, layer_size))
            img_patch = nn.functional.unfold(img, [patch_size, patch_size], 
                                            strides=[patch_size, patch_size])
            result_patch = nn.functional.unfold(result, [patch_size, patch_size], 
                                            strides=[patch_size, patch_size])
            h = (img.shape[2] - patch_size) // patch_size + 1
            w = (img.shape[3] - patch_size) // patch_size + 1
            render_size_y = int(1.25 * H // h)
            render_size_x = int(1.25 * W // w)

            #* -------------------------------------------------------------*#
            #* -------------generate strokes on window type A---------------*#
            #* -------------------------------------------------------------*#
            param, decision = stroke_net_predict(img_patch, result_patch, patch_size, net_g, stroke_num)
            expand_img = original_img
            wA_xid_list, wA_yid_list, wA_fore_list, wA_alpha_list, wA_error_list, wA_params = \
                get_single_layer_lists(param, decision, original_img, render_size_x, render_size_y, h, w, 
                                        meta_brushes, dilation, erosion, stroke_num)

            #* -------------------------------------------------------------*#
            #* -------------generate strokes on window type B---------------*#
            #* -------------------------------------------------------------*#
            #*----- generate input canvas and target patches -----*#
            wB_error_list = []

            img = nn.functional.pad(img, [patch_size // 2, patch_size // 2, 
                                    patch_size // 2, patch_size // 2])
            result = nn.functional.pad(result, [patch_size // 2, patch_size // 2, 
                                        patch_size // 2, patch_size // 2])
            img_patch = nn.functional.unfold(img, [patch_size, patch_size], 
                                            strides=[patch_size, patch_size])
            result_patch = nn.functional.unfold(result, [patch_size, patch_size], 
                                                strides=[patch_size, patch_size])
            h += 1
            w += 1

            param, decision = stroke_net_predict(img_patch, result_patch, patch_size, net_g, stroke_num)

            patch_y = 4 * render_size_y // 5
            patch_x = 4 * render_size_x // 5
            expand_img = nn.functional.pad(original_img, [patch_x // 2, patch_x // 2,
                                            patch_y // 2, patch_y // 2])
            wB_xid_list, wB_yid_list, wB_fore_list, wB_alpha_list, wB_error_list, wB_params = \
                get_single_layer_lists(param, decision, expand_img, render_size_x, render_size_y, h, w, 
                                        meta_brushes, dilation, erosion, stroke_num)
            #* -------------------------------------------------------------*#
            #* -------------rank strokes and plot stroke one by one---------*#
            #* -------------------------------------------------------------*#
            numA = len(wA_error_list)
            numB = len(wB_error_list)
            total_error_list = wA_error_list + wB_error_list
            sort_list = list(np.argsort(total_error_list))
            
            sample = 0
            samples = np.linspace(0, len(sort_list) - 2, frames_per_layer[layer]).astype(int)
            for ii in sort_list:
                ii = int(ii)
                if ii < numA:
                    x_id = wA_xid_list[ii]
                    y_id = wA_yid_list[ii]
                    valid_foregrounds = wA_fore_list[ii]
                    valid_alphas = wA_alpha_list[ii]
                    sparam = wA_params[ii]
                    tmp_foreground, tmp_alpha = get_single_stroke_on_full_image_A(x_id, y_id, 
                    valid_foregrounds, valid_alphas, sparam, original_img, render_size_x, render_size_y, patch_x, patch_y)
                else:
                    x_id = wB_xid_list[ii - numA]
                    y_id = wB_yid_list[ii - numA]
                    valid_foregrounds = wB_fore_list[ii - numA]
                    valid_alphas = wB_alpha_list[ii - numA]
                    sparam = wB_params[ii - numA]
                    tmp_foreground, tmp_alpha = get_single_stroke_on_full_image_B(x_id, y_id, 
                    valid_foregrounds, valid_alphas, sparam, original_img, render_size_x, render_size_y, patch_x, patch_y)

                final_result = tmp_foreground * tmp_alpha + (1 - tmp_alpha) * final_result
                if sample in samples:
                    saveframe = (final_result.numpy().squeeze().transpose([1,2,0])[:,:,::-1] * 255).astype(np.uint8)
                    final_frame_list.append(saveframe)
                    #saveframe = cv2.resize(saveframe, (ow, oh))

                sample += 1
            print("layer %d cost: %.02f" %(layer, time.time() - t0))
            
        
        saveframe = (final_result.numpy().squeeze().transpose([1,2,0])[:,:,::-1] * 255).astype(np.uint8)
        final_frame_list.append(saveframe)
    return final_frame_list