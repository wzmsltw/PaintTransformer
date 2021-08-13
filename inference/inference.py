import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import network
import os
import math
import render_utils
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import cv2
import render_parallel
import render_serial

def main(input_path, model_path, output_dir, need_animation=False, resize_h=None, resize_w=None, serial=False):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    input_name = os.path.basename(input_path)
    output_path = os.path.join(output_dir, input_name)
    frame_dir = None
    if need_animation:
        if not serial:
            print('It must be under serial mode if animation results are required, so serial flag is set to True!')
            serial = True
        frame_dir = os.path.join(output_dir, input_name[:input_name.find('.')])
        if not os.path.exists(frame_dir):
            os.mkdir(frame_dir)
    stroke_num = 8

    #* ----- load model ----- *#
    paddle.set_device('gpu')
    net_g = network.Painter(5, stroke_num, 256, 8, 3, 3)
    net_g.set_state_dict(paddle.load(model_path))
    net_g.eval()
    for param in net_g.parameters():
        param.stop_gradient = True

    #* ----- load brush ----- *#
    brush_large_vertical = render_utils.read_img('brush/brush_large_vertical.png', 'L')
    brush_large_horizontal = render_utils.read_img('brush/brush_large_horizontal.png', 'L')
    meta_brushes = paddle.concat([brush_large_vertical, brush_large_horizontal], axis=0)

    import time
    t0 = time.time()
    
    original_img = render_utils.read_img(input_path, 'RGB', resize_h, resize_w)
    if serial:
        final_result_list = render_serial.render_serial(original_img, net_g, meta_brushes)
        if need_animation:

            print("total frame:", len(final_result_list))
            for idx, frame in enumerate(final_result_list):
                cv2.imwrite(os.path.join(frame_dir, '%03d.png' %idx), frame)
        else:
            cv2.imwrite(output_path, final_result_list[-1])
    else:
        final_result = render_parallel.render_parallel(original_img, net_g, meta_brushes)
        cv2.imwrite(output_path, final_result)
    
    print("total infer time:", time.time() - t0)

if __name__ == '__main__':
    
    main(input_path='input/chicago.jpg',
         model_path='paint_best.pdparams',
         output_dir='output/',
         need_animation=True,  # whether need intermediate results for animation.
         resize_h=512,         # resize original input to this size. None means do not resize.
         resize_w=512,         # resize original input to this size. None means do not resize.
         serial=True)          # if need animation, serial must be True.