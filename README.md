# Paint Transformer: Feed Forward Neural Painting with Stroke Prediction

> [[Paper](https://arxiv.org/abs/2108.03798)] [[Paddle Implementation](https://github.com/wzmsltw/PaintTransformer)]


## Overview

This repository contains the official PaddlePaddle implementation of paper:

*Paint Transformer: Feed Forward Neural Painting with Stroke Prediction*,

Songhua Liu\*, Tianwei Lin\*, Dongliang He, Fu Li, Ruifeng Deng, Xin Li, Errui Ding, Hao Wang (* indicates equal contribution)

ICCV 2021 (Oral)

![](picture/picture.png)

## Prerequisites

* Linux or macOS
* Python 3.6+
* PaddlePaddle 2.0+ and other dependencies (numpy, cv2, and other common python libs)

  ```shell
  python -m pip install paddlepaddle-gpu
  ```

## Getting Started

* Clone this repository:

  ```shell
  git clone https://github.com/wzmsltw/PaintTransformer
  cd PaintTransformer
  ```

* Download pretrained model from [Google Drive](https://drive.google.com/file/d/1G0O81qSvGp0kFCgyaQHmPygbVHFi1--q/view?usp=sharing) and move it to inference directory:

  ```shell
  mv [Download Directory]/paint_best.pdparams inference/
  cd inference
  ```

* Inference: 

  ```shell
  python inference.py
  ```

  * Input image path, output path, and etc can be set in the main function.
  * Notably, there is a flag *serial* as one parameter of the main function:
    * If *serial* is True, strokes would be rendered serially. The consumption of video memory will be low but it requires more time. **Serial inference can achieve better rendering quality.**
    * If *serial* is False, strokes would be rendered in parallel. The consumption of video memory will be high but it would be faster.
    * If animated results are required, *serial* must be True.

* Train:

  * You can send email to us for the training codes.

## More Results

Input             |  Animated Output
:-------------------------:|:-------------------------:
![](picture/1.jpg)  |  ![](picture/1.gif)
![](picture/2.jpg)  |  ![](picture/2.gif)
![](picture/3.jpg)  |  ![](picture/3.gif)

## App

* Do not want to run the code? Try an App [_一刻相册_](https://photo.baidu.com/) downloaded from [here](https://photo.baidu.com/union/youa/home)!

<img src="https://github.com/wzmsltw/PaintTransformer/blob/main/picture/yike.jpg" width="500px"/>

## Citation

* If you find ideas or codes useful for your research, please cite:

  ```
  @inproceedings{liu2021paint,
    title={Paint Transformer: Feed Forward Neural Painting with Stroke Prediction},
    author={Liu, Songhua and Lin, Tianwei and He, Dongliang and Li, Fu and Deng, Ruifeng and Li, Xin and Ding, Errui and Wang, Hao},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision},
    year={2021}
  }
  ```

## Contact
For any question, please file an issue or contact
```
Songhua Liu: songhua.liu@smail.nju.edu.cn
Tianwei Lin: lintianwei01@baidu.com
```