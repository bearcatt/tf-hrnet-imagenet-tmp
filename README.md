# Tensorflow version of High-Resolution Networks for Image Classification

# Instructions

* Follow the instructions in [Tensorflow official Inception repo](https://github.com/tensorflow/models/blob/master/research/inception/README.md#getting-started) to prepare the ImageNet data.
* Run `pip install tensorflow-gpu` to install tensorflow.
* Run `python imagenet_main.py` to train the model from scratch.
* Run `python imagenet_main.py --pretrained_model_checkpoint_path path/to/checkpoint` to continue training from a previous checkpoint.
* Run `python imagenet_main.py --pretrained_model_checkpoint_path path/to/checkpoint --eval_only True` to evaluate a previous checkpoint.

# ImageNet Pre-trained Models

* [HRNet-W32]() (Top-1 Accuracy: 77.0%, Top-5 Accuracy: 93.5%)