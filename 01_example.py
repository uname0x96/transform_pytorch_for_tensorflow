# @Author: cong235
# @Date:   2019-12-25T16:34:38+07:00
# @Project: transform_pytorch_for_tensorflow
# @Filename: 01_example.py
# @Last modified by:   cong235
# @Last modified time: 2019-12-25T17:40:49+07:00
# @License: MIT

from PIL import Image
import numpy as np
import tensorflow as tf
from transforms_tf import tf_transforms

if tf.version

img = Image.open('doge.jpg')
img_tf = tf.convert_to_tensor(np.transpose(np.asarray(img), (2,0,1)))

transform_img = tf_transforms.Compose([
    tf_transforms.RandomGrayscale(),
    tf_transforms.ToTensor()
])

print(transform_img(img))
