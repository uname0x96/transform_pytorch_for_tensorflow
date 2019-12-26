# @Author: cong235
# @Date:   2019-12-25T16:34:38+07:00
# @Project: transform_pytorch_for_tensorflow
# @Filename: 01_example.py
# @Last modified by:   cong235
# @Last modified time: 2019-12-26T10:04:04+07:00
# @License: MIT

from PIL import Image
import numpy as np
import tensorflow as tf
from transforms_tf import tf_transforms

import torch
import torchvision.transforms as pt_transforms

#You should activate the eager mode if your TF "more than 0 but less than 2"
if (tf.__version__.split('.')[0] == 1):
    tf.enable_eager_execution()

img = Image.open('doge.jpg')
img_tf = tf.convert_to_tensor(np.transpose(np.asarray(img), (2,0,1)))
img_pt = torch.from_numpy(np.transpose(np.asarray(img), (2,0,1)))

def my_transform_1(x):
    return x

def _random_colour_space(x):
    output = x.convert("HSV")
    return output
pt_transforms.Lambda(lambda x: _random_colour_space(x))

# Transform Torch
transform_pt = pt_transforms.Compose([
    pt_transforms.ToTensor(),
    pt_transforms.ToPILImage(),
    pt_transforms.ToTensor(),
    pt_transforms.Normalize([0.5,0.6,0.7],[0.5,0.6,0.7]),
    pt_transforms.ToPILImage(),
    pt_transforms.Resize((20,20)),
    pt_transforms.Scale((50,50)),
    pt_transforms.CenterCrop((23,23)),
    pt_transforms.Pad((0,1,0,1)),
    pt_transforms.Lambda(lambda x: my_transform_1(x)),
    pt_transforms.RandomApply(p=0.85,transforms=[pt_transforms.Resize((20,20))]),
    # pt_transforms.RandomAffine(degrees=[-45, 45],translate=[0.15, 0.15],scale=[1.0, 1.2])
])
result_pt = transform_pt(img)
# print(result_pt)

# Transform TF
transform_tf = tf_transforms.Compose([
    tf_transforms.ToTensor(),
    tf_transforms.ToPILImage(),
    tf_transforms.ToTensor(),
    tf_transforms.Normalize([0.5,0.6,0.7],[0.5,0.6,0.7]),
    tf_transforms.ToPILImage(),
    tf_transforms.Resize((20,20)),
    tf_transforms.Scale((50,50)),
    tf_transforms.CenterCrop((23,23)),
    tf_transforms.Pad((0,1,0,1)),
    tf_transforms.Lambda(lambda x: my_transform_1(x)),
    tf_transforms.RandomApply(p=0.85,transforms=[tf_transforms.Resize((20,20))]),
    # tf_transforms.RandomAffine(degrees=[-45, 45],translate=[0.15, 0.15],scale=[1.0, 1.2])
])
result_tf = transform_tf(img)
# print(result_tf)

print('Check value from pytorch and TF')
np.testing.assert_allclose(result_tf, result_pt, rtol=1e-03, atol=1e-05)
print('Passed')
