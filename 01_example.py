from PIL import Image
import numpy as np
import tensorflow as tf
from transforms_tf import tf_transforms

img = Image.open('doge.jpg')
img_tf = tf.convert_to_tensor(np.transpose(np.asarray(img), (2,0,1)))

transform_img = tf_transforms.Compose([
    tf_transforms.RandomGrayscale(),
    tf_transforms.ToTensor()
])

print(transform_img(img))
