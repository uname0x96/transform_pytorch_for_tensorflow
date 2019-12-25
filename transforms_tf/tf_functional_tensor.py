from __future__ import division
import tensorflow as tf


def _is_tensor_a_torch_image(input):
    return len(input.shape) == 3

def vflip(img):
    # type: (Tensor) -> Tensor
    """Vertically flip the given the Image Tensor.

    Args:
        img (Tensor): Image Tensor to be flipped in the form [C, H, W].

    Returns:
        Tensor:  Vertically flipped image Tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return tf.reverse(img, [-2])


def hflip(img):
    # type: (Tensor) -> Tensor
    """Horizontally flip the given the Image Tensor.

    Args:
        img (Tensor): Image Tensor to be flipped in the form [C, H, W].

    Returns:
        Tensor:  Horizontally flipped image Tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return tf.reverse(img, [-1])


def crop(img, top, left, height, width):
    # type: (Tensor, int, int, int, int) -> Tensor
    """Crop the given Image Tensor.

    Args:
        img (Tensor): Image to be cropped in the form [C, H, W]. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.

    Returns:
        Tensor: Cropped image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a tensorflow image.')

    return img[..., top:top + height, left:left + width]


def rgb_to_grayscale(img):
    # type: (Tensor) -> Tensor
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140

    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].

    Returns:
        Tensor: Grayscale image.

    """
    if img.shape[0] != 3:
        raise TypeError('Input Image does not contain 3 Channels')

    # Work around
    ori_dtype = img.dtype
    img = tf.cast(img, dtype=tf.float32)

    return tf.cast((0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]), dtype=ori_dtype)


def adjust_brightness(img, brightness_factor):
    # type: (Tensor, float) -> Tensor
    """Adjust brightness of an RGB image.

    Args:
        img (Tensor): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        Tensor: Brightness adjusted image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a tensorflow image.')

    return _blend(img, tf.zeros_like(img), brightness_factor)

def center_crop(img, output_size):
    # type: (Tensor, BroadcastingList2[int]) -> Tensor
    """Crop the Image Tensor and resize it to desired size.

    Args:
        img (Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions

    Returns:
            Tensor: Cropped image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    _, image_width, image_height = img.shape
    print(image_width, image_height)
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))

    return crop(img, crop_top, crop_left, crop_height, crop_width)


def five_crop(img, size):
    # type: (Tensor, BroadcastingList2[int]) -> List[Tensor]
    """Crop the given Image Tensor into four corners and the central crop.
    .. Note::
        This transform returns a List of Tensors and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.

    Returns:
       List: List (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a tensorflow image.')

    assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    _, image_width, image_height = img.shape
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop(img, 0, 0, crop_width, crop_height)
    tr = crop(img, image_width - crop_width, 0, image_width, crop_height)
    bl = crop(img, 0, image_height - crop_height, crop_width, image_height)
    br = crop(img, image_width - crop_width, image_height - crop_height, image_width, image_height)
    center = center_crop(img, (crop_height, crop_width))

    return [tl, tr, bl, br, center]

def ten_crop(img, size, vertical_flip=False):
    # type: (Tensor, BroadcastingList2[int], bool) -> List[Tensor]
    """Crop the given Image Tensor into four corners and the central crop plus the
        flipped version of these (horizontal flipping is used by default).
    .. Note::
        This transform returns a List of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
       vertical_flip (bool): Use vertical flipping instead of horizontal

    Returns:
       List: List (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
                Corresponding top left, top right, bottom left, bottom right and center crop
                and same for the flipped image's tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)

    return first_five + second_five

def _blend(img1, img2, ratio):
    # type: (Tensor, Tensor, float) -> Tensor
    bound = 1 if img1.dtype in [tf.float16, tf.float32, tf.float64] else 255

    # Workaround, TF seem like working not good with other dtype except Float**
    ori_dtype = img1.dtype
    img_float = tf.cast(ratio * img1 + (1 - ratio) * img2, dtype=tf.float32)
    img_float = tf.keras.backend.clip(img_float, 0, bound)
    return tf.cast(img_float, dtype=ori_dtype)


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    img = Image.open('doge.jpg')
    img_tf = tf.convert_to_tensor(np.asarray(img))
    # print(img_tf[:1])
    img_tf = tf.transpose(img_tf, perm=[2,0,1])
    print(five_crop(img_tf,(23,23)))
