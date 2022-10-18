
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from PIL import Image
#from scipy.misc import imread, imresize, imsave
#from scipy.misc import imresize
import imageio
from pylab import *

import tensorflow as tf
import scipy.stats as st

from nets import inception_v3, inception_v4, inception_resnet_v2, resnet_v2

slim = tf.contrib.slim

FLAGS = tf.flags.FLAGS

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)



tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')


tf.flags.DEFINE_string(
    'checkpoint_path', ' ', 'Path to checkpoint for inception network.')




tf.flags.DEFINE_string(
   'input_dir', ' ', 'Input directory with images.')

tf.flags.DEFINE_string(
   'output_dir', ' ', 'Output directory with images.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'image_resize', 330, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 5, 'How many images process at one time.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'num_iter', 10, 'Number of iterations.')

tf.flags.DEFINE_float(
    'prob', 0.5, 'probability of using diverse inputs.')

 
tf.flags.DEFINE_float(
    'momentum', 1, 'Momentum.')

tf.flags.DEFINE_string(
    'GPU_ID', '0', 'which GPU to use.')



print("print all settings\n")
print(FLAGS.master)
print(FLAGS.__dict__)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.GPU_ID

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


kernel = gkern(7, 3).astype(np.float32)
stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
stack_kernel = np.expand_dims(stack_kernel, 3)


def load_images(input_dir, output_dir, batch_shape):
  """Read png images from input directory in batches.
  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png'))[:1000]:
    temp_name = str.split(filepath, '/')
    output_name = output_dir + '/'+ temp_name[-1]
    # check if the file exist
    if os.path.isfile(output_name) == False:
 
      with tf.io.gfile.GFile(filepath, "rb") as f:
        image = imageio.imread(f, pilmode='RGB').astype(np.float) / 255.0
 
      images[idx, :, :, :] = image * 2.0 - 1.0
      filenames.append(os.path.basename(filepath))
      idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0
  if idx > 0:
    yield filenames, images


def save_images(images, filenames, output_dir):
  """Saves images to the output directory.
  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
     # imageio.imsave(f, (images[i, :, :, :] + 1.0) * 0.5 * 255, format='png')
      imageio.imsave(f, Image.fromarray(uint8((images[i, :, :, :] + 1.0) * 0.5 * 255)), format='png')



beta1 = 0.99
beta2 = 0.999
num_iter1 = FLAGS.num_iter
weight=0
t = np.arange(1,num_iter1+0.1,1)
y1 = np.sqrt(1 - beta2**t) / (1 - beta1**t)

for x1 in y1:
    weight+=x1




def graph(x, y, i, x_max, x_min, grad):
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  eps_iter = 2.0 / 255.0
  num_iter = FLAGS.num_iter
  alpha = eps / num_iter
  num_classes = 1001
  momentum = FLAGS.momentum
  
  x_nes = x  #MI

#====================================Res-101=========================================   
  
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits1, end_points1 = resnet_v2.resnet_v2_101(
      input_diversity_1(x_nes), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)  

  pred = tf.argmax(end_points1['predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits1)
  noise = tf.gradients(cross_entropy, x)[0]
 
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits2, end_points2 =  resnet_v2.resnet_v2_101(
       1 / 2 *input_diversity_2(x_nes), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)  

  pred = tf.argmax(end_points2['predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits2)
  noise2 = tf.gradients(cross_entropy, x)[0]
  
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits3, end_points3 = resnet_v2.resnet_v2_101(
       1 / 4 *input_diversity_3(x_nes), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)  

  pred = tf.argmax(end_points3['predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits3)
  noise3 = tf.gradients(cross_entropy, x)[0]
  
  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits4, end_points4 = resnet_v2.resnet_v2_101(
       1 / 8 *input_diversity_4(x_nes), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)  

  pred = tf.argmax(end_points4['predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits4)
  noise4 = tf.gradients(cross_entropy, x)[0]

  with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits5, end_points5 = resnet_v2.resnet_v2_101(
       1 / 16 *input_diversity_5(x_nes), num_classes=num_classes, is_training=False, reuse=tf.AUTO_REUSE)  

  pred = tf.argmax(end_points5['predictions'], 1)
  first_round = tf.cast(tf.equal(i, 0), tf.int64)
  y = first_round * pred + (1 - first_round) * y
  one_hot = tf.one_hot(y, num_classes)
  cross_entropy = tf.losses.softmax_cross_entropy(one_hot, logits5)
  noise5 = tf.gradients(cross_entropy, x)[0] 


  noise = noise+noise2+noise3+noise4+noise5
  #noise = tf.nn.depthwise_conv2d(noise, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')  #TI    
  noise = noise / tf.reduce_mean(tf.abs(noise), [1,2,3], keep_dims=True)
  
  # accumulate the gradient 
  noise = momentum * grad + noise

  #x = x + eps_iter * tf.sign(noise)
  x = x + alpha * tf.sign(noise)
  x = tf.clip_by_value(x, x_min, x_max)
  i = tf.add(i, 1)
  return x, y, i, x_max, x_min, noise


def stop(x, y, i, x_max, x_min, grad):
  #num_iter = int(min(FLAGS.max_epsilon+4, 1.25*FLAGS.max_epsilon))
  num_iter = FLAGS.num_iter
  return tf.less(i, num_iter)


 
#==============================TIM================================================
def input_diversity_1(image):   
    rnd1 = tf.random_uniform((), 0, 11, dtype=tf.int32)
    rnd2 = tf.random_uniform((), 0, 11, dtype=tf.int32)
    rnd3 = tf.random_uniform((), 0, 11, dtype=tf.int32)
    rnd4 = tf.random_uniform((), 0, 11, dtype=tf.int32)
    x = tf.image.pad_to_bounding_box(image, rnd1, rnd2, 299+rnd1+rnd3, 299+rnd2+rnd4)
    output_tensor = tf.image.crop_to_bounding_box(x, rnd3, rnd4, 299, 299)
    img = tf.cond(tf.random.uniform([], 0, 1) < 0.5, lambda: output_tensor, lambda: image)
    return img

#==========================Resize_small REIM==========================================
def input_diversity_2(input_tensor):
  rnd = tf.random_uniform((), 270, FLAGS.image_width, dtype=tf.int32)
  rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  h_rem = FLAGS.image_height - rnd
  w_rem = FLAGS.image_width - rnd
  pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
  pad_bottom = h_rem - pad_top
  pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
  pad_right = w_rem - pad_left
  padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
  padded.set_shape((input_tensor.shape[0], FLAGS.image_height, FLAGS.image_width, 3))
  return tf.cond(tf.random_uniform(shape=[1])[0] < 0.5, lambda: padded, lambda: input_tensor)

#================================CIM==============================================
def input_diversity_3(input_tensor):
    rnd = tf.random_uniform((), 279, 299, dtype=tf.int32)
    rescaled = tf.image.resize_image_with_crop_or_pad(input_tensor, rnd, rnd)
    h_rem = 299 - rnd
    w_rem = 299 - rnd
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
    padded.set_shape((input_tensor.shape[0], 299, 299, 3))
    return tf.cond(tf.random.uniform([], 0, 1) < 0.5, lambda: padded, lambda: input_tensor)

#=========================SIM===================================================
def input_diversity_4(input_tensor):
    k = tf.random_uniform((), 1, 10, dtype=tf.float32)
    return tf.cond(tf.random.uniform([], 0, 1) < 0.5, lambda: input_tensor/k, lambda: input_tensor)

#========================Rotation RIM=================================================
def input_diversity_5(image):
    output_tensor = tf.contrib.image.rotate(image, angles = tf.random.uniform([], -10, 10) /180 * math.pi)
    return tf.cond(tf.random.uniform([], 0, 1) < 0.5, lambda: output_tensor, lambda: image)

#========================翻转=================================================
def input_diversity_6(image):
    
  #img1 = tf.image.flip_up_down(image)
  img2 = tf.image.flip_left_right(image)
  #output_tensor = tf.cond(tf.random.uniform([], 0, 1) < 0.5, lambda: img1, lambda: img2)
  img = tf.cond(tf.random.uniform([], 0, 1) < 0.5, lambda: img2, lambda: image)
  return img


#========================亮度=================================================
def input_diversity_8(image):
  
  imaged = tf.cast(image,tf.float32) #这句必须加上
  img = tf.image.random_brightness(imaged,max_delta=5)#随机调整亮度函数  
  img1 = tf.clip_by_value(img, 0, 299)
  img = tf.cond(tf.random.uniform([], 0, 1) < 0.5, lambda: img1, lambda: image)
  return img

    rnd = tf.random_uniform((), 1, 16, dtype=tf.float32)
    return tf.cond(tf.random_uniform(shape=[1])[0] < 0.5, lambda: input_tensor/rnd, lambda: input_tensor)



def main(_):
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    x_max = tf.clip_by_value(x_input + eps, -1.0, 1.0)
    x_min = tf.clip_by_value(x_input - eps, -1.0, 1.0)

    y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
    i = tf.constant(0,float)
    grad = tf.zeros(shape=batch_shape)
    x_adv, pre, _, _, _, _ = tf.while_loop(stop, graph, [x_input, y, i, x_max, x_min, grad])
    # Run computation
    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, FLAGS.checkpoint_path)
      for filenames, images in load_images(FLAGS.input_dir, FLAGS.output_dir, batch_shape):
        adv_images = sess.run(x_adv, feed_dict={x_input: images})
        save_images(adv_images, filenames, FLAGS.output_dir)



if __name__ == '__main__':
  tf.app.run()
