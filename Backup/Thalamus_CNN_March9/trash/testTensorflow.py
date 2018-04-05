import tensorflow as tf
from tf_unet import unet

hello = tf.constant('Hello There nasty')

sess = tf.Session()
print sess.run(hello)
