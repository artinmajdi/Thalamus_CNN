import tensorflow as tf
from tf_unet import unet
sess =  tf.Session()

a = tf.constant('hey there')

print sess.run(a)
