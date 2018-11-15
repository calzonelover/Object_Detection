import tensorflow as tf

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, name='input_x')
    y = tf.placeholder(tf.float32, name='input_y')
z = tf.add(x,y, name='output')

sess = tf.Session()

a = sess.run(z, feed_dict={x: [3 , 3], y: [4.5, 4.5]})

writer = tf.summary.FileWriter("repo", sess.graph)

print(a)