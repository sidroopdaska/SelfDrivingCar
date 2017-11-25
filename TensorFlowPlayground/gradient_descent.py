# import tensorflow as tf
# import numpy as np
#
# x = tf.Variable(tf.cast(np.random.randint(0, 10000), tf.float32), trainable=True)
# y = tf.add(tf.multiply(x, x), 5)
#
# epochs = 100
# learn_rate = tf.placeholder(tf.float32)
#
# optimiser = tf.train.Gr(learn_rate).minimize(y)
#
# with tf.Session() as session:
#     session.run(tf.global_variables_initializer())
#     for epoch_i in range(epochs):
#         loss = session.run(y)
#         x_val = session.run(x)
#         session.run(optimiser, feed_dict={learn_rate: 0.1})
#         print(f"Epoch:{epoch_i}  Loss:{loss:.4f}  X:{x_val:.2f}")