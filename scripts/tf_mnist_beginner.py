# Import the MNIST data set.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Define some initial variables.
import tensorflow as tf
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Implement the model using the built-in softmax function.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define the operation to compute cross-entropy.
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# Use a built-in optimization algorithm to define how to proceed with training.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Create a session and initialize the variables.
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Run the training algorithm for 1000 iterations.
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Calculate the accuracy of the trained model.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))