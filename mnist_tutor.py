import tensorflow as tf
import numpy as np

# Define the model we'll use
class Mnist:
  def __init__(self):
    # Create placeholders
    #   (x, y)
    #   x: [BATCH_SIZE, 28, 28] -> think of this as a list of [28, 28] arrays.
    #   y: [BATCH_SIZE]         -> think of this as a list of integers (correct class labels).
    self.x = tf.placeholder(shape=[None, 28, 28], dtype=tf.float32)  # assuming a float input in [0, 1]
    self.y = tf.placeholder(shape=[None], dtype=tf.int64)

    # Define a fully-connected network with two hidden layers
    #   (input=784) --> 20 --> 20 --> 10
    out = tf.layers.flatten(self.x)  # [BS, 28, 28] --> [BS, 28*28] = [BS, 784]
    out = tf.layers.dense(out, units=20, activation=tf.nn.relu)  # [BS, 784] --> [BS, 20]
    out = tf.layers.dense(out, units=20, activation=tf.nn.relu)  # [BS, 20] --> [BS, 20]
    out = tf.layers.dense(out, units=10, activation=None)  # [BS, 20] --> [BS, 10]

    # Compute loss and accuracy
    self.preds = tf.argmax(out, axis=1)
    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.preds, self.y), tf.float32))
    self.loss = tf.losses.sparse_softmax_cross_entropy(self.y, out)

if __name__ == '__main__':
  # Load data (don't worry too much about this part)
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train / 255.0
  x_test = x_test / 255.0
  print('Loaded {} training, {} test examples'.format(len(x_train), len(x_test)))

  # Create the model and optimizer
  model = Mnist()
  optimizer = tf.train.AdamOptimizer(0.01)
  train_op = optimizer.minimize(model.loss)
  init_op = tf.global_variables_initializer()

  # For Tensorboard logging
  tf.summary.scalar('train_loss', model.loss)
  tf.summary.scalar('train_acc', model.accuracy)
  summary_op = tf.summary.merge_all()

  with tf.Session() as sess:
    # Initializes variables in the graph
    sess.run(init_op)

    # Create a summary writer
    writer = tf.summary.FileWriter('./logs/model', graph=tf.get_default_graph())

    # Main training loop
    for step in range(1000):
      # Create a mini-batch
      indices = np.random.permutation(len(x_train))[:64]
      x_batch = x_train[indices]
      y_batch = y_train[indices]

      _, loss, accuracy, summary = sess.run(
          [train_op, model.loss, model.accuracy, summary_op],
          feed_dict={model.x: x_batch, model.y: y_batch})

      if step % 5 == 0:
        writer.add_summary(summary, step)

      if step % 50 == 0:
        print('Step: {}, loss: {}, accuracy: {}'.format(step, loss, accuracy))

    print('Training complete!')

    # Evaluate performance on the test set
    test_loss, test_accuracy = sess.run(
        [model.loss, model.accuracy],
        feed_dict={model.x: x_test, model.y: y_test})

    print('Test loss: {}, accuracy: {}'.format(test_loss, test_accuracy))
