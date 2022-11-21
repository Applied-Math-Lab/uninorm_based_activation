from typing import Tuple
import tensorflow as tf
import numpy as np
import teddy as td
import nn


def one_hot(bef: np.ndarray) -> np.ndarray:
	vals = bef.max() + 1
	aft = np.zeros((bef.size, vals), dtype = np.float64)
	aft[np.arange(bef.size), bef] = 1.0
	return aft


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()
	train_images = np.float64(mnist_train[0]).reshape((60000, 28 * 28))
	train_images *= (1.0 / 256.0)
	train_labels = one_hot(np.asarray(mnist_train[1], dtype=np.int32))
	test_images = np.float64(mnist_test[0]).reshape((10000, 28 * 28))
	test_images *= (1.0 / 256.0)
	test_labels = one_hot(np.asarray(mnist_test[1], dtype=np.int32))
	return train_images, train_labels, test_images, test_labels


class Model:
	def __init__(self) -> None:
		self.h0 = nn.LayerLinear(784, 300)
		self.h2 = nn.LayerLinear(300, 10)
		self.optimizer = tf.keras.optimizers.SGD(learning_rate = 0.3)
		self.params = self.h0.params + self.h2.params

	def act(self, incoming: tf.Tensor) -> tf.Tensor:
		y = self.h0.act(incoming)
		y = tf.nn.relu(y)
		y = self.h2.act(y)
		y = tf.nn.tanh(y)
		return y

	def input_gradient(self, x: tf.Tensor) -> tf.Tensor:
		with tf.GradientTape() as g:
			g.watch(x)
			y = self.act(x)
			grads = g.gradient(y, x)
		return grads

	def cost(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
		return tf.reduce_mean(tf.reduce_sum(tf.square(y - self.act(x)), axis = 1), axis = 0)

	def refine(self, x: tf.Tensor, y: tf.Tensor) -> None:
		self.optimizer.minimize(lambda: self.cost(x, y), self.params)

	def misclassification(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
		err = tf.not_equal(tf.argmax(y, axis=1), tf.argmax(self.act(x), axis = 1))
		return tf.reduce_sum(tf.cast(err, tf.int32))


# Optimisation variables
epochs = 100
batch_size = 128
model = Model()

# Load some data
train_images, train_labels, test_images, test_labels = load_mnist_data()

# start the session
total_batch = int(len(train_labels) / batch_size)
all_indexes = np.arange(train_images.shape[0])

for epoch in range(epochs):

	# Measure progress
	mis = model.misclassification(test_images, test_labels)
	print("mis = " + str(mis))

	# Do some training
	np.random.shuffle(all_indexes)
	batch_start = 0
	while batch_start + batch_size <= train_images.shape[0]:

		# Make a batch
		batch_indexes = all_indexes[batch_start:batch_start + batch_size]
		batch_images = train_images[batch_indexes]
		batch_labels = train_labels[batch_indexes]

		# Refine the model
		model.refine(batch_images, batch_labels)

		# Advance to the next batch
		batch_start += batch_size

# Measure progress
mis = model.misclassification(test_images, test_labels)
print("mis = " + str(mis))
