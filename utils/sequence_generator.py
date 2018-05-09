from keras.utils import Sequence
from keras.utils import to_categorical

import numpy as np

class SequenceGenerator(Sequence):

	def __init__(self, x_set, y_set, batch_size, max_features, num_classes):
		self.x, self.y = x_set, y_set
		self.batch_size = batch_size
		self.max_features = max_features

	def __len__(self):
		return int(np.ceil(len(self.x) / float(self.batch_size)))

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

		# Convert input to one-hot
		return np.array([to_categorical(x, num_classes=self.max_features) for x in batch_x]), np.array(batch_y)


class SequenceGeneratorForPrediction(Sequence):

	def __init__(self, x_set, batch_size, max_features):
		self.x = x_set
		self.batch_size = batch_size
		self.max_features = max_features

	def __len__(self):
		return int(np.ceil(len(self.x) / float(self.batch_size)))

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

		# Convert input to one-hot
		return np.array([to_categorical(x, num_classes=self.max_features) for x in batch_x])
