from keras.preprocessing import sequence
from keras.utils import to_categorical


def pad_samples(x_train, x_test, maxlen):

	x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
	x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

	print('x_train shape:', x_train.shape)
	print('x_test shape:', x_test.shape)
	print("")

	return x_train, x_test



def one_hot_labels(y_train, y_test, num_classes):

	y_train = to_categorical(y_train, num_classes=num_classes)
	y_test = to_categorical(y_test, num_classes=num_classes)

	return y_train, y_test


