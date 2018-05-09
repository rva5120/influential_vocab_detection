from keras.models import Model
from keras.layers import Input, Dense, merge, Dropout, Activation
from keras.layers import Embedding, Lambda
from keras.layers import Conv1D, GlobalAveragePooling1D

from sklearn.metrics import confusion_matrix
import numpy as np


def get_cnn_classifier(maxlen, max_features, embedding_dims, filters, kernel_size, hidden_dims, num_classes, one_hot_input=True):

	# Input Layer
	# -----------
	# Placeholder for input that contains word indices
	if one_hot_input:
		x = Input(shape=(maxlen,max_features), dtype='float32', name="input")
	else:
		x = Input(shape=(maxlen,), dtype='int32', name="input")


	# Embedding Layer w/Dropout
	# -------------------------
	# Transform: word indices --> word embeddings
	# (batch_size, 400) --> (batch_size, 400, 50)
	if one_hot_input:
		e = Dense(embedding_dims)(x)
	else:
		e = Embedding(max_features, embedding_dims, name="Embedding_Layer")(x)
	e_drop = Dropout(0.2, name="Embedding_Layer_Dropout")(e)
	print(e_drop.shape)
	print(e.shape)

	# Convolutional Layer 
	# -------------------
	# Transform: word embeddings --> 3-gram features obtained from applying a filter
	# (batch_size, 400) --> (batch_size, 398, 250)
	c = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1, name="Conv_1D_3-gram_Layer")(e_drop)


	# Global Max Pooling 
	# ------------------
	# Transform: 3-gram features --> max 3-gram features for each dimension of the filter
	# (batch_size, 398, 250) --> (batch_size, 250)
	max_c = GlobalAveragePooling1D(name="Average_Pooling_Layer")(c)


	# Dense Layer w/Dropout and ReLu Activation
	# -----------------------------------------
	# Fully connected NN layer to transform: max 3-gram features --> new feature representation
	# (batch_size, 250) --> (batch_size, 250)
	h = Dense(hidden_dims, name="Fully_Connected_Hidden_Layer")(max_c)
	h_drop = Dropout(0.2, name="Hidden_Layer_Dropout")(h)
	z = Activation('relu', name="Hidden_Layer_Activation")(h_drop)


	# Output layer w/Sigmoid Activation 
	# ---------------------------------
	# Transforms: feature representation --> probability of being in class 1 (positive sentiment)
	# (batch_size, 250) --> (batch_size, 1)
	o = Dense(num_classes, name="output")(z)
	preds = Activation('softmax', name="output_activation")(o)

	# Define model from graph
	model = Model(x, preds)

	return model



def save_model_performance(f, model_history):

	# Write Training and Validation Accuracies to a file
	f.write("----- Model Performance ------\n")
	f.write("Training Accuracy: ")
	f.write(str(100*model_history.history['acc'][-1]))
	f.write("\n")
	f.write("Validation Accuracy: ")
	f.write(str(100*model_history.history['val_acc'][-1]))

	return



def get_confusion_matrix(model, predict_generator, y_test):

	# Get predictions for test dataset
	y_pred = model.predict_generator(predict_generator, verbose=1)
	y_pred = [np.argmax(row) for row in y_pred]

	# Convert true predictions to indices
	y_true = [np.argmax(row) for row in y_test]

	#print(y_pred[0])
	#print(y_true[0])

	# Compute Confusion Matrix
	conf_matrix = confusion_matrix(y_true, y_pred)

	return conf_matrix
