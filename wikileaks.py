"""

Analysis of influential words in the Wikileaks Cables Dataset
-------------------------------------------------------------
The is a modified version of the keras tutorial on sentiment classification
of the IMDB movie review dataset. We use the Wikileaks dataset instead, and
perform a 3-way/2-way classification of unclassified, confidential and secret 
(if 3-way) documents.

This tutorial shows how to extract the top words that make a document belong
to a particular class. We achieve this by using adversarial machine learning
tecniques that consist on computing the gradients of the loss of the model
with respect to the input. The loss is computed as:
	loss = pred_class - target_class
where the target_class is a class in the dataset that is most different
compared to the pred_class. For example, if the pred_class of a document
is "unclassified", the target class will be "secret". This will give us
a high absolute value for the gradients of the words that are the least
secret. We say that these words are the most unclassified.

We record the top 3 words with the highest gradients, and get an overall
distribution of the most influential words for each class.


By Raquel Alvarez

"""
import pickle

import sys
import argparse

import numpy as np

from tensorflow import Session
from keras.backend import set_session

from keras.models import load_model
from keras.utils import to_categorical

#from sklearn.metrics import confusion_matrix

from utils.sequence_generator import SequenceGenerator, SequenceGeneratorForPrediction
from utils.colors import colors
from utils import data_preprocessing, models, influential_vocab

from tqdm import tqdm





#################################
# Arguments 			#
#################################
# Setup argument parser
parser = argparse.ArgumentParser(sys.argv[0], description='Extract the most influential words per class of a dataset.')
parser.add_argument('--num-classes', type=int, default=2, help='Number of classes in the dataset')
parser.add_argument('--pre-trained', type=bool, default=False, help='True/False use pre-trained model. Default is True.')
parser.add_argument('--use-class-weights', type=bool, default=False, help='True/False use class weights. Default is False.')
args = parser.parse_args()

# Parse arguments
num_classes = args.num_classes
pre_trained = args.pre_trained
use_class_weights = args.use_class_weights





#################################
# Keras Setup			#
#################################
sess = Session()		# Allows us to be able to pass the session to other libs
set_session(sess)		# Keras uses the session we pass here





#################################
# Wikileaks Data Loading	#
#################################
# Load Data
print("")
print(colors.BOLD + "Loading Data..." + colors.END)

# Open dataset files
x_file = open("dataset/wikileaks/"+str(num_classes)+"-way/dataset_vars/x_dataset.pkl", "rb")
y_file = open("dataset/wikileaks/"+str(num_classes)+"-way/dataset_vars/y_dataset.pkl", "rb")
word_index_file = open("dataset/wikileaks/"+str(num_classes)+"-way/dataset_vars/word_index_dict.pkl", "rb")
index_word_file = open("dataset/wikileaks/"+str(num_classes)+"-way/dataset_vars/index_word_dict.pkl", "rb")
max_doc_len_file = open("dataset/wikileaks/"+str(num_classes)+"-way/dataset_vars/max_len.pkl", "rb")

# Load precomputed structures
X = pickle.load(x_file)
y = pickle.load(y_file)
word_index_dict = pickle.load(word_index_file)
index_word_dict = pickle.load(index_word_file)
max_doc_len = pickle.load(max_doc_len_file)

# Convert labels to indices
#	unclassified 	0
#	confidential 	1
#	secret		2
idx = 0
for label in y:
	if label == "unclassified":
		y[idx] = 0
	elif label == "confidential":
		y[idx] = 1
	else:
		y[idx] = 2
		#print("Wrong in 2-way data...")
	idx += 1

# Convert lists to numpy arrays
X = np.asarray(X)
y = np.asarray(y)

# Shuffle dataset
np.random.seed(113)
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Separate dataset
mid = int(len(X) / 2)
x_train = X[0:mid]
y_train = y[0:mid]
x_test = X[mid+1:]
y_test = y[mid+1:]

# Gather stats for training data
uncl_cnt = 0
conf_cnt = 0
secr_cnt = 0
for label in y_train:
	if label == 0:
		uncl_cnt += 1
	elif label == 1:
		conf_cnt += 1
	else:
		#print("Error 2")
		secr_cnt += 1

print("--- Training Stats ---")
print("Unclassified: "+str(uncl_cnt))
print("Confidential: "+str(conf_cnt))
print("Secret: "+str(secr_cnt))

# Close files
x_file.close()
y_file.close()
word_index_file.close()
index_word_file.close()
max_doc_len_file.close()

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print("")





#################################
# Parameters			#
#################################
# Dataset Parameters
max_features = len(word_index_dict)	# max number of unique words recognized from the dataset
maxlen = max_doc_len			# max length of a document
num_classes = num_classes		# number of classes

# Training Parameters
batch_size = 32		# run groups of 32 samples every training iteration
epochs = 5		# number of times to run over entire dataset during training

# Model Parameters
embedding_dims = 50			# number of dimensions of the word embedding vectors
filters = 250				# number of filters to apply on the conv. layer
kernel_size = 3				# window size of 3 words (3-grams)
hidden_dims = 250			# dense layer of 250 dimensions
model_exists = pre_trained		# set this to True if a pre-trained model exists
use_class_weights = use_class_weights	# train model using class weights





#################################
# Unbalanced Dataset Handling	#
#################################
# Setup weights so the model "pays more attention" to
# the under-represented class "Secret"
class_weight = dict()
class_weight[0] = 1	# Unclassified
class_weight[1] = 2	# Confidential
class_weight[2] = 4	# Secret





#################################
# Data Preprocessing		#
#################################
# Pad documents accordingly
print(colors.BOLD + "Padding sequences..." + colors.END)
x_train, x_test = data_preprocessing.pad_samples(x_train, x_test, maxlen)

# Save index-like (0,1,2) y vector for highest gradient computation
y_train_idx = np.copy(y_train)
y_test_idx = np.copy(y_test)

# Convert labels to one-hot vectors
print(colors.BOLD + "Converting word indices to one-hot vectors...\n" + colors.END)
y_train, y_test = data_preprocessing.one_hot_labels(y_train, y_test, num_classes)

# Setup data generator to feed to the model
print(colors.BOLD + "Setting up Data Generators...\n" + colors.END)
train_generator = SequenceGenerator(x_train, y_train, batch_size, max_features, num_classes)
test_generator = SequenceGenerator(x_test, y_test, batch_size, max_features, num_classes)
predict_generator = SequenceGeneratorForPrediction(x_test, batch_size, max_features)





#################################
# Classification Model		#
#################################
model_file = 'pre_trained_models/wikileaks_model_'+str(num_classes)+'-way.h5'

if model_exists:

	# Load model
	print(colors.BOLD + colors.GREEN + 'Loading model...' + colors.END)

	model = load_model(model_file)

	print(model.summary())

else:

	# Build model
	print(colors.BOLD + colors.GREEN + 'Building model...' + colors.END)
	model = models.get_cnn_classifier(maxlen, max_features, embedding_dims, filters, kernel_size, hidden_dims, num_classes, one_hot_input=True)
	print(model.summary())

	# Generate model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	# Train and test model
	#model_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
	if use_class_weights:
		model_history = model.fit_generator(generator=train_generator, validation_data=test_generator, epochs=epochs, class_weight=class_weight)
	else:
		model_history = model.fit_generator(generator=train_generator, validation_data=test_generator, epochs=epochs)

	# Save model for future use
	model.save(model_file)

	# Print model accuracy
	print("")
	print(colors.BOLD + colors.UNDERLINE  + "Model Performance" + colors.END)
	print(colors.WHITE + colors.BLUE_BG + "Training Accuracy: " + str(100*model_history.history['acc'][-1]) + colors.END)
	print(colors.WHITE + colors.BLUE_BG + "Validation Accuracy: " + str(100*model_history.history['val_acc'][-1]) + colors.END)

	# Write model performance to a file
	#f = open("model_history.log", 'w')
	#models.save_model_performance(f, model_history)
	#f.close()





#################################
# Confusion Matrix		#
#################################
print("")
print(colors.BOLD + colors.UNDERLINE + "Confusion Matrix" + colors.END)
#conf_matrix = models.get_confusion_matrix(model, predict_generator, y_test)
#print(conf_matrix)
print("")





###############################################
# Most Influential Words for the Test Dataset #
###############################################
print("")
print(colors.BOLD + colors.UNDERLINE  + "Most Influential Words - Dataset (word: # of times it was the top 3 word on 1 of 100 random documents)" + colors.END)
print("")

top3_counter_unclassified, top3_counter_confidential, top3_counter_secret = influential_vocab.get_most_influential_words_for_dataset(x_train, y_train_idx, max_features, maxlen, num_classes, model, index_word_dict)


print(colors.GREEN_BG + colors.BLACK + "Unclassified Class Words: " + colors.END + colors.GREEN)
print(top3_counter_unclassified.most_common())
print(colors.END)

print(colors.YELLOW_BG + colors.BLACK + "Confidential Class Words: " + colors.END + colors.YELLOW)
print(top3_counter_confidential.most_common())
print(colors.END)

print(colors.RED_BG + colors.WHITE + "Secret Class Words: " + colors.END + colors.RED)
print(top3_counter_secret.most_common())
print(colors.END)





##########################################
# Top 3 Influential Words for a Document #
##########################################
print("")
print(colors.BOLD + colors.UNDERLINE  + "Most Influential Words - Document" + colors.END)

document = x_test[18]
classification = y_test_idx[18]		# can be replaced with model.predict(document) if the document's unlabeled

top3_document = influential_vocab.get_most_influential_words_for_document(document, classification, max_features, maxlen, num_classes, model, index_word_dict)

# Print results
if classification == 0:
	print(colors.GREEN)
elif classification == 1:
	print(colors.YELLOW)
else:
	print(colors.RED)
print(top3_document)
print(colors.END)



# Close TF session
sess.close()
