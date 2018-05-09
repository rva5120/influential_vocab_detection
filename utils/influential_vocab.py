from collections import Counter
from six.moves import xrange
import math

import numpy as np

import keras.backend as K
from keras.utils import to_categorical


def compute_gradients(model, target, adversarial_document):

	# Get the necessary placeholder variables
	model_input = model.input			# input document
	model_output = model.layers[9].output		# output probability
	model_embeddings = model.layers[1].output	# word embeddings

	# Loss Function
	loss_function = K.abs(model_output - target)	# -----> test with binary_cross loss (grad update?)
	#loss_function = (-1)*(K.abs(K.round(model_output) - target)*K.log(model_output)) + ((1 - K.abs(K.round(model_output) - target))*K.log(1 - model_output))

	#print("Loss function def")
	#print(loss_function)

	# Gradients Function
	grads_function = K.gradients(loss_function, model_input)[0]
	grads_function_em = K.gradients(loss_function, model_embeddings)[0]

	#print("Grads function def")
	#print(grads_function)

	# Function to collect gradients and cost
	pred_prob = K.function([model_input] + [K.learning_phase()], [model_output])
	grads = K.function([model_input] + [K.learning_phase()], [grads_function])
	grads_em = K.function([model_input] + [K.learning_phase()], [grads_function_em])
	loss = K.function([model_input] + [K.learning_phase()], [loss_function])

	#print("Loss, grads, pred_prob")
	#print(loss)
	#print(grads)
	#print(pred_prob)

	# Calculate class prob and gradients
	prob = pred_prob([adversarial_document, 0])#[0][0][0]
	loss_out = loss([adversarial_document, 0])
	gradients = grads([adversarial_document, 0])#[0][0]
	gradients_em = grads_em([adversarial_document, 0])[0][0]

	#print("Prob, loss, gradients")
	#print(prob)
	#print(gradients)
	#print(loss_out)
	#print(gradients[0].shape)

	#exit()

	return gradients, gradients_em



def prep_doc(sample, classification_idx, max_features, num_classes):

	document = np.copy(to_categorical(sample, num_classes=max_features))
	classification = np.copy(classification_idx)

	# Document to analyze
	#original_document = np.copy(document)
	adversarial_document = np.copy(document)
	adversarial_document = np.reshape(adversarial_document, (-1, adversarial_document.shape[0], adversarial_document.shape[1]))

	sc = 0

	# Get other class
	if classification == 0:
		target = np.zeros((num_classes))
		target[1] = 1
		original_classification_name = "unclassified"
		#print(colors.YELLOW + "Original review is negative." + colors.END)
	elif classification == 1:
		target = np.zeros((num_classes))
		target[0] = 1
		original_classification_name = "confidential"
		#print(colors.YELLOW + "Original review is positive." + colors.END)
	else:
		# Ignore secret documents
		target = np.zeros((num_classes))
		target[0] = 1
		original_classification_name = "secret"
		sc = 1

	return adversarial_document, classification, target



def get_most_influential_gradient(adversarial_document, gradients, row):

	word_idx = np.where(adversarial_document[0][row] == adversarial_document[0][row].max())[0][0]
	word_grad = gradients[row][word_idx]

	return word_idx, word_grad



def get_most_influential_words_for_dataset(x, y_idx, max_features, maxlen, num_classes, model, index_word_dict):

	# Bookeeping variables
	top3_counter_unclassified = Counter()
	top3_counter_confidential = Counter()
	top3_counter_secret = Counter()
	top3_words_per_doc_unclassified = list()
	top3_words_per_doc_confidential = list()
	top3_words_per_doc_secret = list()

	# Iterator
	i = 0

	# Number of documents to sample
	num_sampling_docs = 100

	secret_cnt = 0

	# Gather word embedding gradients for analysis
	word_embedding_grads = 0
	most_influential_input_grads = 0
	indexes = 0

	# Loop through each review, up until 100
	while((len(top3_words_per_doc_unclassified) < num_sampling_docs or len(top3_words_per_doc_confidential) < num_sampling_docs) and (i < len(x))):	

		# Keep track of highest gradients	
		top3_words = ["" for _ in range(3)]
		top3_grads = [0.0 for _ in range(3)]

		# Prepare document to analyze
		adversarial_document, classification, target = prep_doc(x[i], y_idx[i], max_features, num_classes)

		# Compute gradients
		gradients, gradients_em = compute_gradients(model, target, adversarial_document) 

		# Word embedding gradient analysis variables
		word_embedding_grads = list()
		most_influential_input_grads = list()
		indexes = list()

		# Calculate the highest gradient for each selected word (by column)
		for row in xrange(0, maxlen):

			# Get the most influetial gradient
			word_idx, word_grad = get_most_influential_gradient(adversarial_document, gradients, row)
			
			# Save the word index to get the actual word from the dictionary later
			indexes.append(word_idx)
			# Save the word embedding gradient for the word for analysis [to be removed]
			word_embedding_grads.append(gradients_em[row])
			# Absolute value of gradients
			word_grad = math.fabs(word_grad)
			# Save the most influential gradient for analysis [to be removed]
			most_influential_input_grads.append(word_grad)

			# If word is already on the list, then don't add it
			if index_word_dict[word_idx] not in top3_words:
				# If the magnitude of the gradient is greater than the top 3, replace accordingly
				if top3_grads[0] < word_grad:
					top3_grads[2] = top3_grads[1]
					top3_grads[1] = top3_grads[0]
					top3_grads[0] = word_grad
					top3_words[0] = index_word_dict[word_idx]
				elif top3_grads[1] < word_grad:
					top3_grads[2] = top3_grads[1]
					top3_grads[1] = word_grad
					top3_words[1] = index_word_dict[word_idx]
				elif top3_grads[2] < word_grad:
					top3_grads[2] = word_grad
					top3_words[2] = index_word_dict[word_idx]


		# Add the words to the counter
		if classification == 0:
			if len(top3_words_per_doc_unclassified) < num_sampling_docs:
				top3_words_per_doc_unclassified.append(top3_words)
				top3_counter_unclassified.update(top3_words)
		elif classification == 1:
			if len(top3_words_per_doc_confidential) < num_sampling_docs:
				top3_words_per_doc_confidential.append(top3_words)
				top3_counter_confidential.update(top3_words)
		else:
			# we ignore secret documents for now
			if len(top3_words_per_doc_secret) < num_sampling_docs:
				top3_words_per_doc_secret.append(top3_words)
				top3_counter_secret.update(top3_words)
			secret_cnt += 1


		# Update iterator
		i += 1


	return top3_counter_unclassified, top3_counter_confidential, top3_counter_secret



def get_most_influential_words_for_document(x, y_idx, max_features, maxlen, num_classes, model, index_word_dict):
	
	# Keep track of highest gradients	
	top3_words = ["" for _ in range(3)]
	top3_grads = [0.0 for _ in range(3)]

	# Prepare document to analyze
	adversarial_document, classification, target = prep_doc(x, y_idx, max_features, num_classes)

	# Compute gradients
	gradients, gradients_em = compute_gradients(model, target, adversarial_document) 

	# Word embedding gradient analysis variables
	#word_embedding_grads = list()
	#most_influential_input_grads = list()
	#indexes = list()

	# Calculate the highest gradient for each selected word (by column)
	for row in xrange(0, maxlen):

		# Get the most influetial gradient
		word_idx, word_grad = get_most_influential_gradient(adversarial_document, gradients, row)
			
		# Save the word index to get the actual word from the dictionary later
		#indexes.append(word_idx)
		# Save the word embedding gradient for the word for analysis [to be removed]	
		#word_embedding_grads.append(gradients_em[row])
		# Absolute value of gradients
		word_grad = math.fabs(word_grad)
		# Save the most influential gradient for analysis [to be removed]
		#most_influential_input_grads.append(word_grad)

		# If word is already on the list, then don't add it
		if index_word_dict[word_idx] not in top3_words:
			# If the magnitude of the gradient is greater than the top 3, replace accordingly
			if top3_grads[0] < word_grad:
				top3_grads[2] = top3_grads[1]
				top3_grads[1] = top3_grads[0]
				top3_grads[0] = word_grad
				top3_words[0] = index_word_dict[word_idx]
			elif top3_grads[1] < word_grad:
				top3_grads[2] = top3_grads[1]
				top3_grads[1] = word_grad
				top3_words[1] = index_word_dict[word_idx]
			elif top3_grads[2] < word_grad:
				top3_grads[2] = word_grad
				top3_words[2] = index_word_dict[word_idx]

	return top3_words
