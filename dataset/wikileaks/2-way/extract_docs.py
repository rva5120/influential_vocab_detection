from tqdm import tqdm

from collections import Counter
import string
import re
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np


# Dataset files
f = open("../cables/cables_out.csv", "r")
o = open("../cables_dataset.csv", "w")
v = open("vocab/vocabulary.pkl", "wb")
v_aux = open("vocab/vocabulary_df.pkl", "wb")


# Data Structures
vocab = Counter()	# number of times a word appears throughout the dataset
vocab_df = Counter()	# number of documents a word appears in


# Choose 25k unclassified, 25k confidential and all secret
choose_unc = np.random.choice(75000, 25000, replace=False)
current_unc = 0

choose_con = np.random.choice(97000, 25000, replace=False)
current_con = 0



#########################################################
# Document Extraction					#
#########################################################

# There are 8 fields: "field0","field1", ... , "field7"
# Using regex, get the contents of field 4 and field 7
#	field 4   classification
#	field 7   document

# Regex to capture the field values for fields 4 and 7
regex = re.compile(r'(".*"),(".*"),(".*"),(".*"),(".*"),(".*"),(".*"),(".*")')

# Stopwords
stopwords = set(stopwords.words('english'))

# Lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

# Sensitive vocabulary
sensitive_vocab = set(["unclas", "declassifiedreleased", "declassified", "unclassified", "classified", "confidential", "secret"])

# Uninformative vocabulary
uninformative_vocab = set(["would", "also", "could", "tags", "ref", "many", "well", "since", "although", "within", "subject", "decl", "prel", "u", "na", "eo"])

# Punctuation and numbers (For Python 3)
punc_table = str.maketrans(dict.fromkeys(string.punctuation))
num_table = str.maketrans(dict.fromkeys('0123456789'))

other_records = 0

# Save classification and document
for line in tqdm(f.readlines()):

	#################
	# Get record	#
	#################
	# Get record grouped by field
	groups = re.findall(regex, line)[0]

	# Get classification
	classification = groups[4]
	classification = classification.lower()
	classification = classification.translate(punc_table)
	#classification = classification.translate(None, string.punctuation)
	# Get document
	document = groups[7]
	document = document.split(" ")


	#################	
	# Clean record	#
	#################
	# Remove empty words and set to lower case
	document = [word.lower() for word in document if word is not '']

	# Remove end of line character
	document = [word.split("\n")[0] for word in document]

	# Remove punctuation within a word
	# (For Python 3) 
	document = [word.translate(punc_table) for word in document]
	#document = [word.translate(None, string.punctuation) for word in document]

	# Remove numbers
	# (For Python 3) 
	document = [word.translate(num_table) for word in document]
	#document = [word.translate(None, '0123456789') for word in document]

	# Remove punctuations, stopwords, single characters, and empty words
	document = [word for word in document if (word not in string.punctuation) and (word not in stopwords) and (len(word) > 1) and (word is not '')]

	# Remove sensitive words that could bias the classifier (unclassified, classified, confidential, secret)
	document = [word for word in document if word not in sensitive_vocab]

	# Remove uninformative words to reduce the vocabulary size for better performance
	document = [word for word in document if word not in uninformative_vocab]

	# Word lemmatization to reduce the vocabulary size
	document = [wordnet_lemmatizer.lemmatize(word) for word in document]


	#################
	# Add record	#
	#################
	# Decide if record should be added
	if classification == "unclassified":
		if current_unc in choose_unc:
			# Add words to the vocabulary
			vocab.update(document)
			vocab_df.update(set(document))
			# Save the record
			record = list()
			record.append(classification)
			record = record + document
			# Write to a file
			o.write(', '.join(word for word in record) + "\n")

		# Update current unclassified document number
		current_unc += 1

	elif classification == "confidential":
		if current_con in choose_con:
			# Add words to the vocabulary
			vocab.update(document)
			vocab_df.update(set(document))
			# Save the record
			record = list()
			record.append(classification)
			record = record + document
			# Write to a file
			o.write(', '.join(word for word in record) + "\n")
		
		# Update current confidential document number
		current_con += 1

	else:	
		continue



# Save the vocabulary and word frequency counts
pickle.dump(vocab, v)
pickle.dump(vocab_df, v_aux)


# Close files
f.close()
o.close()
v.close()
v_aux.close()
