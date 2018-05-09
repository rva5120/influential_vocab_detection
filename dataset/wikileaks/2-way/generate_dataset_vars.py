import pickle
from tqdm import tqdm


# Load files
datafile = open("../cables_dataset.csv", "r")
vocabfile = open("vocab/vocabulary.pkl", "rb")

xfile = open("dataset_vars/x_dataset.pkl", "wb")
yfile = open("dataset_vars/y_dataset.pkl", "wb")
dictfile = open("dataset_vars/word_index_dict.pkl", "wb")
idxfile = open("dataset_vars/index_word_dict.pkl", "wb")
maxlenfile = open("dataset_vars/max_len.pkl", "wb")

# Load vocabulary
vocab = pickle.load(vocabfile)
ordered_vocab = vocab.most_common()


# Build dictionary of word:index
vocab_dict = dict()

vocab_dict["<PAD>"] = 0
vocab_dict["<UNK>"] = 1

idx = 2
single_instance_ctr = 0

for word in tqdm(ordered_vocab):
	# Count number of words that appear only once
	if word[1] < 10000:
		single_instance_ctr += 1
	else:
		# Add word to the vocabulary only if it appears at least 2x
		vocab_dict[word[0]] = idx
	# Update word index key
	idx += 1


# Save vocabulary dictionary
pickle.dump(vocab_dict, dictfile)


# Build dictionary of index:word
idx_dict = {vocab_dict[word]:word for word in vocab_dict}
pickle.dump(idx_dict, idxfile)


# Data Structures
X_unc = list()	# Unclassified
y_unc = list()
X_con = list()	# Confidential
y_con = list()
X = list()
y = list()


# Stats
other = set()
count_other = 0
shortest_record = 10000000
longest_record = -1
avg_record = 0		# compute avg at some point...

# Python 3 comp.
table = str.maketrans(dict.fromkeys("\n"))

uncl_len = 0
uncl_docs = 0
conf_len = 0
conf_docs = 0

# Get data
for line in tqdm(datafile.readlines()):

	# Gather record
	record = line.split(", ")
	doc = record[1:]
	label = record[0]

	# Convert words to vocabulary indices (we only keep words that are present in the vocab)
	# This can be updated to add "UNK" as a token for words that are not present in the
	# vocabulary. This is not implemented here, but can be if needed.
	doc = [vocab_dict[word.translate(table)] for word in doc if word in vocab_dict]

	# Save record according to classification
	if label == "unclassified":
		X_unc.append(doc)
		y_unc.append(label)
		uncl_len += len(doc)
		uncl_docs += 1
		# Gather stats
		if len(doc) > longest_record:
			longest_record = len(doc)
		if len(doc) < shortest_record:
			shortest_record = len(doc)
	elif label == "confidential":
		X_con.append(doc)
		y_con.append(label)
		conf_len += len(doc)
		conf_docs += 1
		# Gather stats
		if len(doc) > longest_record:
			longest_record = len(doc)
		if len(doc) < shortest_record:
			shortest_record = len(doc)
	else:
		count_other += 1
		other.add(label)


# Get stats
print("---")
print("Unclassified documents: "+str(len(X_unc)))
print("Confidential documents: "+str(len(X_con)))
print("---")
print("Unused documents: "+str(count_other))
print("Labels of unused documents: "+str(' -- '.join(list(other))))
print("---")
print("Longest record: "+str(longest_record)+" words")
print("Shortest record: "+str(shortest_record)+" words")
print("---")
print("Avg. Unclassified record length: "+str(float(uncl_len)/uncl_docs))
print("Avg. Condifential record length: "+str(float(conf_len)/conf_docs))
print("---")
print("Number of words cut from the vocabulary: "+str(single_instance_ctr))
print("---")
print("Vocabulary size: "+str(len(vocab_dict))+" words")
print("---")


# Combine data
X = X_unc + X_con
y = y_unc + y_con
print("Total: "+str(len(X)))


# Save data
print("Saving dataset...")
pickle.dump(X, xfile)
pickle.dump(y, yfile)
pickle.dump(longest_record, maxlenfile)


# Close files
datafile.close()
vocabfile.close()
xfile.close()
yfile.close()
dictfile.close()
idxfile.close()
maxlenfile.close()
