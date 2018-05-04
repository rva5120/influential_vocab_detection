# Influential Vocabulary Detection
This tool can be used to find the most influential words on a document. We define _most influential_ as the words that influence a trained classifier the most to give it a particular classification.

- Dependencies
To use this tool, you must have Keras installed with a TensorFlow backend.
1. To install TensorFlow, follow these instructions: https://www.tensorflow.org/install/
2. To install Keras, follow these instructions: https://keras.io/#installation
3. To install NLTK, follow these instructions: https://www.nltk.org/install.html
  - NLTK will prompt you to download stopwords and WordNetLemmatizer. To do so, follow these instructions:
    - Open the python interpreter, and run the following commands:
    [code]
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')

- Usage Details
There are several tools inside of this repository:
1. Inside of the _dataset/_ folder, you will find 2 scripts:
  - fix_25k.py: this script takes in the Wikileaks cables.csv file, and it extracts the document and its original classification. It also builds a vocabulary. This needs to be moved...
  - dataset_vars.py
