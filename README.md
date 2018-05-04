# Influential Vocabulary Detection
This tool can be used to find the most influential words on a document. We define _most influential_ as the words that influence a trained classifier the most to give it a particular classification.

We use a Convolutional Neural Network model, as suggested by Keras and others, that can classify IMDB and Wikileaks documents with the following accuracies (50/50 train/test split):

| Dataset       | Training | Testing |
| ------------- |:--------:|:-------:|
| IMDB          |      84% |     83% |
| Wikileaks     |      99% |     99% |

## Dependencies
To use this tool, you must have Keras installed with a TensorFlow backend.
1. To install **TensorFlow**, follow these instructions: https://www.tensorflow.org/install/
2. To install **Keras**, follow these instructions: https://keras.io/#installation
3. To install **NLTK**, follow these instructions: https://www.nltk.org/install.html 

NLTK will prompt you to download stopwords and WordNetLemmatizer. To do so, run these commands on a python interpreter:
```python
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
```
4. To install **Numpy**, follow these instructions: https://www.scipy.org/install.html
5. To install **TQDM**, follow these instructions: https://github.com/tqdm/tqdm#installation
6. To install **Scikit-Learn**, follow these instructions: http://scikit-learn.org/stable/install.html


## Usage
1. If you have a _dataset of documents encoded by vocabulary index and a matching label for each document_ (sample file: `dataset file`), then you can proceed to use this script:
```bash
python script_1.py
```

2. If you have a _dataset of raw documents and with corresponding labels_, then you can use this script:
```bash
python script_2.py
```

3. For the _IMDB dataset test_, use this:
```bash
python imdb.py
```

4. For the _Wikileaks dataset test_, follow these instructions:
⋅⋅1. Download the cables.csv files from the Internet Archive, and then place it inside of the _dataset/_ folder.
⋅⋅2. The file will be very large, so be aware of that. To be able to use our provided scripts in _dataset/wikileaks/_, you will have to remove the 

5. For a _new project_, refer to one of the existing scripts and modify it accordingly. If you have any issues, _please submit an issue above_.
  

## Sample Output


## List of Improvements and Changes
- [x] Finish README.md

If you would like more functionality to be added or find any bugs, please submit an issue on this page. Thank you!



There are several tools inside of this repository:
1. Inside of the _dataset/_ folder, you will find 2 scripts:
  - fix_25k.py: this script takes in the Wikileaks cables.csv file, and it extracts the document and its original classification. It also builds a vocabulary. This needs to be moved...
  - dataset_vars.py
