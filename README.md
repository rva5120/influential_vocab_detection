# Influential Vocabulary Detection
This tool can be used to find the most influential words on a document. We define _most influential_ as the words that influence a trained classifier the most to give it a particular classification.

We use a Convolutional Neural Network model, as suggested by Keras and others, that can classify IMDB and Wikileaks documents with the following accuracies (50/50 train/test split):

| Dataset                     | Training | Testing | Dataset Size                                           |
|-----------------------------|:--------:|:-------:|:-------------------------------------------------------|
| IMDB                        |      84% |     83% | 25K positive, 25K negative reviews ????                |
| Wikileaks (2-way)           |      99% |     99% | 25K unclassified, 25K classified documents             |
| Wikileaks (3-way imbalanced)|      85% |     86% | 25K unclassified, 25K classified, 12K secret documents |

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
1. For the _IMDB dataset test_, use this:
```bash
python imdb.py
```

2. For the _Wikileaks dataset test_, follow these instructions:
   1. To use the pre-trained model for 2-way classification, run:
   ```bash
   python wikileaks.py
   ```
   
   2. To use the pre-trained model for 3-way classification, run:
   ```bash
   python wikileaks.py --num-classes 3
   ```
   
   3. To run the test from the raw cables.csv file, follow these steps:
      1. Download the cables.csv file from the Internet Archive, and then place it inside of the _dataset/wikileaks/_ folder.
         1. For unclassified and confidential documents, run the bash script in _dataset/wikileaks/**2-way**/prepare_dataset.sh_:
         ```bash
         ./prepare_dataset.sh
         ```
         2. For unclassified, confidential and secret documents (with an unbalanced secret class: 25K unclassified, 25K classified, 12K secret), run the bash script in _dataset/wikileaks/**3-way**/prepare_dataset.sh_.
      2. Once the dataset has been prepared, you may run `wikileaks.py` as described in 1 and 2.

4. For a _new project_, refer to one of the existing scripts and modify it accordingly. If you have any requests or problems, _please submit an issue above with your dataset details and needs_.


## Sample Output
Below is the output after running the IMDB test:
...img...


## Handling Unbalanced Datasets with Class Weights
Keras provides a functionality where one can assign class weights to resolve the issue of under-represented classes in the dataset. We provide the functionality in the code to do this, if desired. Be aware that the weights to be assigned to each class must be tuned accordingly.

To use this functionality, the model will have to be created and trained from scratch, since the pre-trained was not prepared using class weights:
```bash
python wikileaks.py --num-classes 3 --pre-trained False --use-class-weights True
```

## List of Improvements and Changes
- [x] Finish README.md

If you would like more functionality to be added or find any bugs, please submit an issue on this page. Thank you!
