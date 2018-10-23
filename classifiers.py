
'''
Sentiment Analysis for IMDb using different classifiers
You can download the training and test data from:
ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
Author: Claudio
'''


'''----------------------------------------------------------------------------------------
Moving all the data into the correct folders and renaming it
'''
import os
import shutil

path = ['./aclImdb/train/neg/', './aclImdb/train/pos/', './aclImdb/test/neg/', './aclImdb/test/pos/']
move_to = ['./df/neg/', './df/pos/']
mood = ['neg_','pos_']

for i in range(0,2):
    if not os.path.exists(move_to[i]):
        os.makedirs(move_to[i])

# Move all the old training and test files into a training data folder, sorted by negative and positive
# Additionally, the files are renamed
for i in range (0,4):
    files = os.listdir(path[i])
    counter = 0
    for f in files:
        src = path[i] + f
        dst = move_to[i%2] + mood[i%2] + str(int(i/2)*len(files) + counter) + '.txt'
        os.rename(src, dst)
        counter += 1
        

# Create Training and Test data (70%, 30%)
from sklearn.model_selection import train_test_split
import pickle

path = ['./df/neg/', './df/pos/']

# Constructing a dataframe
# Each element consists of a text and its according label
df = []
for i in range(0,2):
    files = os.listdir(path[i])
    for f in files:
        content = open(path[i] + f, 'r')
        df.append({'text': content.read().replace('\b', ' ').replace('\n', ' ').replace('"',"'").replace('<br />',' '),
                    'label': i})
        
X = [d['text'] for d in df]
y = [d['label'] for d in df]

# Split the training data from the testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, stratify= y, random_state= 100)

# Storing the dataframe in a pickle
pickle.dump(df, open('df.pkl', 'wb'))


'''----------------------------------------------------------------------------------------
Create different Classifiers
'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score

classifier_paths = [
    'NB_mono_CV.pkl',
    'LR_mono_CV.pkl',
    'NB_bi_CV.pkl',
    'LR_bi_CV.pkl',
    'NB_mono_TfidfV.pkl',
    'LR_mono_TfidfV.pkl',
    'NB_bi_TfidfV.pkl',
    'LR_bi_TfidfV.pkl'
]

classifier_names = [
    'Naive Bayes with Monogram CounterVectorizer',
    'Logistic Regression with Monogram CounterVectorizer',
    'Naive Bayes with Bigram CounterVectorizer',
    'Logistic Regression with Bigram CounterVectorizer',
    'Naive Bayes with Monogram TfidfVectorizer',
    'Logistic Regression with Monogram TfidfVectorizer',
    'Naive Bayes with Bigram TfidfVectorizer',
    'Logistic Regression with Bigram TfidfVectorizer'
]

classifiers = [
    Pipeline([('vectorizer', CountVectorizer()),('NB_model', MultinomialNB())]),
    Pipeline([('vectorizer', CountVectorizer()),('LR_model', LogisticRegression())]),
    Pipeline([('vectorizer', CountVectorizer(ngram_range= (1,2))),('NB_model', MultinomialNB())]),
    Pipeline([('vectorizer', CountVectorizer(ngram_range= (1,2))),('NB_model', LogisticRegression())]),
    Pipeline([('vectorizer', TfidfVectorizer()),('NB_model', MultinomialNB())]),
    Pipeline([('vectorizer', TfidfVectorizer()),('LR_model', LogisticRegression())]),
    Pipeline([('vectorizer', TfidfVectorizer(ngram_range= (1,2))),('NB_model', MultinomialNB())]),
    Pipeline([('vectorizer', TfidfVectorizer(ngram_range= (1,2))),('NB_model', LogisticRegression())])
]

'''----------------------------------------------------------------------------------------
Train, test and print the results of the different classifiers
'''
'''
for i in range(0,len(classifiers)):
    # Train
    clf = classifiers[i]
    clf.fit(X_train, y_train)
    pickle.dump(clf, open('LR_mono_CV.pkl', 'wb'))

    # Test
    clf = pickle.load(open(classifier_paths[i], "rb" ))
    y_pred = clf.predict(X_test)
    
    # Print
    print(classifier_names[i])
    print("AUC: {:.4f}".format(roc_auc_score(y_test, y_pred)))
    print("Precision (positive): {:.4f}".format(precision_score(y_test, y_pred, pos_label=1)))
    print("Precision (negative): {:.4f}".format(precision_score(y_test, y_pred, pos_label=0)))
    print("Recall (positive): {:.4f}".format(recall_score(y_test, y_pred, pos_label=1)))
    print("Recall (negative): {:.4f}".format(recall_score(y_test, y_pred, pos_label=0)))
    print('\n')
'''

'''----------------------------------------------------------------------------------------
5.1
Use Facebook FastText to train the model
Preprocess the training data
__label__positive and __label__negative needs to be added
'''
'''
file = open('training.txt', 'w+')
for i in range(0, len(X_train)):
    if y_train[i] == 0:
        file.write('__label__negative {}\n'.format(X_train[i]))
    elif y_train[i] == 1:
        file.write('__label__positive {}\n'.format(X_train[i]))
    else:
        pass

file = open('testing.txt', 'w+')
for i in range(0, len(X_test)):
    if y_test[i] == 0:
        file.write('__label__negative {}\n'.format(X_test[i]))
    elif y_test[i] == 1:
        file.write('__label__postitive {}\n'.format(X_test[i]))
    else:
        pass


# Use fastText to train data
from fastText import train_supervised, load_model

model = train_supervised(input= 'training.txt', epoch=25, lr=1.0, wordNgrams=2, verbose=2, minCount=1)
model.save_model("model.bin")


# Test fastText model
model = load_model("model.bin")
y_pred = []

for i in range(0, len(X_test)):
    if model.predict(X_test[i])[0][0] == '__label__negative':
        y_pred.append(0)
    elif model.predict(X_test[i])[0][0] == '__label__positive':
        y_pred.append(1)
    else:
        pass   
    
print('FastText Model')
print("AUC: {:.4f}".format(roc_auc_score(y_test, y_pred)))
print("Precision (positive): {:.4f}".format(precision_score(y_test, y_pred, pos_label=1)))
print("Precision (negative): {:.4f}".format(precision_score(y_test, y_pred, pos_label=0)))
print("Recall (positive): {:.4f}".format(recall_score(y_test, y_pred, pos_label=1)))
print("Recall (negative): {:.4f}".format(recall_score(y_test, y_pred, pos_label=0)))
print('\n')    
'''
'''----------------------------------------------------------------------------------------
Neural Network model
'''
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.layers.core import input_data, fully_connected, flatten, dropout
from tflearn.layers.recurrent import lstm
from tflearn.layers.estimator import regression
import numpy as np

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Padding
X_train = pad_sequences(X_train, maxlen=100, value=0.)
X_test = pad_sequences(X_test, maxlen=100, value=0.)

# Converting to binary vector
y_train = to_categorical(y_train,2)
y_test = to_categorical(y_test,2)

# Build a network
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, 
    optimizer='adam',
    learning_rate=0.001,
    loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X_train,
    y_train,
    validation_set=(X_test, y_test),
    show_metric=True,
    batch_size=32)
'''----------------------------------------------------------------------------------------
Store the best classifier as a pickle named model
'''
best_clf = pickle.load(open('LR_bi_CV.pkl', "rb" ))
pickle.dump(best_clf, open('model.pkl', 'wb'))

