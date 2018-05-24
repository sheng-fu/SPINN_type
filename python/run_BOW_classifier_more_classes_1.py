import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import itertools
import os
import json
import random
import sys
import time
import math
import csv
import numpy as np
import string, re
from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

from spinn.data.nli import load_nli_data
#from spinn.models.base import load_data_and_embeddings

#from spinn.models.base import get_data_manager, get_flags, get_batch
#from spinn.models.base import flag_defaults, init_model, log_path
#from spinn import util
#import parse_output_hidden as parse_tool

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, output = ""):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output, dpi=500)

output_f = open('BOW_output_multi_with_linear_1.txt', 'w')

nli_data = load_nli_data.load_data('../../../snli_1.0/snli_1.0_dev.jsonl')
#print(nli_data[3]['sentence_2_spans'])
#print(nli_data[3]['sentence_2_labels'])
#print(nli_data[2]['sentence_2_labels'])
#print(nli_data[0]['sentence_2_labels'])
#print(nli_data[3]['hypothesis'])
#print(nli_data[3]['hypothesis_transitions'])
#print(sum(nli_data[3]['hypothesis_transitions']))
#print(len(nli_data[3]['sentence_2_labels']))
#output_f.write(nli_data[2])
#load word embeddings



data_for_classifier = []
words = []

for data in nli_data:
	for key in data['sentence_1_labels'].keys():
		#print(key)
		item = {}
		item['label'] = data['sentence_1_labels'][key][0]
		item['words'] = data['sentence_1_labels'][key][1]
		item['length'] = len(item['words'])
		#words = words + data['sentence_1_labels'][key][1]
		data_for_classifier.append(item)
	for key in data['sentence_2_labels'].keys():
		item = {}
		item['label'] = data['sentence_2_labels'][key][0]
		item['words'] = data['sentence_2_labels'][key][1]
		item['length'] = len(item['words'])
		data_for_classifier.append(item)
		#words = words + data['sentence_2_labels'][key][1]

#count_words = Counter([w.lower() for w in words])
#voc = [count_words.keys()]

dim = 300

embeddings_index = {}
f = open(os.path.join('', '../../../glove/glove.840B.300d.txt'), encoding='utf8')
for line in f:
    values = line.split(' ')
    word = values[0]
    #print(word)
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

output_f.write('glove loaded! \n')


#data_for_classifier = [x for x in data_for_classifier if x['length'] <= 3]


for data in data_for_classifier:
	temp = np.zeros(300)
	for i in data['words']:
		if i in embeddings_index.keys():
			temp = temp + np.array(embeddings_index[i])
	data['BOW'] = temp

#output_f.write(data_for_classifier[0])

thres = round(len(data_for_classifier)*0.8)

#random.shuffle(data_for_classifier)
		
label_to_ix = {}
count = 0
for i in [x['label'] for x in data_for_classifier]:
	if i not in label_to_ix.keys():
		label_to_ix[i] = count
		count += 1

#results = [x for x in data_for_classifier if x['label'] in ['(NP', '(VP', '(PP']]
input_for_classifier = [x['BOW'] for x in data_for_classifier]
labels_for_classifier = [label_to_ix[x['label']] for x in data_for_classifier]


x_train = input_for_classifier[:thres]
y_train = labels_for_classifier[:thres]
x_test = input_for_classifier[thres:]
y_test = labels_for_classifier[thres:]



output_f.write('linear classifier')

clf = linear_model.LogisticRegression(multi_class='multinomial', solver = 'sag')

clf.fit(x_train , y_train)

output_f.write(str(clf.score(x_train,y_train)))
output_f.write('\n')
output_f.write(str(clf.score(x_test,y_test)))
output_f.write('\n')
output_f.write(str(label_to_ix))
output_f.write('\n')
output_f.write(str(Counter(y_train)))
output_f.write('\n')
output_f.write(str(Counter(y_test)))
output_f.write('\n')

ix_to_label = {v:k[1:] for k,v in label_to_ix.items()}
print(label_to_ix)
print(ix_to_label)
print(list(ix_to_label.keys())[:8])

#index_8_class = [x for x in list(range(len(y_test))) if y_test[x] in list(range(0,8))]
y_test = [x if x in list(range(0,8)) else 8 for x in y_test]
#x_test_8_class = [x_test[x] for x in list(range(len(x_test))) if x in index_8_class]

pred_test = clf.predict(x_test)
pred_test = [x if x in list(range(0,8)) else 8 for x in pred_test ]

#y_test_label = [ix_to_label[x] if x in list(range(0,8)) else 'other' for x in y_test ]
#pred_test_label = [ix_to_label[x] if x in list(range(0,8)) else 'other' for x in pred_test ]

#print(y_test[:10])
#print(pred_test[:10])
#print(y_test_label[:10])
#print(pred_test_label[:10])


linear_conf = confusion_matrix(y_test, pred_test)
plot_confusion_matrix(linear_conf, classes=list(ix_to_label.values())[:8]+['other'], normalize=True,
                      title='Normalized confusion matrix', output = "Linear_BOW_8_class.png")


mlp = MLPClassifier(hidden_layer_sizes=(100),solver='sgd',learning_rate_init=0.002,max_iter=500)

#output_f.write(y_train)
#output_f.write(y_test)

mlp.fit(x_train, y_train)

output_f.write(str(mlp.score(x_train,y_train)))
output_f.write('\n')
output_f.write(str(mlp.score(x_test,y_test)))
output_f.write('\n')

output_f.close()

pred_test = mlp.predict(x_test)
pred_test = [x if x in list(range(0,8)) else 8 for x in pred_test ]

#pred_test_label = [ix_to_label[x] if x in list(range(0,8)) else 'other' for x in pred_test ]

linear_conf = confusion_matrix(y_test, pred_test)
plot_confusion_matrix(linear_conf, classes=list(ix_to_label.values())[:8]+['other'], normalize=True,
                      title='Normalized confusion matrix', output = "MLP_BOW_19_class.png")
		






