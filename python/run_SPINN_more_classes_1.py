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

output_f = open('SPINN_output_with_linear_multi_1.txt', 'w')

def Dictionary():
	d=[]
	with open("sp-pi-2016-correct-eval1.txt", "r") as f:
		for line in f:
			values=line.split('\t')
			example = {}
			example['example_id'] = values[0]
			example['XP_label'] = values[1]
			example['words'] = re.sub("\\[|\\]|\\'|,", '', values[2].lower()).split(' ')
			example['hidden'] = [float(x) for x in re.sub("\\[|\\]|\\'|,", '', values[3].lower()).split(' ')]
			if len(example['hidden']) == 600:
				d.append(example)
            
    #return {}
    #You want to return the created dictionary
	return d

#embeddings_index = {}
#f = open(os.path.join('', '../glove.6B.50d.txt'), encoding='utf8')
#for line in f:
#    values = line.split()
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs
#f.close()

data_for_classifier = Dictionary()



#output_f.write(1)

#random.shuffle(data_for_classifier)

label_to_ix = {}
count = 0
for i in [x['XP_label'] for x in data_for_classifier]:
	if i not in label_to_ix.keys():
		label_to_ix[i] = count
		count += 1


#label_to_ix = {"(NP": 0, "(VP": 1, "(PP": 2}
#data_for_classifier= [x for x in data_for_classifier if x['XP_label'] in ['(NP', '(VP', '(PP']]
thres = round(len(data_for_classifier)*0.8)
input_for_classifier = [x['hidden'] for x in data_for_classifier]
labels_for_classifier = [label_to_ix[x['XP_label']] for x in data_for_classifier]


x_train = input_for_classifier[:thres]
y_train = labels_for_classifier[:thres]
x_test = input_for_classifier[thres:]
y_test = labels_for_classifier[thres:]

output_f.write(str(thres) + '\n')
output_f.write(str(len(data_for_classifier)) + '\n')
output_f.write(str(len(x_test)) + '\n')
output_f.write(str(len(y_test)) + '\n')




output_f.write('linear classifier' + '\n')

clf = linear_model.LogisticRegression(multi_class='multinomial', solver = 'sag')
clf.fit(x_train, y_train)

output_f.write(str(Counter(y_train)))
output_f.write('\n')
output_f.write(str(Counter(y_test)))
output_f.write('\n')

output_f.write(str(clf.score(x_train,y_train)))
output_f.write('\n')
output_f.write(str(clf.score(x_test,y_test)))
output_f.write('\n')

ix_to_label = {v:k[1:] for k,v in label_to_ix.items()}

#index_8_class = [x for x in list(range(len(y_test))) if y_test[x] in list(range(0,8))]
y_test = [x if x in list(range(0,8)) else 8 for x in y_test]
#x_test_8_class = [x_test[x] for x in list(range(len(x_test))) if x in index_8_class]

pred_test = clf.predict(x_test)
pred_test = [x if x in list(range(0,8)) else 8 for x in pred_test ]

#y_test_label = [ix_to_label[x] if x in list(range(0,8)) else 'other' for x in y_test ]
#pred_test_label = [ix_to_label[x] if x in list(range(0,8)) else 'other' for x in pred_test ]


linear_conf = confusion_matrix(y_test, pred_test)
plot_confusion_matrix(linear_conf, classes=list(ix_to_label.values())[:8]+['other'], normalize=True,
                      title='Normalized confusion matrix', output = "Linear_SPINN_8_class.png")

mlp = MLPClassifier(hidden_layer_sizes=(100),solver='sgd',learning_rate_init=0.002,max_iter=500)

#output_f.write(y_train)
#output_f.write(y_test)

mlp.fit(x_train, y_train)

output_f.write(str(mlp.score(x_train,y_train)))
output_f.write('\n')
output_f.write(str(mlp.score(x_test,y_test)))
output_f.write('\n')
output_f.write(str(Counter(y_train)))
output_f.write('\n')
output_f.write(str(Counter(y_test)))
output_f.write('\n')

output_f.close()

		
pred_test = mlp.predict(x_test)
pred_test = [x if x in list(range(0,8)) else 8 for x in pred_test ]

linear_conf = confusion_matrix(y_test, pred_test)
plot_confusion_matrix(linear_conf, classes=list(ix_to_label.values())[:8]+['other'], normalize=True,
                      title='Normalized confusion matrix', output = "MLP_SPINN_8_class.png")
		


