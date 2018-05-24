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

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier



output_f = open('SPINN_output_with_linear_300_multi_2.txt', 'w')

def Dictionary():
	d=[]
	with open("sp-pi-2016-correct-eval1.txt", "r") as f:
		for line in f:
			values=line.split('\t')
			example = {}
			example['example_id'] = values[0]
			example['XP_label'] = values[1]
			example['words'] = re.sub("\\[|\\]|\\'|,", '', values[2].lower()).split(' ')
			example['hidden'] = [float(x) for x in re.sub("\\[|\\]|\\'|,", '', values[3].lower()).split(' ')][300:]
			if len(example['hidden']) == 300:
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

clf = linear_model.SGDClassifier()
clf.fit(x_train, y_train)

output_f.write(str(Counter(y_train)))
output_f.write('\n')
output_f.write(str(Counter(y_test)))
output_f.write('\n')

output_f.write(str(clf.score(x_train,y_train)))
output_f.write('\n')
output_f.write(str(clf.score(x_test,y_test)))
output_f.write('\n')


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

		



