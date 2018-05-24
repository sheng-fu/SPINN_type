import os
import json
import random
import sys
import time
import math
import re

import numpy as np

from spinn.data.nli import load_nli_data
#from spinn.models.base import load_data_and_embeddings


def log_to_parse_dict(log_path):
	'''
	read a SPINN evaluation (batch size = 1) log, 
	parse the printouts of hidden vectors to a dictionary of dictionaries
	
	'''
	lines = [line.rstrip('\n') for line in open(log_path, 'r')]
	output = {}

	count = False
	for line in lines:
		if 'jpg' in line:
			temp_jpg = re.sub(".+ |\\[|\\]|\\'", '', line)
			output[temp_jpg] = {}
			output[temp_jpg]["sentence_1"] = []
			output[temp_jpg]["sentence_2"] = []
			output[temp_jpg]["path"] = []
			count_path = 0
		if len(re.findall("\\[0|\\[1|\\[2", line)) != 0:
			if "[1 " in line:
				output[temp_jpg]["path"].append('first')
			if " 1]" in line:
				output[temp_jpg]["path"].append('second')  
		if '[ ' in line:
			count = True
			bucket = []
		if count:
			splited = line.split(' ')
			for i in splited:
				if "." in i:
					bucket.append(float(re.sub("]", "", i)))
			if ']' in line:
				count = False
				if output[temp_jpg]["path"][count_path] == 'first':
					output[temp_jpg]["sentence_1"].append(bucket)
				else:
					output[temp_jpg]["sentence_2"].append(bucket)
				count_path += 1
	return output
		
def print_hidden(log_dict, nli_data, path):
	'''
	matching up the hidden vectors that are printed out and the labels for nodes at the binary parse
	'''

	f = open(path, 'w')

	#f.write('example_id' + '\t' + 'XP_label' + '\t' + 'words' + '\t' + 'hidden' + '\n')
	
	for data in nli_data:
		if max(data['sentence_1_labels'].keys()) + 1 == len(log_dict[data['example_id']]['sentence_1']):
			for key in data['sentence_1_labels'].keys():
				XP_label = data['sentence_1_labels'][key][0]
				words = data['sentence_1_labels'][key][1]
				hidden = log_dict[data['example_id']]['sentence_1'][key]
				f.write(data['example_id'] + '\t' + XP_label + '\t' + str(words) + '\t' + str(hidden) + '\n')
		if max(data['sentence_2_labels'].keys()) + 1 == len(log_dict[data['example_id']]['sentence_2']):
			for key in data['sentence_2_labels'].keys():
				XP_label = data['sentence_2_labels'][key][0]
				words = data['sentence_2_labels'][key][1]
				hidden = log_dict[data['example_id']]['sentence_2'][key]
				f.write(data['example_id'] + '\t' + XP_label + '\t' + str(words) + '\t' + str(hidden) + '\n')		
	
	f.close()	
	
print('sanity check')

nli_data = load_nli_data.load_data('../../../snli_1.0/snli_1.0_dev.jsonl')
#print(nli_data[0])
log_dict = log_to_parse_dict('../../../logs/sp-pi-2016-correct-eval1.log')
print_hidden(log_dict, nli_data, 'sp-pi-2016-correct-eval1.txt')



