#!/usr/bin/env python

import json
import codecs

SENTENCE_PAIR_DATA = True
FIXED_VOCABULARY = None

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
}


def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)
    return tokens, transitions



def process_parse(parse):
	parse_output = ''
	for i in range(len(parse)-1):
		if parse[i+1] == ')':
			parse_output = parse_output + parse[i]; parse_output = parse_output + ' '
		else:
			parse_output = parse_output + parse[i]
	parse_output = parse_output + ')' 
	parse_output = parse_output.split(' ')
	
	label_span = []
	for i in range(len(parse_output)):
		if parse_output[i][0] == '(':
			label = parse_output[i]
			paren_count = 0
			potential_span = parse_output[i:]
			for j in range(len(potential_span)):
				if potential_span[j][0] == '(':
					paren_count += 1
				elif potential_span[j] == ')':
					paren_count = paren_count - 1
				if paren_count == 0:
					span = potential_span[:j]
					break
			span = [w for w in span if w[0] != '(']
			span = [w.replace(')', '') for w in span]
			span = list(filter(None, span))
			
			label_span.append((label, span))
	return label_span
	
def return_labels(parse_output, span_list):
	#print(span_list)
	labels = {}
	for i in range(len(span_list)):
		#print(i)
		for j in parse_output:
			#print(j)
			if span_list[i] == j[1]:
				labels[i] = (j[0], span_list[i])
				break
				
	return labels

    
def convert_binary_bracketing_span(parse, lowercase=False):
    spans = []


    for i in range(len(parse.split(' '))):
        if parse.split(' ')[i] != "(":
            if parse.split(' ')[i] == ")":
                paren_count = 1
                #print(paren_count)
                span_temp = parse.split(' ')[:i]
                for j in reversed(range(len(span_temp))):
                    if span_temp[j] == ")":
                        paren_count += 1; #print(span_temp)
                    if span_temp[j] == "(":
                        paren_count = paren_count-1
                    if paren_count == 0:
                        output = parse.split(' ')[j:i]
                        output = [w.replace('(', '') for w in output]
                        output = [w.replace(')', '') for w in output]
                        output = list(filter(None, output))
                        spans.append(output)
                        #spans.append(re.sub('\\(|\\)', '',' '.join(span_temp[j:])))
                        break 
    return spans

def load_data(path, lowercase=False, choose=lambda x: True, eval_mode=False):
    print("Loading", path)
    examples = []
    failed_parse = 0
    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue

            if not choose(loaded_example):
                continue

            example = {}
            example["label"] = loaded_example["gold_label"]
            example["premise"] = loaded_example["sentence1"]
            example["hypothesis"] = loaded_example["sentence2"]
            example["example_id"] = loaded_example.get('pairID', 'NoID')
            if loaded_example["sentence1_binary_parse"] and loaded_example["sentence2_binary_parse"]:
                (example["premise_tokens"], example["premise_transitions"]) = convert_binary_bracketing(
                    loaded_example["sentence1_binary_parse"], lowercase=lowercase)
                (example["hypothesis_tokens"], example["hypothesis_transitions"]) = convert_binary_bracketing(
                    loaded_example["sentence2_binary_parse"], lowercase=lowercase)
                example['sentence_1_spans'] = convert_binary_bracketing_span(loaded_example['sentence1_binary_parse'])
                example['sentence_2_spans'] = convert_binary_bracketing_span(loaded_example['sentence2_binary_parse'])
                example['sentence_1_labels'] = return_labels(process_parse(loaded_example['sentence1_parse']), example['sentence_1_spans'])
                example['sentence_2_labels'] = return_labels(process_parse(loaded_example['sentence2_parse']), example['sentence_2_spans'])
                examples.append(example)
            else:
                failed_parse += 1
    if failed_parse > 0:
        print((
            "Warning: Failed to convert binary parse for {} examples.".format(failed_parse)))
    return examples


if __name__ == "__main__":
    # Demo:
    examples = load_data('snli-data/snli_1.0_dev.jsonl')
    print(examples[0])
