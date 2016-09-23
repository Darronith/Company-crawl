import json
import nltk.data
import copy
import math
from nltk.tag import StanfordNERTagger
from difflib import SequenceMatcher

import os
java_path = "C:/Program Files/Java/jre1.8.0_45/bin/java.exe"
os.environ['JAVAHOME'] = java_path

input_dir = '..\\data_sample\\'
output_dir = '..\\golden_labels\\'
report_dir = '..\\reports\\'

list_file = '..\\list.txt'
list_of_file_names = open(list_file, 'r')

total_report = open('..\\report.txt', 'w')
num_of_files = 0
failed_match = 0

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
st = StanfordNERTagger("../ws/stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz", "../ws/stanford_ner/stanford-ner.jar", encoding='utf-8')

# format: [json_field_name, label_name]
json_fields = [['Company Contact', 'CONTACT'], ['Company Name', 'COMPANY'], ['State', 'STATE'], ['City', 'CITY'], ['Address', 'ADDRESS'], ['Zip', 'ZIP']]
# format: [json_field_name, label_name, global_counter, unique_counter]
total_label_count = copy.deepcopy(json_fields)
for item in total_label_count:
    item.append(0)
    item.append(0)

for file_name in list_of_file_names:
    file_name = file_name.rstrip()  # remove whitespace ending
    print("Processing: "+file_name)
    company_data = json.load(open(input_dir+file_name, 'r', encoding='utf-8'))
    file = open(output_dir+file_name, 'w', encoding='utf-8')
    labelling_report = open(report_dir+file_name, 'w')
    # format: [[word_list], label_name, local_counter]
    target_list = []
    for field in range(0, len(json_fields)):
        for item in company_data[json_fields[field][0]]:
            if isinstance(item, float) and math.isnan(item):  # Nan symbol means the target is always skippable
                target_list.append([json_fields[field][1], 'Skip', 0])
            else:
                if json_fields[field][1] == 'ZIP':  # ZIP code needs additional splitting
                    target = nltk.word_tokenize(item)
                    target = target[0].split('-')
                    target_list.append([target, json_fields[field][1], 0])
                else:
                    target = nltk.word_tokenize(item)
                    target_list.append([target, json_fields[field][1], 0])

    for content in company_data['content']:
        for sentence in tokenizer.tokenize(content):
            word_chain = []
            temp_target_list = copy.deepcopy(target_list)
            chain_label = 'New'
            position = 0
            word_list = nltk.word_tokenize(sentence)
            i = 0
            while i < len(word_list):
                # print('current word '+word_list[i])
                success = False
                for t in range(0, len(temp_target_list)):
                    target = temp_target_list[t]
                    if position < len(target[0]) and target[1] != 'Skip':  # prevents out of range indexing
                        if SequenceMatcher(None, target[0][position], word_list[i]).ratio() > 0.685 or target[0][position][:1] == word_list[i][:1] and position > 0:
                            success = True
                            chain_label = target[1]  # the last target that matches will be the label of the chain
                            target_number = t
                        else:  # label name 'Skip' means this target has failed to match
                            target[1] = 'Skip'
                if success:
                    position += 1  # marks the progress of the matching
                    word_chain.append(word_list[i])

                # every target is on 'Skip' mode and the chain is built, or all target failed immediately and
                # the current word is getting the 'O' label
                if not success:
                    if len(word_chain) == 1:
                        file.write(word_chain[0]+" U-"+chain_label+'\n')
                        i -= 1  # chain breaking words may start a new chain, so we check it again (after resetting the local variables)
                        target_list[target_number][2] += 1
                    elif len(word_chain) > 1:
                        file.write(word_chain[0]+" B-"+chain_label+'\n')
                        for j in range(1, len(word_chain)-1):
                            file.write(word_chain[j]+" I-"+chain_label+'\n')
                        file.write(word_chain[-1]+" L-"+chain_label+'\n')
                        i -= 1  # chain breaking word
                        target_list[target_number][2] += 1
                    else:
                        file.write(word_list[i]+" O"+'\n')
                    word_chain = []
                    temp_target_list = copy.deepcopy(target_list)
                    chain_label = 'New'
                    position = 0
                i += 1
            file.write('\n')  # marks the end of a sentence

    if 1 == 2:
        for content in company_data['content']:
            for sentence in tokenizer.tokenize(content):
                labelled_sentence = st.tag(nltk.word_tokenize(sentence))
                label_type = 'New'  # buffer label
                current_label_type = 'New'  # current word's label
                word_list = []  # buffer words with the same label
                # label prefixes:
                # B - 'beginning'
                # I - 'inside'
                # L - 'last'
                # O - 'outside'
                # U - 'unit'
                for i in range(0, len(labelled_sentence)+1):
                    if i < len(labelled_sentence):
                        labelled_token = labelled_sentence[i]
                    # label renaming
                    if labelled_token[1] == 'LOCATION':
                            current_label_type = 'LOC'
                    elif labelled_token[1] == 'PERSON':
                            current_label_type = 'PER'
                    else:
                        current_label_type = 'O'
                    if i == len(labelled_sentence):
                        current_label_type = 'Last'

                    if current_label_type != label_type:
                        if label_type != 'O':
                            if len(word_list) == 1:
                                file.write(word_list[0]+" U-"+label_type+'\n')
                            if len(word_list) > 1:
                                file.write(word_list[0]+" B-"+label_type+'\n')
                                for i in range(1, len(word_list)-1):
                                    file.write(word_list[i]+" I-"+label_type+'\n')
                                file.write(word_list[-1]+" L-"+label_type+'\n')
                        else:
                            if len(word_list) != 0:
                                for w in word_list:
                                    file.write(w+" O"+'\n')
                        label_type = current_label_type
                        word_list = [labelled_token[0]]
                    else:
                        word_list.append(labelled_token[0])
                file.write('\n')  # marks the end of a sentence
    file.close()
    print("Labelling done in: "+file_name+'\n')
    labelling_report.write('Labelling in: '+file_name+'\n\n')

    at_least_once = []  # list of labels that were used at least once in the current document
    # local report
    for t in target_list:
        if t[1] != 'Skip':
            report = 'Labelled '+str(t[2])+' word chains with '+t[1]+', pattern used: '+' '.join(t[0])+'.'
            # print(report)
            labelling_report.write(report+'\n')
            for tt in total_label_count:
                if tt[1] == t[1]:
                    tt[2] += t[2]
                    if not at_least_once.__contains__(t[1]) and t[2] > 0:
                        at_least_once.append(t[1])
        else:
            report = 'Failed to find '+t[0]+', pattern was \'Nan\'.'
            # print(report)
            labelling_report.write(report+'\n')
            failed_match += 1
    labelling_report.write('\n')
    num_of_files += 1
    # incrementing the unique counter
    for tt in total_label_count:
        if at_least_once.__contains__(tt[1]):
            tt[3] += 1
    at_least_once.clear()

# global report
print('Processed '+str(num_of_files)+' files.')
total_report.write('Labelled '+str(num_of_files)+' files.\n\n')
for t in total_label_count:
    total_report.write('['+t[1]+']  entity found  ['+str(t[2])+']  times.\n')
total_report.write('\n')
for t in total_label_count:
    total_report.write('['+t[1]+']  entity found at least once in  ['+str(t[3])+']  files.   [ '+str("{0:.2f}".format(t[3]/num_of_files*100))+'% ]\n')
total_report.write('\nNumber of failed searches due to \'Nan\' in json: '+str(failed_match))
