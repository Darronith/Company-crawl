import json
import nltk.data
import copy
import math
from difflib import SequenceMatcher

import os
java_path = "C:/Program Files/Java/jre1.8.0_45/bin/java.exe"
os.environ['JAVAHOME'] = java_path

#input_dir = '..\\data_sample\\'
#output_dir = '..\\silver_labels\\'
#report_dir = '..\\reports\\'

input_dir = '..\\..\\data_set\\data\\crawl\\data\\'
output_dir = '..\\..\\data_set\\data\\silver_labels\\'
report_dir = '..\\..\\data_set\\data\\reports\\'

list_file = '..\\list.txt'
list_of_file_names = open(list_file, 'r')

total_report = open('..\\report.txt', 'w')
useless_pages = open('..\\useless.txt', 'w')
almost_useless_pages = open('..\\almost_useless.txt', 'w')
num_of_files = 0
failed_match = 0
num_of_useless_pages = 0

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# format: [json_field_name, label_name]
json_fields = [['Company Name', 'COMPANY'], ['Company Contact', 'CONTACT'], ['State', 'STATE'], ['City', 'CITY'], ['Address', 'ADDRESS'], ['Zip', 'ZIP']]
# format: [json_field_name, label_name, global_counter, unique_counter, Nan_counter]
total_label_count = copy.deepcopy(json_fields)
for item in total_label_count:
    item.append(0)
    item.append(0)
    item.append(0)

# case sensitive fields
case_sensitive = ['STATE', 'CONTACT']

progress_count = 0
for file_name in list_of_file_names:
    progress_count += 1
    if progress_count > 5000 and progress_count <= 10000:
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
                        if not case_sensitive.__contains__(json_fields[field][1]):
                            uppercase = []
                            for t in target:
                                t = t.upper()
                                uppercase.append(t)
                            target = uppercase
                        target_list.append([target, json_fields[field][1], 0])
                        # in addresses, if the first word is a number, create another target without it
                        if json_fields[field][1] == 'ADDRESS' and target[0].isdigit():
                            target_list.append([target[1:], json_fields[field][1], 0])

        for content in company_data['content']:
            for sentence in tokenizer.tokenize(content):
                word_chain = []
                temp_target_list = copy.deepcopy(target_list)
                chain_label = 'New'
                position = 0
                word_list = nltk.word_tokenize(sentence)
                i = 0
                while i < len(word_list)+1:
                    if i != len(word_list):  # prevents out of range indexing
                        # print('current word '+word_list[i])
                        success = False
                        for t in range(0, len(temp_target_list)):
                            target = temp_target_list[t]
                            if position < len(target[0]) and target[1] != 'Skip':  # prevents out of range indexing
                                # matching the target with the words in the document
                                current_word = word_list[i]
                                if not case_sensitive.__contains__(target[1]):
                                    current_word = current_word.upper()
                                if SequenceMatcher(None, target[0][position], current_word).ratio() > 0.685 or target[0][position][:1] == current_word[:1] and position > 0:
                                    success = True
                                    chain_label = target[1]  # the last target that matches will be the label of the chain
                                    target_number = t
                                else:  # label name 'Skip' means this target has failed to match
                                    target[1] = 'Skip'
                    else:  # after processing the sentence one more loop to flush the last word chain
                        success = False

                    if success:
                        position += 1  # marks the progress of the matching
                        word_chain.append(word_list[i])

                    # success == False if every target is on 'Skip' mode and the chain is built,
                    # or all target failed immediately and the current word is getting the 'O' label
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
                            if i != len(word_list):  # out of range indexing prevention
                                file.write(word_list[i]+" O"+'\n')
                        word_chain = []
                        temp_target_list = copy.deepcopy(target_list)
                        chain_label = 'New'
                        position = 0
                    i += 1
                file.write('\n')  # marks the end of a sentence

        file.close()
        print("Labelling done in: "+file_name+'\n')
        labelling_report.write('Labelling in: '+file_name+'\n\n')

        at_least_once = []  # list of labels that were used at least once in the current document
        nan_list = []
        useless_page = True
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
                            useless_page = False

            else:
                report = 'Failed to find '+t[0]+', pattern was \'Nan\'.'
                # print(report)
                labelling_report.write(report+'\n')
                failed_match += 1
                nan_list.append(t[0])

        almost_no_match = True
        threshold = 1
        # we check if there were more types of label than defined in threshold
        # company name could be in some url without meaningful context, so we exclude it
        for label in at_least_once:
            if label != 'COMPANY':
                threshold -= 1
        if threshold < 0:
            almost_no_match = False

        labelling_report.write('\n')
        num_of_files += 1
        print("\nProcessed "+str(num_of_files)+' files.\n')
        # incrementing the unique counter
        for tt in total_label_count:
            if at_least_once.__contains__(tt[1]):
                tt[3] += 1
            if nan_list.__contains__(tt[1]):
                tt[4] += 1
        at_least_once.clear()
        if useless_page:
            num_of_useless_pages += 1
            useless_pages.write(file_name+'\n')
            labelling_report.write('This document was useless.\n')
        if almost_no_match:
            almost_useless_pages.write(file_name+'\n')
            labelling_report.write('This document was almost useless.\n')

# global report
# print('Processed '+str(num_of_files)+' files.')
total_report.write('Labelled '+str(num_of_files)+' files.\n\n')
for t in total_label_count:
    total_report.write('['+t[1]+']  entity found  ['+str(t[2])+']  times.\n')
total_report.write('\n')
for t in total_label_count:
    total_report.write('['+t[1]+']  entity found at least once in ['+str(t[3])+'] files. [ '+str("{0:.2f}".format(t[3]/num_of_files*100))+'% ]\n')
    total_report.write('['+str(t[4])+'] times the data was missing (Nan). ['+str("{0:.2f}".format(t[4]/num_of_files*100))+']\n\n')
total_report.write('\nNumber of failed searches due to \'Nan\' in json: '+str(failed_match))
total_report.write('\nNumber of useless documents: '+str(num_of_useless_pages))
