import json
import nltk.data
import copy
import math
import re
from difflib import SequenceMatcher

import os

java_path = "C:/Program Files/Java/jre1.8.0_45/bin/java.exe"
os.environ['JAVAHOME'] = java_path

# input_dir = '..\\data_sample\\'
# output_dir = '..\\silver_labels\\'
# report_dir = '..\\reports\\'

input_dir = '..\\..\\data_set\\data\\crawl\\data\\'
output_dir = '..\\..\\data_set\\data\\silver_labels\\'
report_dir = '..\\..\\data_set\\data\\reports\\'

list_file = '..\\list.txt'
list_of_file_names = open(list_file, 'r')

total_report = open('..\\report.txt', 'w')
useless_pages = open('..\\useless.txt', 'w')
almost_useless_pages = open('..\\almost_useless.txt', 'w')
uneven_pairs = open('..\\uneven_pairs.txt', 'w')
uneven_pairs.write('Format: LABEL: Actual words in document  --  [Pattern used]\n\n')
every_pair = open('..\\every_pair.txt', 'w')
every_pair.write('Format: LABEL: Actual words in document  --  [Pattern used]\n\n')
num_of_files = 0
failed_match = 0
num_of_useless_pages = 0
total_num_of_targets = 0
num_of_skipped_targets = 0

# controls how many words can be skipped in targets when matching
num_of_target_skip = 2
# controls how many words can be skipped in documents when matching
num_of_word_skip = 2
# controls the ratio of word chain length to target length to decide if the matching is accepted
accept_ratio = 2.0

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# format: [json_field_name, label_name]
json_fields = [['Company Name', 'COMPANY'], ['Company Contact', 'CONTACT'], ['State', 'STATE'], ['City', 'CITY'],
               ['Address', 'ADDRESS'], ['Zip', 'ZIP']]
# format: [json_field_name, label_name, global_counter, unique_counter, Nan_counter]
total_label_count = copy.deepcopy(json_fields)
for item in total_label_count:
    item.append(0)
    item.append(0)
    item.append(0)

# case sensitive fields
case_sensitive = ['STATE', 'CONTACT']

# interchangeable words
switchable_words = {'&': 'and',
                  'inc': 'incorporation'}

progress_count = 0
for file_name in list_of_file_names:
    progress_count += 1
    if progress_count > 2 and progress_count <= 40:
        file_name = file_name.rstrip()  # remove whitespace ending
        print("Processing: " + file_name)
        company_data = json.load(open(input_dir + file_name, 'r', encoding='utf-8'))
        file = open(output_dir + file_name, 'w', encoding='utf-8')
        labelling_report = open(report_dir + file_name, 'w')
        # format: [[word_list], label_name, local_counter, offset, skipped_words, score, is_normal, first_match, skip]
        target_list = []
        for field in range(0, len(json_fields)):
            item_list = company_data[json_fields[field][0]]
            # only process unique items
            item_set = set(item_list)
            # skipped targets due to repetition
            num_of_skipped_targets += (len(item_list)-len(item_set))
            target_set = set()
            for item in item_set:
                # Nan symbol means the target is always skippable
                if isinstance(item, float) and math.isnan(item):
                    target_list.append(['NaN', json_fields[field][1], 0, 0, 0, -100, 0, -1, 1])
                    continue
                target = nltk.word_tokenize(item)
                # case sensitivity check
                if not case_sensitive.__contains__(json_fields[field][1]):
                    uppercase = []
                    for t in target:
                        t = t.upper()
                        uppercase.append(t)
                    target = uppercase

                # replace short terms
                for i in range(0, len(target)):
                    t = target[i]
                    if t.lower() in switchable_words:
                        if not case_sensitive.__contains__(json_fields[field][1]):
                            target[i] = switchable_words[t.lower()].upper()
                        else:
                            target[i] = switchable_words[t.lower()]

                if json_fields[field][1] == 'ZIP':  # ZIP code needs additional splitting
                    target = target[0].split('-')
                if not target_set.__contains__(' '.join(target)):
                    target_list.append([target, json_fields[field][1], 0, 0, 0, 0, 0, -1, 0])
                    target_set.add(' '.join(target))
                    # in addresses, if the first word is a number, create another target without it
                    #if json_fields[field][1] == 'ADDRESS' and target[0].isdigit():
                    #    target_list.append([target[1:], json_fields[field][1], 0])
        # for each target, we create a set to store unique word chains labelled with that target
        list_of_sets_of_labelled_chains = []
        for t in target_list:
            list_of_sets_of_labelled_chains.append(set())

        for content in company_data['content']:
            for sentence in tokenizer.tokenize(content):
                word_chain = []
                temp_target_list = copy.deepcopy(target_list)
                chain_label = 'New'
                position = 0
                word_list = nltk.word_tokenize(sentence)
                i = 0
                while i < len(word_list) + 1:
                    if i != len(word_list):  # prevents out of range indexing
                        # print('current word '+word_list[i])
                        in_chain = False
                        for t in range(0, len(temp_target_list)):
                            target = temp_target_list[t]
                            success = False
                            # if 2 consecutive words has been skipped, stop matching this target
                            # also, if the current word couldn't match with the beginning of the target,
                            # stop matching now and try later again
                            if target[4] >= num_of_word_skip+1 or target[8] == 1:
                                target[8] = 1
                                continue
                            #offset = target[3]
                            if (position + target[3]) < len(target[0]) and target[8] != 1:  # prevents out of range indexing
                                # matching the target with the words from the document
                                current_word = word_list[i]
                                #in_chain = True
                                if not case_sensitive.__contains__(target[1]):
                                    current_word = current_word.upper()
                                if current_word.lower() in switchable_words:
                                    if not case_sensitive.__contains__(target[1]):
                                        current_word = switchable_words[current_word.lower()].upper()
                                    else:
                                        current_word = switchable_words[current_word.lower()]
                                matched = False
                                for j in range(0, num_of_target_skip+1):
                                    if not success:
                                        mod_position = position+target[3]+j
                                        if len(target[0]) > mod_position:  # preventing out of range indexing
                                            # short words need to be identical for matching
                                            if len(target[0][mod_position]) <= 4:
                                                sim_threshold = 0.95
                                            else:
                                                sim_threshold = 0.785
                                            # any number needs perfect matching
                                            if current_word.isdigit():
                                                if current_word == target[0][mod_position]:
                                                    matched = True
                                            # Person matching: fist word - 78.5%, every other, after the first match - only first letter
                                            elif target[1] == 'CONTACT':
                                                if SequenceMatcher(None, target[0][mod_position], current_word).ratio() > sim_threshold or target[0][mod_position][:1] == current_word[:1] and target[7] >= 0:
                                                    matched = True
                                            # anything else needs 78.5% matching on every word
                                            else:
                                                if SequenceMatcher(None, target[0][mod_position], current_word).ratio() > sim_threshold:
                                                    matched = True
                                        # out of range indexing could happen, but it's necessary to allow word skipping (regardless of stretching out of the length of the target)
                                        if matched:
                                            in_chain = True
                                            success = True
                                            #chain_label = target[1]  # the last target that matches will be the label of the chain
                                            #target_number = t
                                            target[3] += j
                                            target[4] = 0
                                            target[5] += 1 - j*0.55  # increase score because successful matching, but also decrease for skipping words in target
                                            if target[7] == -1:
                                                target[7] = mod_position
                                        elif j == 2:  # label name 'Skip' means this target has failed to match
                                            #target[1] = 'Skip'
                                            success = True
                                            target[3] -= 1
                                            target[4] += 1
                                            target[5] -= 0.55  # some penalty for skipping a word
                                            if position == 0:
                                                target[8] = 1
                                            else:
                                                in_chain = True
                                        if target[3] != 0:
                                            target[6] = 1
                        # if the chain continues, we increment the number of skipped words on every
                        # target, that has been set to 'Skip' state, because this way, the target with
                        # the highest score will have the proper amount of skipped words to function properly
                        if in_chain:
                            for t in temp_target_list:
                                if t[8] == 1:
                                    t[4] += 1
                                    t[5] -= 0.55
                    else:  # after processing the sentence one more loop to flush the last word chain
                        in_chain = False

                    if in_chain:
                        position += 1  # marks the progress of the matching
                        word_chain.append(word_list[i])

                    # success == False if every target is on 'Skip' mode and the chain is built,
                    # or all target failed immediately and the current word is getting the 'O' label
                    if not in_chain:
                        max_index = -1
                        max_score = -100
                        uneven = False
                        not_outside = False
                        for j in range(0, len(temp_target_list)):
                            t = temp_target_list[j]
                            #t[5] = min(len(t[0]), len(word_chain)) - 0.55 * abs(len(t[0]) - len(word_chain))
                            if t[5] > max_score:
                                max_score = t[5]
                                max_index = j
                        target = temp_target_list[max_index]
                        chain_label = target[1]
                        # remove words from the end of the chain equal to the number of skipped words associated with the target with the highest score
                        if target[4] != 0 and len(word_chain) != 0:
                            word_chain = word_chain[:-target[4]]
                        # a single word with O label shouldn't set back the word index by 1 ( which is the amount of skipped words by any targets )
                        if len(word_chain) != 0:
                            i -= target[4]

                        if len(word_chain) == 1:
                            # special case where the Unit label is probably not appropriate,
                            # because the chain is small and the word is small and it is not a STATE
                            # additionally, if the target too long, matching only once is not enough
                            if len(word_chain[0]) <= 3 and chain_label != 'STATE' or len(target[0]) > 2:
                                file.write(word_chain[0] + " O" + '\n')
                            else:
                                file.write(word_chain[0] + " U-" + chain_label + '\n')
                                if target[6] != 0:
                                    uneven = True
                                not_outside = True
                            i -= 1  # chain breaking words may start a new chain, so we check it again (after resetting the local variables)
                            target_list[max_index][2] += 1
                        elif len(word_chain) > 1:
                            if len(word_chain) >= len(target[0])/accept_ratio:
                                file.write(word_chain[0] + " B-" + chain_label + '\n')
                                for j in range(1, len(word_chain) - 1):
                                    file.write(word_chain[j] + " I-" + chain_label + '\n')
                                file.write(word_chain[-1] + " L-" + chain_label + '\n')
                                i -= 1  # chain breaking word
                                target_list[max_index][2] += 1
                                if target[6] != 0:
                                    uneven = True
                                not_outside = True
                            else:
                                for j in range(0, len(word_chain) - 1):
                                    file.write(word_chain[j] + " O" + '\n')
                        else:
                            if i != len(word_list):  # out of range indexing prevention
                                file.write(word_list[i] + " O" + '\n')
                        # collecting not normal matchings
                        if uneven:
                            uneven_pairs.write(chain_label+': '+' '.join(word_chain)+'\t--\t['+' '.join(temp_target_list[max_index][0])+']\n')
                        # collecting sets of chains to present in local reports
                        if not_outside:
                            list_of_sets_of_labelled_chains[max_index].add(' '.join(word_chain))
                            if progress_count < 100:
                                every_pair.write(chain_label+': '+' '.join(word_chain)+'\t--\t['+' '.join(temp_target_list[max_index][0])+']\n')
                            else:
                                every_pair.close()
                        word_chain = []
                        temp_target_list = copy.deepcopy(target_list)
                        chain_label = 'New'
                        position = 0
                    i += 1
                file.write('\n')  # marks the end of a sentence

        file.close()
        print("Labelling done in: " + file_name + '\n')
        labelling_report.write('Labelling in: ' + file_name + '\n\n')

        at_least_once = []  # list of labels that were used at least once in the current document
        nan_list = []
        useless_page = True
        # local report
        for i in range(0, len(target_list)):
            t = target_list[i]
            if t[8] != 1:
                report = 'Labelled ' + str(t[2]) + ' word chains with ' + t[1] + ', pattern used: ' + ' '.join(
                    t[0]) + '.'
                # print(report)
                labelling_report.write(report + '\n')
                labelling_report.write('->  '+', '.join(list_of_sets_of_labelled_chains[i])+'\n\n')
                for tt in total_label_count:
                    if tt[1] == t[1]:
                        tt[2] += t[2]
                        if not at_least_once.__contains__(t[1]) and t[2] > 0:
                            at_least_once.append(t[1])
                            useless_page = False

            else:
                report = 'Failed to find ' + t[1] + ', pattern was \'Nan\'.\n'
                # print(report)
                labelling_report.write(report + '\n')
                failed_match += 1
                nan_list.append(t[0])

        almost_no_match = True
        threshold = 1
        # we check if there were more types of label than defined in threshold
        # company name could be in some url without meaningful context, so we exclude it
        for label in at_least_once:
            if label != 'COMPANY':
                threshold -= 1
        if threshold < 0 or useless_page:
            almost_no_match = False

        labelling_report.write('\n')
        num_of_files += 1
        total_num_of_targets += len(target_list)
        print("\nProcessed " + str(num_of_files) + ' files.\n')
        # incrementing the unique counter
        for tt in total_label_count:
            if at_least_once.__contains__(tt[1]):
                tt[3] += 1
            if nan_list.__contains__(tt[1]):
                tt[4] += 1
        at_least_once.clear()
        if useless_page:
            num_of_useless_pages += 1
            useless_pages.write(file_name + '\n')
            labelling_report.write('This document was useless.\n')
        if almost_no_match:
            almost_useless_pages.write(file_name + '\n')
            labelling_report.write('This document was almost useless.\n')

# global report
# print('Processed '+str(num_of_files)+' files.')
total_report.write('Labelled ' + str(num_of_files) + ' files.\n\n')
for t in total_label_count:
    total_report.write('[' + t[1] + ']  entity found  [' + str(t[2]) + ']  times.\n')
total_report.write('\n')
for t in total_label_count:
    total_report.write('[' + t[1] + ']  entity found at least once in [' + str(t[3]) + '] files. [ ' + str(
        "{0:.2f}".format(t[3] / num_of_files * 100)) + '% ]\n')
    total_report.write('[' + str(t[4]) + '] times the data was missing (Nan). [' + str(
        "{0:.2f}".format(t[4] / num_of_files * 100)) + '%]\n\n')
total_report.write('\nNumber of failed searches due to \'Nan\' in json: ' + str(failed_match))
total_report.write('\nNumber of useless documents: ' + str(num_of_useless_pages))
total_report.write('\nNumber of targets: ' + str(total_num_of_targets))
total_report.write('\nNumber of skipped targets due to repetition: ' + str(num_of_skipped_targets))
