import json
import nltk.data
import copy
import math
from collections import defaultdict
import shutil
import os

##################################################################
# SETUP CONSTANTS AND PATHS

java_path = "C:/Program Files/Java/jre1.8.0_45/bin/java.exe"
os.environ['JAVAHOME'] = java_path

input_dir = '..\\..\\data_set\\data\\crawl\\'
output_dir = '..\\..\\data_set\\data\\silver_labels\\'
report_dir = '..\\..\\data_set\\data\\reports\\'
global_report_dir = '..\\..\\data_set\\data\\global_reports\\'

# controls whether we want to process only the index page or all four pages (for statistics only)
only_index = False

# controls how many words can be skipped in targets when matching
num_of_target_skip = 2  # default: 2
# controls how many words can be skipped in documents when matching
num_of_word_skip = 2  # default: 2
# controls the ratio of word chain length to target length to decide if the matching is accepted
accept_ratio = 2.0  # default: 2.0

# position of first file in list.txt that needs to be processed
from_num = 3590
# position of last file
to_num = 3591
# number of files per global report
global_report_interval = 1

##########################################################################

list_of_file_names = os.listdir(input_dir)
'''
if os.path.exists(global_report_dir+'declined_pairs'):
    shutil.rmtree(global_report_dir+'declined_pairs')
os.makedirs(global_report_dir+'declined_pairs')
if os.path.exists(global_report_dir+'general_reports'):
    shutil.rmtree(global_report_dir+'general_reports')
os.makedirs(global_report_dir+'general_reports')
if os.path.exists(global_report_dir+'pairs'):
    shutil.rmtree(global_report_dir+'pairs')
os.makedirs(global_report_dir+'pairs')
if os.path.exists(global_report_dir+'uneven_pairs'):
    shutil.rmtree(global_report_dir+'uneven_pairs')
os.makedirs(global_report_dir+'uneven_pairs')
'''
useless_pages = open(global_report_dir+'useless.txt', 'w', encoding='utf-8')
almost_useless_pages = open(global_report_dir+'almost_useless.txt', 'w', encoding='utf-8')

# collects matched pairs with holes
uneven_dict = defaultdict(int)
# collects valid matches that were declined later by various rules
declined_matches_dict = defaultdict(int)
# collects every match
pair_dict = defaultdict(int)
# how many files have been processed so far...
num_of_files = 0
# counts the number of words and empty lines, which is used to compare to labelled files
num_of_words = 0

# other measurements for reports
failed_match = 0
num_of_useless_pages = 0
total_num_of_targets = 0
num_of_skipped_targets = 0

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# format: [json_field_name, label_name]
json_fields = [['Company Name', 'COMPANY'], ['Company Contact', 'CONTACT'], ['State', 'STATE'],
               ['City', 'CITY'], ['Address', 'ADDRESS'], ['Zip', 'ZIP']]
# collecting data for the global report
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
                  'inc': 'incorporation',
                  'ave': 'avenue',
                   'rd': 'road',
                  '1st': 'first'}

# matching with these words won't count because it's too easy to fit on these due to their commonness
# this change probably won't ruin many valid matches, but it'll surely prevent many invalid ones (be careful, IN can be a state for example)
too_common_words = {'the', 'of', 'and'}

if only_index:
    keys_to_pages = ['index']
else:
    keys_to_pages = ['index', 'contact', 'partners', 'products']

interval_index = 0
progress_count = 0
periodic_report_progress = 0
for file_name in list_of_file_names:
    progress_count += 1
    if from_num < progress_count <= to_num:
        periodic_report_progress += 1
        file_name = file_name.rstrip()  # remove whitespace ending
        print("Processing: " + file_name)
        company_data = json.load(open(input_dir + file_name, 'r', encoding='utf-8'))
        file = open(output_dir + file_name, 'w', encoding='utf-8')
        labelling_report = open(report_dir + file_name, 'w', encoding='utf-8')
        # format: [[word_list], label_name, local_counter, offset, skipped_words, score, is_normal, first_match, skip]
        target_list = []
        for field in range(0, len(json_fields)):
            item_list = company_data[json_fields[field][0]]
            # only process unique items
            item_set = set(item_list)
            # only use the longer variation if one target is contained in another
            temp_set = copy.deepcopy(item_set)
            for item in item_set:
                for item2 in item_set:
                    if item != item2:
                        if isinstance(item, float) and not math.isnan(item) and isinstance(item2, float) and not math.isnan(item2) or not isinstance(item, float) and not isinstance(item2, float):
                            if item in item2 or item2 in item:
                                if len(item) > len(item2) and item2 in temp_set:
                                    temp_set.remove(item2)
                                elif item in temp_set:
                                    temp_set.remove(item)
            item_set = temp_set
            # skipped targets due to repetition
            num_of_skipped_targets += (len(item_list)-len(item_set))
            target_set = set()
            for item in item_set:
                # Nan symbol means the target is always skippable
                if isinstance(item, float) and math.isnan(item):
                    target_list.append([['NaN'], json_fields[field][1], 0, 0, 0, -100, 0, -1, 1])
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
                    # remove ending if it is a point
                    if t[-1:] == '.' and len(t) > 1:
                        t = t[:-1]
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
        # for each target, we create a set to store unique word chains labelled with that target
        # we use this information in local reports
        list_of_dicts_of_labelled_chains = []
        for t in target_list:
            list_of_dicts_of_labelled_chains.append(defaultdict(int))

        for dictionary_element in keys_to_pages:
            if dictionary_element in company_data['content']:
                num_of_words += 1
                for content in company_data['content'][dictionary_element]:
                    for sentence in tokenizer.tokenize(content):
                        word_chain = []
                        temp_target_list = copy.deepcopy(target_list)
                        chain_label = 'New'
                        # tells us the position of our matching progress in the current target
                        position = 0
                        word_list = nltk.word_tokenize(sentence)
                        num_of_words += len(word_list)+1  # +1 for the new line between sentences
                        # tells us the position of our matching progress in the current sentence (word_list)
                        i = 0
                        while i < len(word_list) + 1:
                            if i != len(word_list):  # prevents out of range indexing
                                # this means we are still building a word chain, which means we have at least one target
                                # that is matching, or having less than 3 words skipped
                                in_chain = False
                                for t in range(0, len(temp_target_list)):
                                    target = temp_target_list[t]
                                    # if 2 consecutive words has been skipped, stop matching this target
                                    # also, ignore targets if their skip field is set to 1
                                    # note: every target gets here eventually, which stops the chain building and starts the output printing
                                    if target[4] >= num_of_word_skip+1 or target[8] == 1:
                                        target[8] = 1
                                        continue
                                    # altering the current word by previously defined rules,
                                    # later, the original word will be used in the output
                                    current_word = word_list[i]
                                    if not case_sensitive.__contains__(target[1]):
                                        current_word = current_word.upper()
                                    if current_word[-1:] == '.' and len(current_word) > 1:
                                        current_word = current_word[:-1]
                                    if current_word.lower() in switchable_words:
                                        if not case_sensitive.__contains__(target[1]):
                                            current_word = switchable_words[current_word.lower()].upper()
                                        else:
                                            current_word = switchable_words[current_word.lower()]
                                    # usually every word is unique in any given target, so if one matching session finds
                                    # the same word twice, that means overlapping named entities being merged, which is
                                    # undesired, so we set this target to skip mode
                                    if position+target[3] > 0:
                                        if current_word in target[0][:position+target[3]] and current_word not in too_common_words:
                                            target[8] = 1  # if you find the same word that has already been matched, that probably means, that the matching should start from the second occurrence of said word
                                            continue
                                    # if a word matches from the document with a word from the target, matched == True
                                    matched = False
                                    for j in range(0, num_of_target_skip+1):
                                        # we only search for matching until we find the first one
                                        if not matched:
                                            # normal position is modified by the accumulated offset and j, which allows holes
                                            mod_position = position+target[3]+j
                                            if len(target[0]) > mod_position:  # preventing out of range indexing
                                                target_word = target[0][mod_position]
                                                # short words need to be identical for matching
                                                # sim_threshold defines the maximum acceptable length difference
                                                if len(target_word) <= 4:
                                                    sim_threshold = 0
                                                else:
                                                    sim_threshold = 1
                                                # any number needs perfect matching
                                                if current_word.isdigit():
                                                    if current_word == target_word:
                                                        matched = True
                                                # contact person matching after the first matching pair of words:
                                                # only first letter and at least one of them needs to be exactly 1 character long
                                                elif target[1] == 'CONTACT':
                                                    if (target_word in current_word or current_word in target_word) and abs(len(target_word)-len(current_word)) <= sim_threshold:
                                                        matched = True
                                                    if target_word[:1] == current_word[:1] and target[7] >= 0 and (len(current_word) == 1 or len(target_word) == 1):
                                                        matched = True
                                                else:
                                                    if (target_word in current_word or current_word in target_word) and abs(len(target_word)-len(current_word)) <= sim_threshold:
                                                        matched = True
                                                # invalidating otherwise accepted matches
                                                if matched and target[1] != 'STATE':
                                                    if mod_position != 0:  # allow names to start with 'The'
                                                        if too_common_words.__contains__(target_word.lower()) or too_common_words.__contains__(current_word.lower()):
                                                            matched = False
                                            # out of range indexing could happen, but it's necessary to allow word skipping (regardless of stretching out of the length of the target)
                                            if matched:
                                                in_chain = True
                                                target[3] += j  # increasing the offset appropriately so the modified position will mark the next word in the target
                                                target[4] = 0  # since we have matched a target, the number of skipped words at the end of the chain is 0
                                                target[5] += 1 - j*0.55  # increase score because successful matching, but also decrease for skipping words in target
                                                if target[7] == -1:  # saving the position of the first match, to use it during contact person matching
                                                    target[7] = mod_position
                                            elif j == 2:  # we have skipped ahead as many words in target as allowed, but still no matching
                                                target[3] -= 1  # reducing offset to show where we are in the target
                                                target[4] += 1  # the current word from the document couldn't match, so we skip it and increase the number of skipped words
                                                target[5] -= 0.55  # some penalty for skipping a word
                                                # if the current word couldn't match with the beginning 3 words of the target, stop matching
                                                if position == 0:  # position == 0 means we are at the beginning of the target
                                                    target[8] = 1  # skip this target
                                                    target[4] -= 1  # undo the incrementing, because after this, every skipped target will be incremented
                                                else:
                                                    in_chain = True
                                                    # we are still building a chain, because we allow some stretching in the matching (holes)
                                                    # as long as there is at least one target, that is not on skip state (skipped words < 3 by default),
                                                    # we build the chain waiting for matching pairs
                                            if target[3] != 0 and target[6] == 0:
                                                target[6] = -1  # this means that the offset was used, but we don't know if this will allow further matchings
                                            if target[6] == -1 and matched:
                                                target[6] = 1  # now we are certain, that the offset was used and the matching has holes in it

                                # if the chain continues, we increment the number of skipped words on every
                                # target, that has been set to 'Skip' state, because this way, the target with
                                # the highest score will have the proper amount of skipped words to function properly
                                if in_chain:
                                    for t in temp_target_list:
                                        if t[8] == 1:
                                            t[4] += 1  # skipped words
                                            t[5] -= 0.55  # score
                            else:  # after processing the sentence one more loop to flush the last word chain
                                in_chain = False

                            if in_chain:
                                position += 1  # marks the progress of the matching
                                word_chain.append(word_list[i])

                            # in_chain == False if every target is on Skip mode and the chain is built,
                            # or all target failed immediately (j==2, position==0 case), there is no chain
                            # and the current word is getting the 'O' label
                            if not in_chain:
                                max_index = -1  # index of target with the highest score
                                max_score = -100
                                uneven = False  # matching has holes in it
                                not_outside = False  # not labelled with O
                                for j in range(0, len(temp_target_list)):
                                    t = temp_target_list[j]
                                    if t[5] > max_score:
                                        max_score = t[5]
                                        max_index = j
                                target = temp_target_list[max_index]  # best target
                                chain_label = target[1]  # label of best target
                                # remove words from the end of the chain equal to the number of skipped words associated with the target with the highest score
                                if target[4] != 0 and len(word_chain) != 0:
                                    word_chain = word_chain[:-target[4]]
                                # a single word with O label shouldn't set back the word index by 1 ( which is the amount of skipped words by any targets )
                                if len(word_chain) != 0:
                                    i -= target[4]
                                # getting the word before the chain
                                if 0 <= i-len(word_chain)-1 < len(word_list):
                                    prev_word = word_list[i-len(word_chain)-1]
                                else:
                                    prev_word = ''
                                # getting the word after the chain
                                if i < len(word_list):
                                    next_word = word_list[i]
                                else:
                                    next_word = ''
                                # an informative string to write into various report files
                                string_to_write = chain_label+': '+prev_word+'   <| '+' '.join(word_chain)+' |>   '+next_word+'   --   ['+' '.join(temp_target_list[max_index][0])+']  {'+file_name+'}'
                                if len(word_chain) == 1:
                                    # special case where the Unit label is probably not appropriate,
                                    # because the chain is small and the word is small and it is not a STATE
                                    # additionally, if the target too long, matching only once is not enough
                                    # also, a number alone which is not a ZIP code, is also discarded
                                    if len(word_chain[0]) <= 3 and chain_label != 'STATE' and chain_label != 'ZIP' or len(target[0]) > 2 or word_chain[0].isdigit() and chain_label != 'ZIP':
                                        file.write(word_chain[0] + " O" + '\n')
                                        declined_matches_dict[string_to_write] += 1
                                    # we do not allow single words to match with names or company names, because
                                    # these usually wrong (e.g. matching last names generally doesn't mean it's the same person)
                                    elif chain_label == 'CONTACT' or chain_label == 'COMPANY':
                                        file.write(word_chain[0] + " O" + '\n')
                                        declined_matches_dict[string_to_write] += 1
                                    else:
                                        # unit label is used, because this word chain is only 1 word long
                                        file.write(word_chain[0] + " U-" + chain_label + '\n')
                                        target_list[max_index][2] += 1  # statistic for reports
                                        if target[6] == 1:
                                            uneven = True
                                        not_outside = True
                                    i -= 1  # chain breaking words may start a new chain, so we check it again (after resetting the local variables)
                                elif len(word_chain) > 1:
                                    # chain length must be at least half of the length of the target (by default)
                                    if len(word_chain) >= len(target[0])/accept_ratio:
                                        file.write(word_chain[0] + " B-" + chain_label + '\n')
                                        for j in range(1, len(word_chain) - 1):
                                            file.write(word_chain[j] + " I-" + chain_label + '\n')
                                        file.write(word_chain[-1] + " L-" + chain_label + '\n')
                                        i -= 1  # chain breaking word
                                        target_list[max_index][2] += 1
                                        if target[6] == 1:
                                            uneven = True
                                        not_outside = True
                                    else:
                                        for j in range(0, len(word_chain) - 1):
                                            file.write(word_chain[j] + " O" + '\n')
                                        declined_matches_dict[string_to_write] += 1
                                else:  # word chain is empty
                                    if i != len(word_list):  # out of range indexing prevention
                                        file.write(word_list[i] + " O" + '\n')
                                # collecting not normal matchings
                                if uneven:
                                    uneven_dict[string_to_write] += 1
                                # collecting sets of chains to present in local reports
                                if not_outside:
                                    list_of_dicts_of_labelled_chains[max_index][' '.join(word_chain)] += 1
                                    pair_dict[string_to_write] += 1
                                # resetting variables
                                word_chain = []
                                temp_target_list = copy.deepcopy(target_list)
                                chain_label = 'New'
                                position = 0
                            i += 1
                        file.write('\n')  # marks the end of a sentence
                file.write('\n')  # marks the end of a page

        file.close()
        print("Labelling done in: " + file_name + '\n')
        labelling_report.write('Labelling in: ' + file_name + '\n\n')

        at_least_once = set()  # set of labels that were used at least once in the current document
        nan_set = set()  # set of labels that had NaN in the json field
        useless_page = True
        # local report for every file
        for i in range(0, len(target_list)):
            t = target_list[i]
            if t[0][0] != 'NaN':
                report = 'Labelled ' + str(t[2]) + ' word chains with ' + t[1] + ', pattern used: ' + ' '.join(
                    t[0]) + '.'
                labelling_report.write(report + '\n')
                report = '-> '
                for w, ww in list_of_dicts_of_labelled_chains[i].items():
                    report += w+' ('+str(ww)+'),  '
                report = report[:-3]+'\n\n'
                labelling_report.write(report)
                for tt in total_label_count:
                    if tt[1] == t[1]:
                        tt[2] += t[2]
                        if t[2] > 0:
                            at_least_once.add(t[1])
                            useless_page = False

            else:
                report = 'Failed to find ' + t[1] + ', pattern was \'Nan\'.\n'
                labelling_report.write(report + '\n')
                failed_match += 1
                nan_set.add(t[1])

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

        # incrementing the unique counter for global report
        for tt in total_label_count:
            if tt[1] in at_least_once:
                tt[3] += 1
            if tt[1] in nan_set:
                tt[4] += 1

        if useless_page:
            num_of_useless_pages += 1
            useless_pages.write(file_name + '\n')
            labelling_report.write('This document was useless.\n')
        if almost_no_match:
            almost_useless_pages.write(file_name + '\n')
            labelling_report.write('This document was almost useless.\n')
        # extra data to check if every word is written in the labelled file
        labelling_report.write('Number of processed words and empty lines: '+str(num_of_words-2)+'\n')

        # global report after a given amount of files or when the processing of files is finished
        if periodic_report_progress == global_report_interval or progress_count == to_num:
            periodic_report_progress = 0
            interval_index += 1
            useless_pages.flush()
            almost_useless_pages.flush()

            total_report = open(global_report_dir+'general_reports\\report_'+str(interval_index)+'.txt', 'w', encoding='utf-8')
            total_report.write('Labelled ' + str(num_of_files) + ' files.\n\n')
            for i in range(0, len(total_label_count)):
                t = total_label_count[i]
                total_report.write('[' + t[1] + ']  entity found  [' + str(t[2]) + ']  times.\n')
            total_report.write('\n')
            for i in range(0, len(total_label_count)):
                t = total_label_count[i]
                total_report.write('[' + t[1] + ']  entity found at least once in [' + str(t[3]) + '] files. [ ' + str(
                    "{0:.2f}".format(t[3] / num_of_files * 100)) + '% ]\n')
                total_report.write('[' + str(t[4]) + '] times the data was missing (Nan). [' + str(
                    "{0:.2f}".format(t[4] / num_of_files * 100)) + '%]\n\n')
            total_report.write('\nNumber of failed searches due to \'Nan\' in json: ' + str(failed_match))
            total_report.write('\nNumber of useless documents: ' + str(num_of_useless_pages))
            total_report.write('\nNumber of targets: ' + str(total_num_of_targets))
            total_report.write('\nNumber of skipped targets due to repetition: ' + str(num_of_skipped_targets))
            total_report.close()

            every_pair_set = open(global_report_dir+'pairs\\every_pair_'+str(interval_index)+'.txt', 'w', encoding='utf-8')
            every_pair_set.write('Format: LABEL: Actual words in document  --  [Pattern used]  {Original file}  Number of occurrence\n\n')
            for t, tt in pair_dict.items():
                every_pair_set.write(t+'    Occurrence: '+str(tt)+'\n')
            every_pair_set.close()
            pair_dict.clear()

            declined_matches = open(global_report_dir+'declined_pairs\\declined_pairs_'+str(interval_index)+'.txt', 'w', encoding='utf-8')
            declined_matches.write('Format: LABEL: Actual words in document  --  [Pattern used]  {Original file}  Number of occurrence\n\n')
            for t, tt in declined_matches_dict.items():
                declined_matches.write(t+'    Occurrence: '+str(tt)+'\n')
            declined_matches.close()
            declined_matches_dict.clear()

            uneven_pairs_set = open(global_report_dir+'uneven_pairs\\uneven_pairs_'+str(interval_index)+'.txt', 'w', encoding='utf-8')
            uneven_pairs_set.write('Format: LABEL: Actual words in document  --  [Pattern used]  {Original file}  Number of occurrence\n\n')
            for t, tt in uneven_dict.items():
                uneven_pairs_set.write(t+'    Occurrence: '+str(tt)+'\n')
            uneven_pairs_set.close()
            uneven_dict.clear()

useless_pages.close()
almost_useless_pages.close()

