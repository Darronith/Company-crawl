import json
import nltk.data
import copy
import math
from collections import defaultdict
import shutil
import os

# REQUIRED FILE STRUCTURE:
#                 -root:
#                   -NER_tagger.py
#                   -crawl:
#                       -data:
#                           -every file that needs to be processed

# file structure definition
input_dir = 'crawl\\data\\'
output_dir = 'crawl\\silver_labels\\'
report_dir = 'crawl\\local_reports\\'
global_report_dir = 'crawl\\global_reports\\'
# list of file names of the data that needs to be processed
list_of_file_names = os.listdir(input_dir)

##################################################################
# SETUP CONSTANTS AND PATHS

# controls whether we want to process only the index page or all four pages (for statistics only)
only_index = False

# controls how many words can be skipped in targets when matching
num_of_target_skip = 2  # default: 2
# controls how many words can be skipped in documents when matching
num_of_word_skip = 2  # default: 2
# controls the ratio of word chain length to target length to decide if the matching is accepted
accept_ratio = 2.0  # default: 2.0
# number of lower case words allowed in a matching (scales with length)
allowed_lowercase_words = 1  # default: 1

# position before the first file in the list that needs to be processed
from_num = 5000  # default: 0
# position of last file
to_num = 5500  # default: len(list_of_file_names)
# number of files per global report
global_report_interval = 100
# controls the production of local reports, each containing labelling information for the corresponding file
local_report_enabled = True

##########################################################################
# creating empty directory structure
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
if os.path.exists(global_report_dir+'aborted_matchings'):
    shutil.rmtree(global_report_dir+'aborted_matchings')
os.makedirs(global_report_dir+'aborted_matchings')

if not os.path.exists(output_dir[:-1]):
    os.makedirs(output_dir[:-1])
if (not os.path.exists(report_dir[:-1])) and local_report_enabled:
    os.makedirs(report_dir[:-1])
if not os.path.exists(global_report_dir[:-1]):
    os.makedirs(global_report_dir[:-1])

if (not local_report_enabled) and os.path.exists(report_dir[:-1]):
    shutil.rmtree(report_dir[:-1])

useless_pages = open(global_report_dir+'useless.txt', 'w', encoding='utf-8')
almost_useless_pages = open(global_report_dir+'almost_useless.txt', 'w', encoding='utf-8')
# collects aborted matchings due to repetition of a previously matched word
aborted_matchings = set()
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

# case sensitive field(s)
case_sensitive = ['STATE']

# these categories cannot be used on unit labels when the target has multiple words
denied_unit_labels = ['CONTACT', 'COMPANY', 'ADDRESS', 'CITY']

# interchangeable words, keys must be all lower case, multi-worded states are ignored, because current implementation can only handle single words: 'nh': 'New Hampshire', 'nj': 'New Jersey', 'nm': 'New Mexico', 'ny': 'New York', 'nc': 'North Carolina','nd': 'North Dakota', 'ri': 'Rhode Island', 'sc': 'South Carolina', 'sd': 'South Dakota', 'wv': 'West Virginia',
switchable_words = {'alabama':	'AL', 'alaska':	'AK', 'arizona':	'AZ', 'arkansas':	'AR', 'california':	'CA', 'colorado':	'CO',
                    'connecticut':	'CT', 'delaware':	'DE', 'florida':	'FL', 'georgia':	'GA', 'hawaii':	'HI', 'idaho':	'ID',
                    'illinois':	'IL', 'indiana':	'IN', 'iowa':	'IA', 'kansas':	'KS', 'kentucky':	'KY', 'louisiana':	'LA',
                    'maine':	'ME', 'maryland':	'MD', 'massachusetts':	'MA', 'michigan':	'MI', 'minnesota':	'MN',
                    'mississippi':	'MS', 'missouri':	'MO', 'montana':	'MT', 'nebraska':	'NE', 'nevada':	'NV',
                    'ohio':	'OH', 'oklahoma':	'OK', 'oregon':	'OR', 'pennsylvania':	'PA', 'tennessee':	'TN', 'texas':	'TX',
                    'utah':	'UT', 'vermont':	'VT', 'virginia':	'VA', 'washington':	'WA', 'wisconsin':	'WI', 'wyoming':	'WY',
                    '&': 'and',
                    'inc': 'incorporation',
                    'ave': 'avenue',
                    'rd': 'road',
                    '1st': 'first', '2nd': 'second', '3rd': 'third',
                    'ctr': 'center',
                    'street': 'st',
                    'saint': 'st',
                    'e': 'east', 'n': 'north', 's': 'south', 'w': 'west',
                    'drive': 'dr',
                    'doctor': 'dr',
                    'co': 'company',
                    'assoc': 'associates',
                    'assn': 'association',
                    'twp': 'township',
                    'pl': 'place',
                    'U.S.A': 'USA',
                    'svc': 'service',
                    'intl': 'international',
                    'ln': 'lane',
                    'acad': 'academy',
                    'veterinary': 'vet'}

# matching with these words won't count because it's too easy to fit on these due to their commonness
# this change probably won't ruin many valid matches, but it'll surely prevent many invalid ones (be careful, IN can be a state for example)
too_common_words = {'the', 'of', 'and', 'for', '&'}

if only_index:
    keys_to_pages = ['index']
else:
    keys_to_pages = ['index', 'contact', 'partners', 'products']

interval_index = 0
progress_count = 0
periodic_report_progress = 0
#############################
# CREATING TARGETS FROM JSON
#############################
for file_name in list_of_file_names:
    progress_count += 1
    if from_num < progress_count <= to_num:
        periodic_report_progress += 1
        file_name = file_name.rstrip()  # remove whitespace ending
        print("Processing: " + file_name)
        company_data = json.load(open(input_dir + file_name, 'r', encoding='utf-8'))
        file = open(output_dir + file_name, 'w', encoding='utf-8')
        # format: [[word_list], label_name, local_counter, offset, skipped_words, score, is_normal, first_match, skip, matched_words]
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
                                elif len(item) < len(item2) and item in temp_set:
                                    temp_set.remove(item)
            item_set = temp_set
            # skipped targets due to repetition
            num_of_skipped_targets += (len(item_list)-len(item_set))
            target_set = set()
            for item in item_set:
                # Nan symbol means the target is always skippable
                if isinstance(item, float) and math.isnan(item):
                    target_list.append([['NaN'], json_fields[field][1], 0, 0, 0, -100, 0, -1, 1, 0])
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

                if json_fields[field][1] == 'ZIP':  # ZIP code needs additional splitting
                    target = target[0].split('-')
                if not target_set.__contains__(' '.join(target)):
                    target_list.append([target, json_fields[field][1], 0, 0, 0, 0, 0, -1, 0, 0])
                    target_set.add(' '.join(target))
        # for each target, we create a set to store unique word chains labelled with that target
        # we use this information in local reports
        list_of_dicts_of_labelled_chains = []
        for t in target_list:
            list_of_dicts_of_labelled_chains.append(defaultdict(int))
        # a secondary copy of target list is maintaining the actual matching numbers
        temp_target_list = copy.deepcopy(target_list)
        ################################################
        # MATCHING TARGETS WITH WORDS FROM THE DOCUMENT
        ################################################
        for dictionary_element in keys_to_pages:
            num_of_words += 1  # for every new line that separates pages
            if dictionary_element in company_data['content']:
                for content in company_data['content'][dictionary_element]:
                    for sentence in tokenizer.tokenize(content):
                        word_chain = []
                        # resetting constants instead of deepcopying
                        for temp_target in temp_target_list:
                            temp_target[3] = 0
                            temp_target[4] = 0
                            temp_target[6] = 0
                            temp_target[7] = -1
                            temp_target[9] = 0
                            if temp_target[0][0] == 'NaN':
                                temp_target[5] = -100
                                temp_target[8] = 1
                            else:
                                temp_target[5] = 0
                                temp_target[8] = 0
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
                                    if not case_sensitive.__contains__(target[1]) and (current_word[:1].isupper() or not current_word[:1].isalpha()):
                                        current_word = current_word.upper()
                                    if current_word[-1:] == '.' and len(current_word) > 1:
                                        current_word = current_word[:-1]
                                    if current_word.lower() in switchable_words and target[1] != 'STATE':
                                        if not case_sensitive.__contains__(target[1]):
                                            current_word = switchable_words[current_word.lower()].upper()
                                        else:
                                            current_word = switchable_words[current_word.lower()]
                                    # usually every word is unique in any given target, so if one matching session finds
                                    # the same word twice, that means overlapping named entities being merged, which is
                                    # undesired, so we set this target to skip mode
                                    if position+target[3] > 0:
                                        partial_target = target[0][:position+target[3]]
                                        switched_words = partial_target
                                        for p in range(0, len(partial_target)):
                                            if partial_target[p].lower() in switchable_words and target[1] != 'STATE':
                                                if not case_sensitive.__contains__(target[1]):
                                                    switched_words[p] = switchable_words[partial_target[p].lower()].upper()
                                                else:
                                                    switched_words[p] = switchable_words[partial_target[p].lower()]
                                        if current_word in switched_words and current_word.lower() not in too_common_words:
                                            target[8] = 1  # if you find the same word that has already been matched, that probably means, that the matching should start from the second occurrence of said word
                                            if i < len(word_list):
                                                next_word = word_list[i]
                                            else:
                                                next_word = ''
                                            aborted_matchings.add('Word chain: '+' '.join(word_chain)+'  Next word: '+next_word+'  Target: '+' '.join(target[0]))
                                            continue
                                    # if a word matches from the document with a word from the target, matched == True
                                    matched = False
                                    # helps tracking uneven pairs when a too common word would make otherwise 'even' pairs to be marked as uneven (target[6])
                                    too_common_found = False
                                    for j in range(0, num_of_target_skip+1):
                                        # we only search for matching until we find the first one
                                        if not matched:
                                            # normal position is modified by the accumulated offset and j, which allows holes
                                            mod_position = position+target[3]+j
                                            # word from current target at current modified position
                                            target_word = ""
                                            if len(target[0]) > mod_position:  # preventing out of range indexing
                                                target_word = target[0][mod_position]
                                                if target_word.lower() in switchable_words and target[1] != 'STATE':
                                                    if not case_sensitive.__contains__(target[1]):
                                                        target_word = switchable_words[target_word.lower()].upper()
                                                    else:
                                                        target_word = switchable_words[target_word.lower()]
                                                # short words need to be identical for matching
                                                # sim_threshold defines the maximum acceptable length difference
                                                if len(target_word) <= 4:
                                                    sim_threshold = 0
                                                else:
                                                    sim_threshold = 0  # the input already has no excess characters, so we don't need to allow even a letter difference when comparing
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
                                                            too_common_found = True
                                            # out of range indexing could happen, but it's necessary to allow word skipping (regardless of stretching out of the length of the target)
                                            if matched:
                                                in_chain = True
                                                target[3] += j  # increasing the offset appropriately so the modified position will mark the next word in the target
                                                target[4] = 0  # since we have matched a target, the number of skipped words at the end of the chain is 0
                                                target[5] += 1 - j*0.55  # increase score because successful matching, but also decrease for skipping words in target
                                                target[9] += 1  # increase the number of matched words
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
                                            if target[3] != 0 and target[6] == 0 and target_word.lower() not in too_common_words and not too_common_found:
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

                            ##############################
                            # PRINTING THE LABELLED WORDS
                            ##############################
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
                                min_length = 100
                                # use the shortest target of the highest score
                                for j in range(0, len(temp_target_list)):
                                    t = temp_target_list[j]
                                    if t[5] == max_score and len(t[0]) < min_length:
                                        min_length = len(t[0])
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
                                string_to_write = chain_label+': '+prev_word+' <| '+' '.join(word_chain)+' |> '+next_word+' -- ['+' '.join(temp_target_list[max_index][0])+']  {'+file_name+'}'
                                if len(word_chain) == 1:
                                    # special case where the Unit label is probably not appropriate,
                                    # because the chain is small and the word is small and it is not a STATE
                                    # additionally, if the target too long, matching only once is not enough
                                    # also, a number alone which is not a ZIP code, is also discarded
                                    if len(word_chain[0]) <= 3 and chain_label != 'STATE' and chain_label != 'ZIP' or len(target[0]) > 2 or word_chain[0].isdigit() and chain_label != 'ZIP':
                                        file.write(word_chain[0] + " O" + '\n')
                                        declined_matches_dict[string_to_write+' REASON:TOO_SMALL_OR_TOO_SHORT'] += 1
                                    # we do not allow single words to match with names or company names or addresses, because
                                    # these usually wrong (e.g. matching last names generally doesn't mean it's the same person)
                                    elif denied_unit_labels.__contains__(chain_label) and len(target[0]) > 1:
                                        file.write(word_chain[0] + " O" + '\n')
                                        declined_matches_dict[string_to_write+' REASON:ONLY_ONE_WORD'] += 1
                                    else:
                                        # unit label is used, because this word chain is only 1 word long
                                        file.write(word_chain[0] + " U-" + chain_label + '\n')
                                        target_list[max_index][2] += 1  # statistic for reports
                                        if target[6] == 1:
                                            uneven = True
                                        not_outside = True
                                    i -= 1  # chain breaking words may start a new chain, so we check it again (after resetting the local variables)
                                elif len(word_chain) > 1:
                                    # limit the number of lowercase words for better performance
                                    capital_letters = 0
                                    for doc_word in word_chain:
                                        if doc_word.isdigit() or doc_word[:1].isupper() or too_common_words.__contains__(doc_word.lower()) or len(doc_word) == 1:
                                            capital_letters += 1
                                    # number of matched words must be at least half of the length of the target excluding too common words, plus 1 (with default accept ratio)
                                    target_length = 0
                                    for w in target[0]:
                                        if w not in too_common_words and w.lower() != '\'s':  # tokenizer usually separates 's into ' and s and this makes the matching to fail
                                            target_length += 1
                                    if target[9] >= (target_length+1)/accept_ratio and capital_letters >= len(word_chain)-allowed_lowercase_words * (len(word_chain)/3) and capital_letters >= 1:
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
                                        declined_matches_dict[string_to_write+' REASON:TOO_SHORT_OR_FEW_CAPITALS'] += 1
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
                                # resetting constants instead of deepcopying
                                for temp_target in temp_target_list:
                                    temp_target[3] = 0
                                    temp_target[4] = 0
                                    temp_target[6] = 0
                                    temp_target[7] = -1
                                    temp_target[9] = 0
                                    if temp_target[0][0] == 'NaN':
                                        temp_target[5] = -100
                                        temp_target[8] = 1
                                    else:
                                        temp_target[5] = 0
                                        temp_target[8] = 0
                                chain_label = 'New'
                                position = 0
                            i += 1
                        file.write('\n')  # marks the end of a sentence
            file.write('\n')  # marks the end of a page, even if it's empty

        file.close()
        print("Labelling done in: " + file_name + '\n')
        if local_report_enabled:
            labelling_report = open(report_dir + file_name, 'w', encoding='utf-8')
            labelling_report.write('Labelling in: ' + file_name + '\n\n')
        ##################################
        # CREATING REPORTS AND STATISTICS
        ##################################
        at_least_once = set()  # set of labels that were used at least once in the current document
        nan_set = set()  # set of labels that had NaN in the json field
        useless_page = True
        # local report for every file
        for i in range(0, len(target_list)):
            t = target_list[i]
            if t[0][0] != 'NaN':
                if local_report_enabled:
                    report = 'Labelled ' + str(t[2]) + ' word chains with ' + t[1] + ', pattern used: ' + ' '.join(t[0]) + '.'
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
                if local_report_enabled:
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
        if local_report_enabled:
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
            if local_report_enabled:
                labelling_report.write('This document was useless.\n')
        if almost_no_match:
            almost_useless_pages.write(file_name + '\n')
            if local_report_enabled:
                labelling_report.write('This document was almost useless.\n')
        # extra data to check if every word is written in the labelled file
        if local_report_enabled:
            labelling_report.write('Number of processed words and empty lines: '+str(num_of_words-2)+'\n')
            labelling_report.close()

        # global report after a given amount of files or when the processing of files is finished
        if periodic_report_progress == global_report_interval or progress_count == to_num:
            periodic_report_progress = 0
            interval_index = math.ceil(progress_count/global_report_interval)
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
            every_pair_set.write('Format: LABEL: Previous <| Matched words |> Next  --  [Pattern used]  {Original file}  Number of occurrence\n\n')
            for t, tt in pair_dict.items():
                every_pair_set.write(t+'    Occurrence: '+str(tt)+'\n\n')
            every_pair_set.close()
            pair_dict.clear()

            declined_matches = open(global_report_dir+'declined_pairs\\declined_pairs_'+str(interval_index)+'.txt', 'w', encoding='utf-8')
            declined_matches.write('Format: LABEL: Previous <| Matched words |> Next  --  [Pattern used]  {Original file}  Number of occurrence\n\n')
            for t, tt in declined_matches_dict.items():
                declined_matches.write(t+'  Occurrence: '+str(tt)+'\n\n')
            declined_matches.close()
            declined_matches_dict.clear()

            uneven_pairs_set = open(global_report_dir+'uneven_pairs\\uneven_pairs_'+str(interval_index)+'.txt', 'w', encoding='utf-8')
            uneven_pairs_set.write('Format: LABEL: Previous <| Matched words |> Next  --  [Pattern used]  {Original file}  Number of occurrence\n\n')
            for t, tt in uneven_dict.items():
                uneven_pairs_set.write(t+'  Occurrence: '+str(tt)+'\n\n')
            uneven_pairs_set.close()
            uneven_dict.clear()

            aborted_word_chains = open(global_report_dir+'aborted_matchings\\aborted_matchings_'+str(interval_index)+'.txt', 'w', encoding='utf-8')
            aborted_word_chains.write('Format: Word chain: Actual words in document  Next word: This word caused the abort  Target: target\n\n')
            for t in aborted_matchings:
                aborted_word_chains.write(t+'\n\n')
            aborted_matchings.clear()
            aborted_word_chains.close()

useless_pages.close()
almost_useless_pages.close()
