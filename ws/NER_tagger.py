import json
import nltk.data
import copy
from nltk.tag import StanfordNERTagger

import os
java_path = "C:/Program Files/Java/jre1.8.0_45/bin/java.exe"
os.environ['JAVAHOME'] = java_path

input_dir = '..\\data_sample\\'
output_dir = '..\\golden_labels\\'

list_file = '..\\list.txt'
list_of_file_names = open(list_file, 'r')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
st = StanfordNERTagger("../ws/stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz", "../ws/stanford_ner/stanford-ner.jar", encoding='utf-8')

for file_name in list_of_file_names:
    file_name = file_name.rstrip()  # remove whitespace ending
    print("Processing: "+file_name)
    company_data = json.load(open(input_dir+file_name, 'r', encoding='utf-8'))
    file = open(output_dir+file_name, 'w', encoding='utf-8')

    target_list = []
    for item in company_data['Company Contact']:
        contact = nltk.word_tokenize(item)
        target_list.append([contact, 'CONTACT'])
    for item in company_data['Company Name']:
        company_name = nltk.word_tokenize(item)
        target_list.append([company_name, 'COMPANY'])
    for item in company_data['State']:
        state = nltk.word_tokenize(item)
        target_list.append([state, 'STATE'])
    for item in company_data['City']:
        city = nltk.word_tokenize(item)
        target_list.append([city, 'CITY'])
    for item in company_data['Address']:
        address = nltk.word_tokenize(item)
        target_list.append([address, 'ADDRESS'])
    for item in company_data['Zip']:
        zip_code = nltk.word_tokenize(item)
        target_list.append([zip_code, 'ZIP'])

    for content in company_data['content']:
        for sentence in tokenizer.tokenize(content):
            word_chain = []
            temp_target_list = copy.deepcopy(target_list)
            chain_label = 'New'
            position = 0
            word_list = nltk.word_tokenize(sentence)
            i = 0
            while i < len(word_list):
                print('current word '+word_list[i])
                success = False
                for target in temp_target_list:
                    if position < len(target[0]) and target[1] != 'Skip':  # prevents out of range indexing
                        if target[0][position] == word_list[i] or target[0][position][:1] == word_list[i][:1] and position > 0:
                            success = True
                            chain_label = target[1]
                        else:
                            target[1] = 'Skip'
                if success:
                    position += 1
                    word_chain.append(word_list[i])

                if not success:
                    if len(word_chain) == 1:
                        file.write(word_chain[0]+" U-"+chain_label+'\n')
                        i -= 1  # chain breaking words maybe start a new chain
                    elif len(word_chain) > 1:
                        file.write(word_chain[0]+" B-"+chain_label+'\n')
                        for j in range(1, len(word_chain)-1):
                            file.write(word_chain[j]+" I-"+chain_label+'\n')
                        file.write(word_chain[-1]+" L-"+chain_label+'\n')
                        i -= 1  # chain breaking word
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
    print("Labelling done in: "+file_name)
