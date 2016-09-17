import json
import nltk.data
from nltk.tag import StanfordNERTagger

import os
java_path = "C:/Program Files/Java/jre1.8.0_45/bin/java.exe"
os.environ['JAVAHOME'] = java_path

input_dir = '..\\data_sample\\'
output_dir = '..\\labelled_data\\'

list_file = '..\\list.txt'
list_of_file_names = open(list_file, 'r')

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
st = StanfordNERTagger("../ws/stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz", "../ws/stanford_ner/stanford-ner.jar", encoding='utf-8')

for file_name in list_of_file_names:
    file_name = file_name.rstrip()  # whitespace vegzodes leszedes
    print("Processing: "+file_name)
    company_data = json.load(open(input_dir+file_name, 'r'))
    file = open(output_dir+file_name, 'w')

    for content in company_data['content']:  # vegigmegy a weboldalon talalt szovegegysegeken
        for sentence in tokenizer.tokenize(content):  # mondatokra bontja a szovegegyseget
            label = st.tag(nltk.word_tokenize(sentence))
            label_type = 'N'  # uj mondat elso eleme
            for labelled_token in label:
                if label_type == 'N':  # ha elso elem, csak mentsuk el
                    if labelled_token[1] == 'LOCATION':
                        label_type = 'B-LOC'
                    if labelled_token[1] == 'PERSON':
                        label_type = 'B-PER'
                    if labelled_token[1] != 'LOCATION' and labelled_token[1] != 'PERSON':
                        label_type = 'O'
                    prev_token = labelled_token
                    continue

                    # B - 'beginning'
                    # I - 'inside'
                    # L - 'last'
                    # O - 'outside'
                    # U - 'unit'

                if label_type == 'B-LOC':  # az elozo elem B-LOC volt
                    if labelled_token[1] == 'LOCATION':
                        label_type = 'I-LOC'
                        file.write(prev_token[0]+" B-LOC"+'\n')
                    if labelled_token[1] == 'PERSON':
                        label_type = 'B-PER'
                        file.write(prev_token[0]+" U-LOC"+'\n')
                    if labelled_token[1] != 'LOCATION' and labelled_token[1] != 'PERSON':
                        label_type = 'O'
                        file.write(prev_token[0]+" U-LOC"+'\n')
                    prev_token = labelled_token
                    continue

                if label_type == 'I-LOC':  # az elozo elem I-LOC volt
                    if labelled_token[1] == 'LOCATION':
                        label_type = 'I-LOC'
                        file.write(prev_token[0]+" I-LOC"+'\n')
                    if labelled_token[1] == 'PERSON':
                        label_type = 'B-PER'
                        file.write(prev_token[0]+" L-LOC"+'\n')
                    if labelled_token[1] != 'LOCATION' and labelled_token[1] != 'PERSON':
                        label_type = 'O'
                        file.write(prev_token[0]+" L-LOC"+'\n')
                    prev_token = labelled_token
                    continue

                if label_type == 'B-PER':  # az elozo elem B-PER volt
                    if labelled_token[1] == 'LOCATION':
                        label_type = 'B-LOC'
                        file.write(prev_token[0]+" U-PER"+'\n')
                    if labelled_token[1] == 'PERSON':
                        label_type = 'I-PER'
                        file.write(prev_token[0]+" B-PER"+'\n')
                    if labelled_token[1] != 'LOCATION' and labelled_token[1] != 'PERSON':
                        label_type = 'O'
                        file.write(prev_token[0]+" U-PER"+'\n')
                    prev_token = labelled_token
                    continue

                if label_type == 'I-PER':  # az elozo elem I-PER volt
                    if labelled_token[1] == 'LOCATION':
                        label_type = 'B-LOC'
                        file.write(prev_token[0]+" L-PER"+'\n')
                    if labelled_token[1] == 'PERSON':
                        label_type = 'I-PER'
                        file.write(prev_token[0]+" I-PER"+'\n')
                    if labelled_token[1] != 'LOCATION' and labelled_token[1] != 'PERSON':
                        label_type = 'O'
                        file.write(prev_token[0]+" L-PER"+'\n')
                    prev_token = labelled_token
                    continue

                if label_type == 'O':  # az elozo elem O volt
                    if labelled_token[1] == 'LOCATION':
                        label_type = 'B-LOC'
                        file.write(prev_token[0]+" O"+'\n')
                    if labelled_token[1] == 'PERSON':
                        label_type = 'B-PER'
                        file.write(prev_token[0]+" O"+'\n')
                    if labelled_token[1] != 'LOCATION' and labelled_token[1] != 'PERSON':
                        label_type = 'O'
                        file.write(prev_token[0]+" O"+'\n')
                    prev_token = labelled_token
                    continue

            # utolso elem kiirasa
            if label_type == 'B-LOC':
                file.write(prev_token[0]+" U-LOC"+'\n')
            if label_type == 'I-LOC':
                file.write(prev_token[0]+" L-LOC"+'\n')
            if label_type == 'B-PER':
                file.write(prev_token[0]+" U-PER"+'\n')
            if label_type == 'I-PER':
                file.write(prev_token[0]+" L-PER"+'\n')
            if label_type == 'O':
                file.write(prev_token[0]+" O"+'\n')
            file.write('\n')  # mondathatar

    file.close()
    print("Labelling done in: "+file_name)
