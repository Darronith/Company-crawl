import json
import nltk.data
from nltk.tag import StanfordNERTagger
import os

java_path = "C:/Program Files/Java/jre1.8.0_45/bin/java.exe"
os.environ['JAVAHOME'] = java_path

# nltk.internals.config_java("C:/Program Files/Java/jdk1.7.0_17/bin/java.exe")

file_name = '1-800-bakery.com'
input_dir = '..\\data_sample\\'
output_dir = '..\\labelled_data\\'

company_data = json.load(open(input_dir+file_name, 'r'))

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
st = StanfordNERTagger("../ws/stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz", "../ws/stanford_ner/stanford-ner.jar", encoding='utf-8')

file = open(output_dir+file_name,'w')

for content in company_data['content']:  # vegigmegy a weboldalon talalt szovegegysegeken
    for sentence in tokenizer.tokenize(content):  # mondatokra bontja a szovegegyseget
        label = st.tag(nltk.word_tokenize(sentence))
        for labelled_token in label:
            print(labelled_token[0]+" "+labelled_token[1])
            file.write(labelled_token[0]+" "+labelled_token[1]+'\n')
        print('') #mondathatar
        file.write('\n')

file.close()

"""
import nltk
nltk.download('all')
"""


