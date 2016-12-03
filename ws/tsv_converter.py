import os
##from  subprocess import Popen
#import subprocess

input_dir = '..\\stanford_ner\\crawl_silver_test\\crawl_silver_labels\\'
output_dir = '..\\\\stanford_ner\\\\silver_tsv\\\\'
#script_dir = '..\\stanford_ner\\'
test_dir = '..\\stanford_ner\\test_tsv\\'
list_of_file_names = os.listdir(input_dir)

if not os.path.exists(output_dir[:-2]):
    os.makedirs(output_dir[:-2])

if not os.path.exists(test_dir[:-1]):
    os.makedirs(test_dir[:-1])

output_name = output_dir + 'training_data'
total_output_file = open(output_name, 'w', encoding='utf-8')
test_output_file = open(test_dir + 'test_data', 'w', encoding='utf-8')

keys_to_pages = ['index', 'contact', 'partners', 'products']
USE_THIRD_COLUMN = False
progress_count = 0
from_num = 0
to_num = 50  # default: len(list_of_file_names)
above_this_is_tested = 40
for file_name in list_of_file_names:
    progress_count += 1
    file_to_write_into = total_output_file
    if from_num < progress_count <= to_num:
        file_name = file_name.rstrip()
        input_file = open(input_dir + file_name, 'r', encoding='utf-8')
        last_line = ''
        if progress_count < above_this_is_tested:
            file_to_write_into = total_output_file
            append_page_type = True
        else:
            file_to_write_into = test_output_file
            append_page_type = False
        key_index = 0
        for line in input_file:
            if last_line == '\n' and line == '\n' and key_index < 3:
                key_index += 1
            last_line = line
            if line == '\n':
                file_to_write_into.write('\n')
                continue
            line = line.rstrip()
            separated = line.split(' ')
            if separated[1] != 'O':
                un_bilou = separated[1].split('-')
                label = un_bilou[1]
            else:
                label = 'O'
            # label remapping option
            if append_page_type and USE_THIRD_COLUMN:
                file_to_write_into.write(separated[0]+'\t'+label+'\t'+keys_to_pages[key_index]+'\n')
            else:
                file_to_write_into.write(separated[0]+'\t'+label+'\n')
        if last_line != '\n':
            file_to_write_into.write('\n')
            file_to_write_into.write('\n')
    file_to_write_into.flush()

total_output_file.close()
test_output_file.close()

'''
prop_file = open(output_dir + '..\\classifier.prop', 'w', encoding='utf-8')
prop_file.write('trainFile = '+output_name+'\n')
prop_file.write(
    'serializeTo = ner-model.ser.gz\n' +
    'map = word=0,answer=1\n' +
    'qnSize=10\n' +  # quasi-Newton optimizer (L-BFGS) - Number of past guesses
    'saveFeatureIndexToDisk=true\n' +  # feature names aren't actually needed while the core model estimation (optimization) code is running
    'printFeatures=false\n' +  # see all the features generated - Options that generate huge numbers of features include useWordPairs and useNGrams when maxNGramLeng is a large number.
    'useObservedSequencesOnly=true\n' +  # This makes it so that you can only label adjacent words with label sequences that were seen next to each other in the training data.
    'featureDiffThresh=0.05\n' +  # In training, CRFClassifier will train one model, drop all the features with weight (absolute value) beneath the given threshold, and then train a second model.
    'useClassFeature=true\n' +
    'useWord=true\n' +
    'useNGrams=true\n' +
    'noMidNGrams=true\n' +
    'useDisjunctive=true\n' +
    'maxNGramLeng=6\n' +
    'usePrev=true\n' +
    'useNext=true\n' +
    'useSequences=true\n' +
    'usePrevSequences=true\n' +
    'maxLeft=1\n' +  # the order of the CRF
    'useTypeSeqs=true\n' +
    'useTypeSeqs2=true\n' +
    'useTypeySequences=true\n' +
    'wordShape=chris2useLC\n'
)
prop_file.close()
'''
#p = subprocess.Popen(script_dir+"train_classifier.bat", shell=True)
#stdout, stderr = p.communicate()
