import os


input_dir = '..\\stanford_ner\\test\\'
output_dir = '..\\stanford_ner\\'

list_of_file_names = os.listdir(input_dir)

test_file_csv = open(output_dir + 'test_file_list.csv', 'w', encoding='utf-8')
test_csv_string = ''
first = True

for file_name in list_of_file_names:
    if not first:
        test_file_csv.write(',')
        test_csv_string += ','
    test_file_csv.write('test\\\\'+file_name)
    test_csv_string += ('test\\\\'+file_name)
    test_file_csv.flush()
    first = False
test_file_csv.close()

test_script = open(output_dir + 'test_classifier.bat', 'w', encoding='utf-8')

test_script.write('java -cp stanford_ner\stanford-ner.jar;stanford_ner\lib\* edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ner-model.ser.gz -testFiles ')
test_script.write(test_csv_string+' 1>matching_log.txt 2>test_summary.txt\n\n')
test_script.write('pause')
