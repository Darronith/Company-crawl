import os


input_dir = '..\\stanford_ner\\crawl_silver_test\\crawl_silver_labels\\'
output_dir = '..\\stanford_ner\\silver_tsv\\'
list_of_file_names = os.listdir(input_dir)

if not os.path.exists(output_dir[:-1]):
    os.makedirs(output_dir[:-1])

csv_string = ''
file_csv = open(output_dir + '..\\file_list.csv', 'w', encoding='utf-8')

first = True
progress_count = 0
from_num = 0
to_num = 10  # default: len(list_of_file_names)
for file_name in list_of_file_names:
    progress_count += 1
    if from_num < progress_count <= to_num:
        if not first:
            file_csv.write(',')
            csv_string += ','
        file_csv.write('silver_tsv\\\\'+file_name)
        csv_string += ('silver_tsv\\\\'+file_name)
        file_csv.flush()
        first = False

        file_name = file_name.rstrip()
        print("Processing: " + file_name)
        input_file = open(input_dir + file_name, 'r', encoding='utf-8')
        output_file = open(output_dir + file_name, 'w', encoding='utf-8')
        for line in input_file:
            if line == '\n':
                output_file.write('\n')
                continue
            line = line.rstrip()
            separated = line.split(' ')
            if separated[1] != 'O':
                un_bilou = separated[1].split('-')
                label = un_bilou[1]
            else:
                label = 'O'
            # label remapping option
            output_file.write(separated[0]+'\t'+label+'\n')
    output_file.close()

file_csv.close()

prop_file = open(output_dir + '..\\classifier.prop', 'w', encoding='utf-8')
prop_file.write('trainFileList = ')
prop_file.write(csv_string+'\n')
prop_file.write(
    'serializeTo = ner-model.ser.gz\n' +
    'map = word=0,answer=1\n' +

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
    'maxLeft=1\n' +
    'useTypeSeqs=true\n' +
    'useTypeSeqs2=true\n' +
    'useTypeySequences=true\n' +
    'wordShape=chris2useLC\n'
)
