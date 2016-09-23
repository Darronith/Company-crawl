from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

file = open('..\\sim_test.txt', 'r', encoding='utf-8')
file2 = open('..\\sim_test_output.txt', 'w', encoding='utf-8')
prev = ''
for w in file:
    if prev != '':
        if w != '\n':
            similarity = 'Similarity between  ['+prev.rstrip()+"]  and  ["+w.rstrip()+"]:  "+str(similar(prev.rstrip(), w.rstrip()))
            print(similarity)
            file2.write(similarity+'\n')
        else:
            print()
            file2.write('\n')
            prev = ''
    else:
        prev = w.rstrip()
