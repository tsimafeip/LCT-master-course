'''
Created on Nov 8, 2019
Sample structure of run file run.py

@author: cxchu
'''

import sys


def your_typing_function(input_file, result_file):
    '''
    This function reads the input file (e.g. test.tsv) and does typing all given entity mentions.
    The results is saved in the result file (e.g. results.tsv)
    '''
    with open(input_file, 'r', encoding='utf8') as fin, open(result_file, 'w', encoding='utf8') as fout:
        for line in fin.readlines():
            comps = line.rstrip().split("\t")
            id = int(comps[0])
            entity = comps[1]
            sentence = comps[2]

            ## radomly return a 7th word in the sentence as a type
            ## else the first word if the length of the sentence < 7
            words = sentence.split(' ')
            types = []
            if len(words) > 7:
                types.append(words[7])
            else:
                types.append(words[0])
            fout.write(str(id) + "\t" + str(types) + "\n")


'''
*** other code if needed
'''

'''
main function
'''
if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('Expected exactly 2 argument: input file and result file')
    your_typing_function(sys.argv[1], sys.argv[2])
