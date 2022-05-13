'''
Created on Nov 8, 2019
Sample structure of run file run.py

@author: cxchu
'''

import sys

import spacy
from tqdm import tqdm

from main import TRAIN_FILEPATH
from predictors import SimplePredictor


def your_typing_function(input_file, result_file):
    '''
    This function reads the input file (e.g. test.tsv) and does typing all given entity mentions.
    The results is saved in the result file (e.g. results.tsv)
    '''
    nlp = spacy.load('en_core_web_sm')
    predictor = SimplePredictor(nlp=nlp, simplified_type_system=True)
    predictor.train(train_filepath=TRAIN_FILEPATH)

    with open(input_file, 'r', encoding='utf8') as fin, open(result_file, 'w', encoding='utf8') as fout:
        for line in tqdm(fin):
            id, entity, sentence = line.rstrip().split("\t")

            types = predictor.predict_type(sentence=sentence, named_entity=entity)
            fout.write(id + "\t" + str(types) + "\n")
    print(predictor.successful_attempts, predictor.failed_attempts, predictor.no_patterns, predictor.multiple_patterns)


'''
*** other code if needed
'''

'''
main function
'''
if __name__ == '__main__':
    #sys.argv = ['test', 'test.tsv', 'results.tsv']
    #sys.argv = ['test', 'train-test.tsv', 'train-results.tsv']
    if len(sys.argv) != 3:
        raise ValueError('Expected exactly 2 argument: input file and result file')
    your_typing_function(sys.argv[1], sys.argv[2])
