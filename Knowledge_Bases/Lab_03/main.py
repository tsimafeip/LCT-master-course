import spacy

from Lab_03.predictors import SimplePredictor
from helper import get_unique_types, check_entity_in_sentences_train, check_entity_in_sentences_test, \
    check_type_presence_in_sentences_train, check_pos_for_types


TRAIN_FILEPATH = 'setup/train.tsv'

def check_types():
    train_types = get_unique_types('setup/train.tsv')
    test_types = get_unique_types('setup/test-groundtruth.tsv')

    print("Types train set:", len(train_types))
    print("Types test set:", len(test_types))
    print("Predictable types in test set:", len(test_types.intersection(train_types)))
    print("Max possible accuracy:", len(test_types.intersection(train_types)) / len(test_types))
    print("New types in test set:", len(test_types - train_types))


if __name__ == '__main__':
    # check_types()
    # check_entity_in_sentences_train(TRAIN_FILEPATH)
    # check_entity_in_sentences_train(TRAIN_FILEPATH, True)
    # check_type_presence_in_sentences_train(TRAIN_FILEPATH)

    print('Loading English from spacy...')
    nlp = spacy.load('en_core_web_sm')
    print('Loading completed.')

    parsed_sentence = nlp('The Oregon Skyline Trail is a long-distance trail in the Cascade Mountains of Oregon.')
    pos_tags = [token.pos_ for token in parsed_sentence]
    predictor = SimplePredictor(nlp=nlp)
    predictor.train(train_filepath=TRAIN_FILEPATH)
    #check_pos_for_types(nlp, 'setup/train.tsv')


