import spacy

from predictors import SimplePredictor
from helper import get_unique_types, check_entity_in_sentences_train, check_entity_in_sentences_test, \
    check_type_presence_in_sentences_train, check_pos_for_types, read_train_file

TRAIN_FILEPATH = 'train.tsv'

def check_types():
    train_types = get_unique_types('train.tsv')
    test_types = get_unique_types('test-groundtruth.tsv')

    print("Types train set:", len(train_types))
    print("Types test set:", len(test_types))
    print("Predictable types in test set:", len(test_types.intersection(train_types)))
    print("Max possible accuracy:", len(test_types.intersection(train_types)) / len(test_types))
    print("New types in test set:", len(test_types - train_types))

def make_test_from_train(train_filepath: str):
    nlp = spacy.load('en_core_web_sm')
    predictor = SimplePredictor(nlp=nlp)
    predictor.train(train_filepath=train_filepath)

    one_type_samples = 0
    for id, named_entity, types, sentence in read_train_file(train_filepath=train_filepath):
        one_type_samples += (len(types) == 1)

    # with open('new-train-test.tsv', 'w') as train_test_f, open('new-train-test-groundtruth.tsv', 'w') as train_gold_f:
    #     prev_successful = 0
    #     for id, named_entity, types, sentence in read_train_file(train_filepath=train_filepath):
    #         one_type_samples += (len(types) == 1)
            # predictor.predict_type(sentence=sentence, named_entity=named_entity)
            # if predictor.successful_attempts == prev_successful + 1:
            #     prev_successful += 1
            #     train_test_f.write(f"{id}\t{named_entity}\t{sentence}\n")
            #     train_gold_f.write(f"{id}\t{str(types)}\n")
    print(one_type_samples)


if __name__ == '__main__':
    #make_test_from_train(train_filepath=TRAIN_FILEPATH)
    check_types()
    check_entity_in_sentences_train(TRAIN_FILEPATH)
    check_entity_in_sentences_train(TRAIN_FILEPATH, True)
    check_type_presence_in_sentences_train(TRAIN_FILEPATH)

    # print('Loading English from spacy...')
    nlp = spacy.load('en_core_web_sm')
    # print('Loading completed.')
    #
    # parsed_sentence = nlp('The Oregon Skyline Trail is a long-distance trail in the Cascade Mountains of Oregon.')
    # pos_tags = [token.pos_ for token in parsed_sentence]
    # predictor = SimplePredictor(nlp=nlp)
    # predictor.train(train_filepath=TRAIN_FILEPATH)
    #predictor.evaluate_on_train_data(train_filepath=TRAIN_FILEPATH)
    check_pos_for_types(nlp, TRAIN_FILEPATH)


