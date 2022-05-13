import ast
import json
from collections import Counter

from spacy import Language
from tqdm import tqdm


def get_unique_types(source_filepath: str):
    type_set = set()
    with open(source_filepath) as input_f:
        for i, line in enumerate(input_f):

            named_entity, types, *sentences = line.strip().split('\t')

            if len(sentences) > 1:
                # in fact, replace extra tab in sentence by space
                sentences = [" ".join(sentences)]

            # test file does not have sentences
            if len(sentences) == 1:
                sentence = sentences[0]

            # json.loads does not parse single quotations with slashes
            types = ast.literal_eval(types)  # json.loads(types.replace('\'', '"'))
            type_set.update(set(types))

    return type_set


def check_entity_in_sentences_train(train_filepath: str, lowercase_test_entity_if_needed: bool = False):
    entity_in_sentence_count = global_sentence_count = 0

    with open(train_filepath) as input_f:
        for i, line in enumerate(input_f):

            named_entity, types, *sentences = line.strip().split('\t')

            if len(sentences) > 1:
                # in fact, replace extra tab in sentence by space
                sentences = [" ".join(sentences)]

            sentence = sentences[0]

            global_sentence_count += 1
            entity_in_sentence_count += (named_entity in sentence)
            if lowercase_test_entity_if_needed and named_entity not in sentence:
                entity_in_sentence_count += _check_lowercased_ne_presence(i, named_entity, sentence)

    print('Percentage of named entities in train sentences:', entity_in_sentence_count / global_sentence_count)


def check_entity_in_sentences_test(test_filepath: str, lowercase_test_entity_if_needed: bool = False):
    entity_in_sentence_count = global_sentence_count = 0

    with open(test_filepath) as input_f:
        for i, line in enumerate(input_f):

            id, named_entity, *sentences = line.strip().split('\t')

            if len(sentences) > 1:
                # in fact, replace extra tab in sentence by space
                sentences = [" ".join(sentences)]

            sentence = sentences[0]

            global_sentence_count += 1
            entity_in_sentence_count += (named_entity in sentence)
            if lowercase_test_entity_if_needed and named_entity not in sentence:
                entity_in_sentence_count += _check_lowercased_ne_presence(i, named_entity, sentence)

    print('Percentage of named entities in test sentences:', entity_in_sentence_count / global_sentence_count)


def _check_lowercased_ne_presence(sample_index: int, named_entity: str, sentence: str, verbose: bool = False):
    lowercased_named_entity = named_entity[0].lower() + named_entity[1:]
    if verbose and lowercased_named_entity not in sentence:
        print(sample_index, named_entity, sentence)
    return lowercased_named_entity in sentence


def check_type_presence_in_sentences_train(source_filepath: str):
    type_in_sentence_count = global_types_count = 0

    with open(source_filepath) as input_f:
        for i, line in enumerate(input_f):

            named_entity, types, *sentences = line.strip().split('\t')

            if len(sentences) > 1:
                # in fact, replace extra tab in sentence by space
                sentences = [" ".join(sentences)]

            # test file does not have sentences
            if len(sentences) == 1:
                sentence = sentences[0]

            types = ast.literal_eval(types)

            for type in types:
                type_in_sentence_count += (type in sentence)
                global_types_count += 1

    print('Percentage of types in train sentences:', type_in_sentence_count / global_types_count)


def check_pos_for_types(nlp: Language, source_filepath: str):

    types_for_pos = []
    pos_tags = []

    with open(source_filepath) as input_f:
        for i, line in tqdm(enumerate(input_f)):

            _, types, *_ = line.strip().split('\t')

            types = ast.literal_eval(types)

            for type in types:
                doc = nlp(type)
                pos_tags.append(tuple([token.pos_ for token in doc]))
                types_for_pos.append(tuple([token for token in doc]))

    print('POS tags for types:', Counter(pos_tags))