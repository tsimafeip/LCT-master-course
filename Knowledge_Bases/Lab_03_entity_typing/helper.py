import ast
from typing import List, Tuple, Generator

from spacy import Language
from tqdm import tqdm


# generic type Generator[yield_type, send_type, return_type]
def read_train_file(train_filepath: str) -> Generator[Tuple[int, str, List[str], str], None, None]:
    """Returns list of sample_index, named_entity, types, sentence."""
    with open(train_filepath) as input_f:
        for i, line in tqdm(enumerate(input_f)):
            named_entity, types, *sentences = line.strip().split('\t')

            if len(sentences) > 1:
                # in fact, replace extra tab in sentence by space
                sentences = [" ".join(sentences)]

            sentence = sentences[0] if sentences else ""
            types = ast.literal_eval(types)

            yield i, named_entity, types, sentence


def get_unique_types(source_filepath: str):
    type_set = set()
    for i, named_entity, types, sentence in read_train_file(train_filepath=source_filepath):
        type_set.update(set(types))

    return type_set


def check_entity_in_sentences_train(train_filepath: str, lowercase_test_entity_if_needed: bool = False):
    entity_in_sentence_count = global_sentence_count = 0

    for i, named_entity, types, sentence in read_train_file(train_filepath=train_filepath):
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

    for i, named_entity, types, sentence in read_train_file(train_filepath=source_filepath):
        for type in types:
            type_in_sentence_count += (type in sentence)
            global_types_count += 1

    print('Percentage of types in train sentences:', type_in_sentence_count / global_types_count)


def check_pos_for_types(nlp: Language, source_filepath: str):
    original_types = []
    lemma_types = []
    pos_tags = []

    for i, named_entity, types, sentence in read_train_file(train_filepath=source_filepath):
        for type in types:
            doc = nlp(type)
            pos_tags.append(tuple([token.pos_ for token in doc]))
            original_types.append(tuple([token.text for token in doc]))
            lemma_types.append(tuple([token.lemma_ for token in doc]))

    return original_types, lemma_types, pos_tags
