import json
import os.path
from typing import List
from collections import Counter

from spacy import Language
from spacy.tokens import Doc, Span

from helper import read_train_file, check_pos_for_types


class SimplePredictor:
    """Simple predictor based on is-a pattern, which never predicts multiple types."""

    def __init__(self, nlp: Language, max_type_words_count: int = 5):
        self.lemma_type_to_orig = dict()
        self.nlp = nlp
        self._phrases_to_check = {' is a ', ' is an ', ' is the ', ' are a ', ' are the ', ' are an '}
        self.max_type_words_count = max_type_words_count

        # simple statistics for my analysis
        self.no_patterns = 0
        self.multiple_patterns = 0
        self.failed_attempts = 0
        self.successful_attempts = 0

        self._orig_types_filename = 'orig_types.json'
        self._lemma_types_filename = 'lemma_types.json'
        self._pos_tags_filename = 'pos_tags.json'

    def predict_type(self, sentence: str, named_entity: str) -> List[str]:
        return self._predict_type_pos_matching_with_fallback(sentence=sentence)

    def _predict_type_pos_matching_with_fallback(self, sentence: str):
        return self._predict_type_pos_matching(sentence=sentence, fallback_to_substr_matching=True)

    def _get_after_phrases_to_match(self, parsed_sentence: Doc, pattern_phrase: str) -> List[Span]:
        after_phrase = []
        for i in range(len(parsed_sentence) - 1):
            if f" {parsed_sentence[i].text} {parsed_sentence[i + 1].text} " == pattern_phrase:
                after_phrase = parsed_sentence[i + 2:]

        after_phrase_tokens_list = [after_phrase]
        # handle phrases like he is a [songwriter] and [guitar player]
        if 'and' in [token.text for token in after_phrase[:7]]:
            cur_tokens = []
            after_phrase_tokens_list = []

            for token in after_phrase:
                if token.text == 'and':
                    after_phrase_tokens_list.append(cur_tokens)
                    cur_tokens = []
                else:
                    cur_tokens.append(token)
            after_phrase_tokens_list.append(cur_tokens)

        return after_phrase_tokens_list

    @staticmethod
    def _extract_all_possible_subspans(original_span: Span, max_span_len: int) -> List[Span]:
        subspans = []
        for pos_pattern_len in range(max_span_len, 0, -1):
            for i in range(pos_pattern_len):
                subspan = original_span[i:pos_pattern_len]
                subspans.append(subspan)

        return subspans

    def _predict_type_pos_matching(self, sentence: str, fallback_to_substr_matching: bool = False) -> List[str]:
        discovered_patterns = [phrase for phrase in self._phrases_to_check if (phrase in sentence)]

        if not discovered_patterns or len(discovered_patterns) > 2:
            if not discovered_patterns:
                self.no_patterns += 1
            else:
                self.multiple_patterns += 1
            return self._predict_type_substr_matching(sentence=sentence) if fallback_to_substr_matching else []

        pattern_phrase = discovered_patterns[0]
        parsed_sentence = self.nlp(sentence)
        after_phrase_tokens_list = self._get_after_phrases_to_match(parsed_sentence=parsed_sentence,
                                                                    pattern_phrase=pattern_phrase)

        predictions = []
        pos_based_predictions = []
        # looking for exact matching of known types in after phrases
        for after_phrase_tokens in after_phrase_tokens_list:
            subspans = self._extract_all_possible_subspans(original_span=after_phrase_tokens,
                                                           max_span_len=self.max_type_words_count)

            # sort by len of span to match the longest type first
            for subspan in sorted(subspans, key=lambda x: -len(x)):
                potential_type_str = " ".join([token.lemma_.lower() for token in subspan])
                if potential_type_str in self.lemma_type_to_orig:
                    # predict original non-lemmatized and non-lowercased type
                    predictions.append(self.lemma_type_to_orig[potential_type_str])
                    # breaking help to match only the best type, like 'American rapper'
                    # instead of additionally matching 'rapper'
                    break

            for subspan in sorted(subspans, key=lambda x: -len(x)):
                cur_type_pos_pattern = tuple([token.pos_ for token in subspan])

                if cur_type_pos_pattern in self.type_pos_patterns:
                    potential_type_str = " ".join(token.text for token in subspan)
                    pos_based_predictions.append(potential_type_str)
                    break

        if predictions:
            self.successful_attempts += 1
            return predictions
        elif pos_based_predictions:
            self.successful_attempts += 1
            return pos_based_predictions

        self.failed_attempts += 1
        return self._predict_type_substr_matching(sentence=sentence) if fallback_to_substr_matching else []

    def _predict_type_substr_matching(self, sentence: str) -> List[str]:
        lemma_sentence = " ".join(token.lemma_ for token in self.nlp(sentence))

        lemma_types_in_sentence = [known_type for known_type in self.lemma_type_to_orig if known_type in lemma_sentence]
        types_in_sentence = [self.lemma_type_to_orig[lemma_type] for lemma_type in lemma_types_in_sentence]

        return types_in_sentence

    def _preprocess_data(self, train_filepath: str, force: bool = False):
        required_files = [self._orig_types_filename, self._lemma_types_filename, self._pos_tags_filename]
        if not all(os.path.exists(filepath) for filepath in required_files):
            force = True

        filemode = 'w' if force else 'r'

        with open(self._orig_types_filename, filemode) as types_f, \
                open(self._lemma_types_filename, filemode) as lemma_types_f, \
                open(self._pos_tags_filename, filemode) as types_pos_f:
            if force:
                original_types, lemma_types, types_pos = check_pos_for_types(nlp=self.nlp,
                                                                             source_filepath=train_filepath)
                json.dump(original_types, types_f)
                json.dump(lemma_types, lemma_types_f)
                json.dump(types_pos, types_pos_f)
            else:
                original_types = json.load(types_f)
                lemma_types = json.load(lemma_types_f)
                types_pos = [tuple(types_pos_list) for types_pos_list in json.load(types_pos_f)]

        return original_types, lemma_types, types_pos

    def train(self, train_filepath: str, force_preprocessing: bool = False, pos_patterns_number: int = 1000):
        orig_types, lemma_types, pos_tags = self._preprocess_data(train_filepath=train_filepath,
                                                                  force=force_preprocessing)

        orig_types = list(map(" ".join, orig_types))
        lowercased_lemma_types = []
        for lemma_type in lemma_types:
            lowercased_lemma_type = [type_word.lower() for type_word in lemma_type]
            lowercased_lemma_type_str = " ".join(lowercased_lemma_type)
            lowercased_lemma_types.append(lowercased_lemma_type_str)

        self.lemma_type_to_orig = dict(zip(lowercased_lemma_types, orig_types))

        # types_pos is top-8 popular pos patterns sorted in descendant order by length
        # to match the most broad patterns first
        type_pos_patterns = sorted(Counter(pos_tags).most_common(pos_patterns_number),
                                   key=lambda x: (-len(x[0]), -x[1]))

        # x[0] = type_pos_tuple, x[1] = type_pos_count
        self.type_pos_patterns = [pattern for pattern, count in type_pos_patterns]

    def evaluate_on_train_data(self, train_filepath: str):
        # methods = [self._predict_type_substr_matching,
        #          self._predict_type_pos_matching,
        #          self._predict_type_pos_matching_with_fallback]
        for prediction_method in [self._predict_type_pos_matching_with_fallback]:

            true_positives = false_positives = false_negatives = 0

            for i, named_entity, gold_types, sentence in read_train_file(train_filepath=train_filepath):
                # predicted_types = self._predict_type(named_entity=named_entity, sentence=sentence)
                predicted_types = prediction_method(sentence=sentence)
                if predicted_types:
                    for gold_type in gold_types:
                        true_positives += (gold_type in predicted_types)
                        false_negatives += (gold_type not in predicted_types)

                    for predicted_type in predicted_types:
                        false_positives += (predicted_type not in gold_types)

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f_score = 2 * precision * recall / (precision + recall)
            print('Train precision, recall and f-score: {:.3f}, {:.3f}, {:.3f}'.format(precision, recall, f_score))
