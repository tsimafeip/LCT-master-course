import json
from typing import List
from collections import Counter

from spacy import Language

from helper import get_unique_types, read_train_file, check_pos_for_types


class SimplePredictor:
    """Simple predictor based on is-a pattern, which never predicts multiple types."""

    def __init__(self, nlp: Language):
        self.known_types = set()
        self.type_pos_patterns = []
        self.nlp = nlp
        self._phrases_to_check = {' is a ', ' is an ', ' is the ', ' are a ', ' are the ', ' are an '}

    def _predict_type_pos_matching_with_fallback(self, sentence: str):
        return self._predict_type_pos_matching(sentence=sentence, fallback_to_substr_matching=True)

    def _predict_type_pos_matching(self, sentence: str, fallback_to_substr_matching: bool = False) -> List[str]:
        discovered_patterns = [phrase for phrase in self._phrases_to_check if (phrase in sentence)]

        if not discovered_patterns or len(discovered_patterns) > 2:
            return self._predict_type_substr_matching(sentence=sentence) if fallback_to_substr_matching else []

        pattern_phrase = discovered_patterns[0]

        parsed_sentence = self.nlp(sentence)
        after_phrase = []
        for i in range(len(parsed_sentence) - 1):
            if f" {parsed_sentence[i].text} {parsed_sentence[i + 1].text} " == pattern_phrase:
                after_phrase = parsed_sentence[i + 2:]

        pos_pattern_maxlen = 5 # max([len(type_pos_pattern) for type_pos_pattern in self.type_pos_patterns])

        after_phrase_tokens_list = [after_phrase]
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

        # 3, 2, 1
        predictions = []
        for after_phrase_tokens in after_phrase_tokens_list:
            # overrides possible
            pos_pattern_to_text = {}

            # no overrides
            text_to_pos_pattern = {}
            for pos_pattern_len in range(pos_pattern_maxlen, 0, -1):
                for i in range(pos_pattern_len):
                    tokens = after_phrase_tokens[i:pos_pattern_len]
                    pos_pattern = tuple([token.pos_ for token in tokens])
                    text = " ".join([token.text for token in tokens])
                    lemma_text = " ".join([token.lemma_ for token in tokens])

                    pos_pattern_to_text[pos_pattern] = lemma_text
                    text_to_pos_pattern[lemma_text] = pos_pattern

            for trained_pos_pattern in self.type_pos_patterns:
                prediction = pos_pattern_to_text.get(trained_pos_pattern, None)
                if prediction in self.known_types:
                    predictions.append(prediction)
                    break

        if predictions:
            return predictions

        return self._predict_type_substr_matching(sentence=sentence) if fallback_to_substr_matching else []

    def _predict_type_substr_matching(self, sentence: str):
        types_in_sentence = [known_type for known_type in self.known_types if known_type in sentence]

        return types_in_sentence

    def train(self, train_filepath: str, load_pos_from_file: bool = True, pos_patterns_number: int = 8):
        if not load_pos_from_file:
            original_types, types_pos = check_pos_for_types(nlp=self.nlp, source_filepath=train_filepath)
            with open('types.json', 'w') as types_f, open('types_pos.json', 'w') as types_pos_f:
                json.dump(original_types, types_f)
                json.dump(types_pos, types_pos_f)

        with open('types.json') as types_f, open('types_pos.json') as types_pos_f:
            original_types = json.load(types_f)
            types_pos = [tuple(types_pos_list) for types_pos_list in json.load(types_pos_f)]
        self.known_types = set([" ".join(type_words) for type_words in original_types])

        # types_pos is top-8 popular pos patterns sorted in descendant order by length
        # to match the most broad patterns first
        type_pos_patterns = sorted(Counter(types_pos).most_common(pos_patterns_number),
                                   key=lambda x: (-len(x[0]), -x[1]))

        # x[0] = type_pos_tuple, x[1] = type_pos_count
        self.type_pos_patterns = [pattern for pattern, count in type_pos_patterns]

        for prediction_method in \
                [self._predict_type_substr_matching,
                 self._predict_type_pos_matching,
                 self._predict_type_pos_matching_with_fallback]:

            true_positives = false_positives = false_negatives = 0

            for i, named_entity, gold_types, sentence in read_train_file(train_filepath=train_filepath):
                # predicted_types = self._predict_type(named_entity=named_entity, sentence=sentence)
                predicted_types = prediction_method(sentence=sentence)
                for gold_type in gold_types:
                    true_positives += (gold_type in predicted_types)
                    false_negatives += (gold_type not in predicted_types)

                for predicted_type in predicted_types:
                    false_positives += (predicted_type not in gold_types)

            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f_score = 2 * precision * recall / (precision + recall)
            print('Train precision, recall and f-score: {:.3f}, {:.3f}, {:.3f}'.format(precision, recall, f_score))

    def test(self):
        pass
