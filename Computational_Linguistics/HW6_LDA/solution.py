import os
from collections import Counter, defaultdict
from datetime import datetime

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from tqdm import tqdm

'''

TODO:
1) Run full train and analyse trained data
2) Add numpy arrays instead of lists and check speed changes
3) Write comments
'''

DEFAULT_NUM_OF_TOPICS = 20
DEFAULT_NUM_OF_ITERATIONS = 500
DEFAULT_ALPHA = 0.02
DEFAULT_BETA = 0.1


def plot_topic_distribution(topic_distribution: List[int]):
    plt.bar(range(len(topic_distribution)), topic_distribution)
    plt.xticks(np.arange(0, len(topic_distribution), 1.0))

    plt.title('Topic Distribution')
    plt.xlabel('Topic ID')
    plt.ylabel('Words Per Topic')
    plt.show()


class DataHelper:
    """
    This is a class to simplify access to different document-, word- and topic-related collections.
    """

    def __init__(self, input_filename: Optional[str] = None, num_of_topics: int = DEFAULT_NUM_OF_TOPICS):
        """
        Creates set of data collections and helps to use them consistently later.

        Main collections that we need:
        1) corpus: List[Tuple[int, int]] - List of all possible (word_id, doc_id) pairs of length CORPUS_SIZE.
        Source of 'corpus_id' information.
        2) corpus_topic_distribution: List[int] - List of size CORPUS_SIZE with topic_ids as value.
        Shows current topic distribution for the whole corpus.
        3) vocab: List[str] - List of words, so we can encode each word with one int and work with this int later.
        Source of 'word_id' information.
        4) topic_counter: List[int] - List of topic counts of length NUM_OF_TOPICS.
        'topic_id' is an index in this List.
        5) document_topic_counters: List[List[int]] - List of length NUM_OF_DOCS.
        Each element is List of length NUM_OF_TOPICS and topic_count as values.
        6) word_topic_counters: List[List[int]]- List of length VOCAB_SIZE.
        Each element is List of length NUM_OF_TOPICS and topic_count as values.

        Devivatives and helpers
        7) topic_word_counters: List[List[int]] - List of length NUM_OF_TOPICS.
        Each element is List of legth VOCAB_SIZE and values of topic_word_count.
        8) document_id_to_words: List[List[str]] - List of length NUM_OF_DOCS.
        Each element is List of raw document str words in original order without any grouping.
        9) document_to_word_count: List[int] - List of length NUM_OF_DOCS with document_size as values.
        10) word_to_index_in_vocab: Dict[str, int] - Dict with word_str as key and integer index in vocabulary as value.
        """
        # Init helper 7) collection.
        # Will be populated after training on call of one 'get_most_popular_words_per_topic' method
        self.topic_word_counter = None

        # init empty DataHelper
        if input_filename is None:
            return

        document_id_to_words, vocab_counter = self._get_document_id_to_words_and_vocab(input_filename)

        # initialize main collection 3): vocab array and helper word_to_i dictionary
        self.vocab = vocab_counter
        # init helper 10) collection
        self._word_to_i = {word: i for i, word in enumerate(self.vocab)}

        # init helper collection 8)
        self.document_id_to_words = document_id_to_words
        # initialize main collection 1)
        self.word_document_corpus = self._create_word_document_corpus()
        # initialize helper collection 9)
        self.document_to_word_count = [len(words) for words in self.document_id_to_words]

        # init pseudo-constants
        self.NUM_OF_DOCS = len(self.document_id_to_words)
        self.CORPUS_SIZE = len(self.word_document_corpus)
        self.VOCAB_SIZE = len(self.vocab)
        self.NUM_OF_TOPICS = num_of_topics

        # initialize main collection 5)
        self.document_topic_counters = [[0 for topic_id in range(self.NUM_OF_TOPICS)] for doc_id in
                                        range(self.NUM_OF_DOCS)]
        # initialize main collection 6)
        self.word_topic_counters = [[0 for topic_id in range(self.NUM_OF_TOPICS)] for word_id in range(self.VOCAB_SIZE)]
        # initialize main collection 2)
        self.corpus_topic_distribution = [0 for _ in range(self.CORPUS_SIZE)]
        # initialize main collection 4)
        self.topic_counter = [0 for _ in range(self.NUM_OF_TOPICS)]

        # uniformly assign topic and fill collections 2), 4), 5), 6)
        for corpus_index, (word_id, document_id) in enumerate(self.word_document_corpus):
            random_topic_id = self._get_random_topic_id()
            self.document_topic_counters[document_id][random_topic_id] += 1
            self.corpus_topic_distribution[corpus_index] = random_topic_id
            self.word_topic_counters[word_id][random_topic_id] += 1
            self.topic_counter[random_topic_id] += 1

    def run_consistency_tests(self):
        """
        There are too many collections and they should be consistent.
        List of readonly collections: 
            1) corpus
            3) vocab
            8) document_id_to_words
            9) document_to_word_count
            10) word_to_index_in_vocab
        List of mutable collections that should be consistent:
            2) corpus_topic_distribution
            4) topic_counter
            5) document_topic_counters
            6) word_topic_counters
            7) topic_word_counters - can be empty or not filled fully. we will not check it.

        We will check list of mutable collections for consistency.
        """

        # basic size checks
        assert len(self.corpus_topic_distribution) == self.CORPUS_SIZE
        assert len(self.topic_counter) == self.NUM_OF_TOPICS
        assert len(self.document_topic_counters) == self.NUM_OF_DOCS
        assert len(self.word_topic_counters) == self.VOCAB_SIZE

        # let's think that corpus_topic_distribution is a ground truth.

        # check topic counter
        new_topic_counter = [0 for _ in range(self.NUM_OF_TOPICS)]
        new_word_topic_counters = [[0 for topic_id in range(self.NUM_OF_TOPICS)] for word_id in range(self.VOCAB_SIZE)]
        new_document_topic_counters = [[0 for topic_id in range(self.NUM_OF_TOPICS)] for word_id in
                                       range(self.NUM_OF_DOCS)]
        for corpus_id, topic_id in enumerate(self.corpus_topic_distribution):
            word_id, document_id = self.word_document_corpus[corpus_id]
            new_topic_counter[topic_id] += 1
            new_word_topic_counters[word_id][topic_id] += 1
            new_document_topic_counters[document_id][topic_id] += 1

        # main consistency checks
        assert new_topic_counter == self.topic_counter
        assert new_word_topic_counters == self.word_topic_counters
        assert new_document_topic_counters == self.document_topic_counters

    def increase_topic_count_and_change_topic(self, *, corpus_id: int, word_id: int, document_id: int, topic_id: int):
        assert corpus_id < self.CORPUS_SIZE
        self.corpus_topic_distribution[corpus_id] = topic_id
        self._change_topic_count(word_id=word_id, document_id=document_id, topic_id=topic_id, change_num=1)

    def decrease_topic_count(self, *, word_id: int, document_id: int, topic_id: int):
        self._change_topic_count(word_id=word_id, document_id=document_id, topic_id=topic_id, change_num=-1)

    def get_most_popular_words_per_topic(self, topic_id: int, top_count: int = 10) -> List[str]:
        assert topic_id < self.NUM_OF_TOPICS

        if self.topic_word_counter is None:
            self.topic_word_counter = [None for _ in range(self.NUM_OF_TOPICS)]

        # lazy topic initialization - count only on request
        if self.topic_word_counter[topic_id] is None:
            self.topic_word_counter[topic_id] = [0 for _ in range(self.VOCAB_SIZE)]

            for word_id, word_topic_counter in enumerate(self.word_topic_counters):
                self.topic_word_counter[topic_id][word_id] += word_topic_counter[topic_id]

        top_word_ids = Counter(self.topic_word_counter[topic_id]).most_common()[:top_count]
        top_words = [self.vocab[word_id] for word_id, word_count in top_word_ids]
        return top_words

    @classmethod
    def get_main_attribute_names(cls) -> List[str]:
        return ['word_topic_counters', 'vocab', 'document_topic_counters', 'corpus_topic_distribution', 'topic_counter',
                'word_document_corpus']

    def export_trained_helper(self) -> str:
        unique_timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.mkdir(unique_timestamp)

        for attr_name in self.get_main_attribute_names():
            with open(os.path.join(unique_timestamp, f'{attr_name}.json'), 'w') as json_file:
                json.dump(self.__getattribute__(attr_name), json_file)

        return unique_timestamp

    @staticmethod
    def import_trained_helper(import_folder: str) -> 'DataHelper':
        imported_data_helper = DataHelper()

        for attr_name in imported_data_helper.get_main_attribute_names():
            path_to_attr = os.path.join(import_folder, f'{attr_name}.json')
            if os.path.isfile(path_to_attr):
                with open(path_to_attr, 'r') as json_file:
                    attr_value = json.load(json_file)
                    imported_data_helper.__setattr__(attr_name, attr_value)
            else:
                raise FileNotFoundError(f'Attribute with path {path_to_attr} not found.')

        # set pseudo-constants
        imported_data_helper.NUM_OF_TOPICS = len(imported_data_helper.topic_counter)
        imported_data_helper.NUM_OF_DOCS = len(imported_data_helper.document_topic_counters)
        imported_data_helper.CORPUS_SIZE = len(imported_data_helper.corpus_topic_distribution)
        imported_data_helper.VOCAB_SIZE = len(imported_data_helper.vocab)

        return imported_data_helper

    # private methods
    def _change_topic_count(self, word_id: int, document_id: int, topic_id: int, change_num: int):
        assert topic_id < self.NUM_OF_TOPICS
        assert document_id < self.NUM_OF_DOCS
        assert word_id < self.VOCAB_SIZE
        assert change_num == 1 or change_num == -1

        self.topic_counter[topic_id] += change_num
        assert self.topic_counter[topic_id] >= 0

        self.document_topic_counters[document_id][topic_id] += change_num
        assert self.document_topic_counters[document_id][topic_id] >= 0

        self.word_topic_counters[word_id][topic_id] += change_num
        assert self.word_topic_counters[word_id][topic_id] >= 0

    def _get_index_by_word(self, word: str) -> int:
        return self._word_to_i[word]

    def _get_random_topic_id(self) -> int:
        # low - inclusive, high - exclusive
        return np.random.randint(0, self.NUM_OF_TOPICS)

    def _create_word_document_corpus(self) -> List[Tuple[int, int]]:
        word_document_corpus = []

        for document_id, document_words in enumerate(self.document_id_to_words):
            for document_word in document_words:
                word_id = self._get_index_by_word(document_word)
                word_document_corpus.append((word_id, document_id))

        return word_document_corpus

    def _get_document_id_to_words_and_vocab(self, input_filename: str) -> Tuple[List[List[str]], List[str]]:
        """
        Reads input files and creates helper collections.
        
        TODO: add parameter description
        """
        document_id_to_words = []
        vocab_counter = defaultdict(int)
        with open(input_filename, 'r') as input_f:
            for i, line in enumerate(input_f):
                # first line contains number of reviews
                if i == 0:
                    num_of_reviews = int(line.strip())
                    document_id_to_words = [None for _ in range(num_of_reviews)]
                    continue

                document_words = line.strip().split(' ')
                # document words should be list-like, not set.
                # looks logical, since we can increase topic weight by repeating very topic-specific word
                document_id_to_words[i - 1] = document_words

                for document_word in document_words:
                    vocab_counter[document_word] += 1

        vocab = [word for word, word_count in Counter(vocab_counter).most_common()]

        # to save space we will assign lower indices to the most common words
        return document_id_to_words, vocab


class GibbsSampler:
    def __init__(self, data_proxy: DataHelper, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> None:
        self.data_proxy = data_proxy

        self.ALPHA = alpha
        self.BETA = beta

    def _gibbs_sampling(self, *, word_id: int, document_id: int, topic_id: int) -> float:
        """Gibbs sampling according to formula above."""
        c1 = self.data_proxy.word_topic_counters[word_id][topic_id]
        c2 = self.data_proxy.topic_counter[topic_id]

        c3 = self.data_proxy.document_topic_counters[document_id][topic_id]
        c4 = self.data_proxy.document_to_word_count[document_id] - 1

        left_operand = (c1 + self.BETA) / (c2 + self.data_proxy.VOCAB_SIZE * self.BETA)
        right_operand = (c3 + self.ALPHA) / (c4 + self.data_proxy.NUM_OF_TOPICS * self.ALPHA)

        return left_operand * right_operand

    def _softmax(self, x: List[float]):
        p = np.exp(x) / np.sum(np.exp(x))
        return p

    def train(self, num_of_iterations: int = DEFAULT_NUM_OF_ITERATIONS, iterations_to_save: int = -1) -> DataHelper:
        """Trains and returns trained data proxy."""
        for it in tqdm(range(num_of_iterations)):
            if iterations_to_save != -1 and (it + 1) % iterations_to_save == 0:
                self.data_proxy.export_trained_helper()

            for corpus_id, topic_id in enumerate(self.data_proxy.corpus_topic_distribution):
                word_id, document_id = self.data_proxy.word_document_corpus[corpus_id]
                self.data_proxy.decrease_topic_count(word_id=word_id, document_id=document_id, topic_id=topic_id)

                topic_weights = []
                for new_topic_id in range(self.data_proxy.NUM_OF_TOPICS):
                    gibbs_value = self._gibbs_sampling(document_id=document_id, topic_id=new_topic_id, word_id=word_id)
                    topic_weights.append(gibbs_value)

                topic_weights = self._softmax(topic_weights)
                best_new_topic_id = np.random.choice(self.data_proxy.NUM_OF_TOPICS, p=topic_weights) - 1
                self.data_proxy.increase_topic_count_and_change_topic(corpus_id=corpus_id, word_id=word_id,
                                                                      document_id=document_id,
                                                                      topic_id=best_new_topic_id)

            self.data_proxy.run_consistency_tests()

        return self.data_proxy
