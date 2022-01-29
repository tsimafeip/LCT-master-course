import os
from collections import Counter, defaultdict
from datetime import datetime

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from tqdm import tqdm
import pyLDAvis

'''

TODO:
1) Run full train and analyse trained data
2) Add numpy arrays instead of lists and check speed changes
3) Write comments
4) Try to add stemming and lemmatization
'''

DEFAULT_MODEL_FILENAME = 'model.json'
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

    def __init__(self, input_filename: str, num_of_topics: int = DEFAULT_NUM_OF_TOPICS):
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
        6) topic_word_counters: List[List[int]] - List of length NUM_OF_TOPICS.
        Each element is List of legth VOCAB_SIZE and values of topic_word_count.

        Devivatives and helpers
        7) document_id_to_words: List[List[str]] - List of length NUM_OF_DOCS.
        Each element is List of raw document str words in original order without any grouping.
        8) document_to_word_count: List[int] - List of length NUM_OF_DOCS with document_size as values.
        9) word_to_index_in_vocab: Dict[str, int] - Dict with word_str as key and integer index in vocabulary as value.
        """
        document_id_to_words, vocab_counter = self._get_document_id_to_words_and_vocab(input_filename)

        # initialize main collection 3): vocab array and helper word_to_i dictionary
        self.vocab = [word for word, word_count in vocab_counter.most_common()]
        # init helper 10) collection
        self._word_to_i = {word: i for i, word in enumerate(self.vocab)}
        self.vocab_counter = vocab_counter

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
        self.topic_word_counters = [[0 for word_id in range(self.VOCAB_SIZE)] for topic_id in range(self.NUM_OF_TOPICS)]
        # initialize main collection 2)
        self.corpus_topic_distribution = [0 for _ in range(self.CORPUS_SIZE)]
        # initialize main collection 4)
        self.topic_counter = [0 for _ in range(self.NUM_OF_TOPICS)]

        random_topics = np.random.randint(0, self.NUM_OF_TOPICS, self.CORPUS_SIZE)

        # uniformly assign topic and fill collections 2), 4), 5), 6)
        for corpus_id, (word_id, document_id) in enumerate(self.word_document_corpus):
            random_topic_id = random_topics[corpus_id]
            
            self.document_topic_counters[document_id][random_topic_id] += 1
            self.corpus_topic_distribution[corpus_id] = random_topic_id
            self.topic_word_counters[random_topic_id][word_id] += 1
            self.topic_counter[random_topic_id] += 1

    def run_consistency_tests(self):
        """
        There are too many collections and they should be consistent.
        List of readonly collections: 
            1) corpus
            3) vocab
            7) document_id_to_words
            8) document_to_word_count
            9) word_to_index_in_vocab
        List of mutable collections that should be consistent:
            2) corpus_topic_distribution
            4) topic_counter
            5) document_topic_counters
            6) topic_word_counters

        We will check list of mutable collections for consistency.
        """

        # basic size checks
        assert len(self.corpus_topic_distribution) == self.CORPUS_SIZE
        assert len(self.topic_counter) == self.NUM_OF_TOPICS
        assert len(self.document_topic_counters) == self.NUM_OF_DOCS
        assert len(self.topic_word_counters) == self.NUM_OF_TOPICS

        # let's think that corpus_topic_distribution is a ground truth.

        # check topic counter
        new_topic_counter = [0 for _ in range(self.NUM_OF_TOPICS)]
        new_topic_word_counters = [[0 for word_id in range(self.VOCAB_SIZE)] for topic_id in range(self.NUM_OF_TOPICS)]
        new_document_topic_counters = [[0 for topic_id in range(self.NUM_OF_TOPICS)] for word_id in
                                       range(self.NUM_OF_DOCS)]
        for corpus_id, topic_id in enumerate(self.corpus_topic_distribution):
            word_id, document_id = self.word_document_corpus[corpus_id]
            new_topic_counter[topic_id] += 1
            new_topic_word_counters[topic_id][word_id] += 1
            new_document_topic_counters[document_id][topic_id] += 1

        # main consistency checks
        assert new_topic_counter == self.topic_counter
        assert new_topic_word_counters == self.topic_word_counters
        assert new_document_topic_counters == self.document_topic_counters

    def increase_topic_count_and_change_topic(self, *, corpus_id: int, word_id: int, document_id: int, topic_id: int):
        assert corpus_id < self.CORPUS_SIZE
        self.corpus_topic_distribution[corpus_id] = topic_id
        self._change_topic_count(word_id=word_id, document_id=document_id, topic_id=topic_id, change_num=1)

    def decrease_topic_count(self, *, word_id: int, document_id: int, topic_id: int):
        self._change_topic_count(word_id=word_id, document_id=document_id, topic_id=topic_id, change_num=-1)

    def get_most_popular_words_per_topic(self, topic_id: int, top_count: int = 10) -> List[str]:
        assert topic_id < self.NUM_OF_TOPICS

        top_word_ids = Counter(self.topic_word_counters[topic_id]).most_common()[:top_count]
        top_words = [self.vocab[word_id] for word_id, word_count in top_word_ids]
        return top_words

    @classmethod
    def get_main_attribute_names(cls) -> List[str]:
        return ['topic_word_counters', 'vocab', 'document_topic_counters', 'corpus_topic_distribution', 'topic_counter',
                'word_document_corpus']

    def export_trained_helper(self, model_filename: str = DEFAULT_MODEL_FILENAME) -> str:
        unique_timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.mkdir(unique_timestamp)

        data_to_export = {
            attr_name: self.__getattribute__(attr_name) 
            for attr_name in self.get_main_attribute_names()
        }
        
        path_to_model = os.path.join(unique_timestamp, model_filename)
        with open(path_to_model, 'w') as json_file:
            json.dump(data_to_export, json_file)

        return path_to_model

    @staticmethod
    def import_trained_helper(base_data_helper: 'DataHelper', path_to_model_file: str) -> 'DataHelper':
        with open(path_to_model_file, 'r') as json_file:
            model_data = json.load(json_file)
            for attr_name, attr_value in model_data.items():
                base_data_helper.__setattr__(attr_name, attr_value)

        return base_data_helper

    def get_movies_vis_data(self):
        """
        Returns model data in pyLDAvis format.

        I took format from this example:
        https://nbviewer.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb
        """
        movies_model_data = {
            'topic_term_dists': self.topic_word_counters, 
            'doc_topic_dists': self.document_topic_counters,
            'doc_lengths': self.document_to_word_count,
            'vocab': self.vocab,
            'term_frequency': [self.vocab_counter[word] for word in self.vocab]
        }

        return pyLDAvis.prepare(**movies_model_data)

    def compare_data_helpers(self, other: 'DataHelper'):
        # sanity check if we downloaded everything correctly
        for topic_id in range(self.NUM_OF_TOPICS):
            print(topic_id+1, end=" ")
            original_words = self.get_most_popular_words_per_topic(topic_id, 100)
            imported_tr_words = other.get_most_popular_words_per_topic(topic_id, 100)
            if original_words != imported_tr_words:
                print(original_words, imported_tr_words)
                raise Exception(f'Comparison failed on topic #{topic_id+1}')

    def print_top_words_for_all_topics(self, top_words_count: int = 10):
        for topic_id in range(self.NUM_OF_TOPICS):
            print('Topic ID:', topic_id+1)
            print('Top-10 words:', self.get_most_popular_words_per_topic(topic_id, top_words_count))
    
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

        self.topic_word_counters[topic_id][word_id] += change_num
        assert self.topic_word_counters[topic_id][word_id] >= 0

    def _get_index_by_word(self, word: str) -> int:
        return self._word_to_i[word]

    def _get_random_topic_id(self) -> int:
        # low - inclusive, high - exclusive
        return np.random.randint(0, self.NUM_OF_TOPICS)

    def _create_word_document_corpus(self) -> List[Tuple[int, int]]:
        word_document_corpus = []

        for document_id, document_words in enumerate(self.document_id_to_words):
            for document_word in document_words:
                # this means that we filtered out this word like as too rare or too common
                if document_word not in self.vocab_counter:
                    continue

                word_id = self._get_index_by_word(document_word)
                word_document_corpus.append((word_id, document_id))

        return word_document_corpus

    def _get_document_id_to_words_and_vocab(
        self, input_filename: str, no_below: int = 10, no_above: float = 0.5,
    ) -> Tuple[List[List[str]], Dict[str, int]]:
        """
        Reads input files and creates helper collections.

        We also filter vocab as in Gensim:
        https://tedboy.github.io/nlps/generated/generated/gensim.corpora.Dictionary.filter_extremes.html.
        """

        word_to_document_ids = defaultdict(set)
        document_id_to_words = []
        vocab_counter = Counter()
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
                document_id = i - 1
                document_id_to_words[document_id] = document_words

                for document_word in document_words:
                    vocab_counter[document_word] += 1
                    word_to_document_ids[document_word].add(document_id)
        
        total_num_of_docs = len(document_id_to_words)
        filtered_vocab_counter = Counter()
        for word, word_count in vocab_counter.items():
            num_docs_for_word = len(word_to_document_ids[word])
            if num_docs_for_word < no_below:
                continue
            
            if num_docs_for_word > no_above*total_num_of_docs:
                continue

            filtered_vocab_counter[word] = word_count

        # to save space we will assign lower indices to the most common words
        return document_id_to_words, filtered_vocab_counter


class GibbsSampler:
    def __init__(self, data_proxy: DataHelper, alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> None:
        self.data_proxy = data_proxy

        self.ALPHA = alpha
        self.BETA = beta

    def _gibbs_sampling(self, *, word_id: int, document_id: int, topic_id: int) -> float:
        """Gibbs sampling according to formula above."""
        c1 = self.data_proxy.topic_word_counters[topic_id][word_id]
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
