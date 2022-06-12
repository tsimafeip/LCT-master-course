import json
import os.path
import sys
from collections import defaultdict, Counter, deque
from typing import Set, Dict, List, Tuple, Optional

from nltk.corpus import stopwords
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT_NODE = 'entity'
SCORE_DIFF_THRESHOLD = 0.3

NON_INFORMATIVE_HYPERNYMS = {'thing', 'name', 'group', 'element', 'local', 'member'}


class TaxonomyGraphBuilder:
    def __init__(self, wedisalod_filepath: str):
        """Train taxonomy builder."""
        small_to_big, big_to_small = self._read_webisalod(wedisalod_filepath)
        self.hyponym_to_hypernym = small_to_big
        self.hypernym_to_hyponym = big_to_small

    def _read_webisalod(self, wedisalod_filepath: str) -> \
            Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        prep_filename = wedisalod_filepath + '_prep.txt'

        if os.path.isfile(prep_filename):
            return json.load(open(prep_filename))

        def clean(raw_str: str) -> str:
            return raw_str.replace('_', ' ').replace('+', ' ').strip()

        hyponym_to_hypernym = defaultdict(dict)
        hypernym_to_hyponym = defaultdict(dict)

        with open(wedisalod_filepath) as input_f, open(prep_filename, 'w') as output_f:
            for line in tqdm(input_f):
                hyponym_hypernym, score = line.strip().split('\t')
                hyponym, hypernym = hyponym_hypernym.split(';')

                hypernym = clean(hypernym)
                hyponym = clean(hyponym)

                hypernym_words = set(hypernym.split())

                if hypernym_words.intersection(NON_INFORMATIVE_HYPERNYMS):
                    continue

                hyponym_to_hypernym[hyponym][hypernym] = float(score)
                hypernym_to_hyponym[hypernym][hyponym] = float(score)

            json.dump((hyponym_to_hypernym, hypernym_to_hyponym), output_f)

            return hyponym_to_hypernym, hypernym_to_hyponym

    @staticmethod
    def add_non_cycle_edge(g: nx.Graph, node_1: str, node_2: str):
        if node_1 != node_2:
            g.add_edge(node_1, node_2)

    def build_taxonomy_graph(self, sample_terms: Set[str]) -> nx.Graph:
        """
        Limitations:
        1) Max 50 nodes.
        2) All nodes from input.
        3) Depth at least 4.
        """

        g = nx.Graph()
        # leaves are supposed to have less hyponyms than generic upper nodes
        term_traversal_order = sorted(sample_terms,
                                      key=lambda term: len(self.hypernym_to_hyponym.get(term, [])))

        expand_leaf_to_4_level = False
        for i, term in enumerate(term_traversal_order):
            hypernyms = set(self.possible_hypernyms(hyponym=term))
            hypernyms_from_terms = hypernyms.intersection(sample_terms)
            hypernyms_from_terms.discard(term)
            hypernyms_from_terms = list(hypernyms_from_terms)

            for next_term in term_traversal_order[i + 1:]:
                parent_candidates = deque()
                # take the less generic available term
                if next_term in hypernyms_from_terms:
                    # g.add_edge(term, next_term)
                    parent_candidates.append(next_term)

                """
                Idea: we cannot just add less generic candidate. 
                We have to check scores and, probably, add best scoring option or best word count option.

                counts = 
                """
                filtered_parent_candidates = [
                    parent for parent in parent_candidates
                    if self.is_good_hypernym(hypernym=parent, hynonym=term)
                ]

                if filtered_parent_candidates:
                    best_parent_from_terms = filtered_parent_candidates[0]
                    best_hypernym, score = self.get_best_hypernym_by_score(term)
                    if best_hypernym and self.is_good_hypernym(hypernym=best_parent_from_terms, hynonym=best_hypernym):
                        self.add_non_cycle_edge(g, term, best_hypernym)
                        self.add_non_cycle_edge(g, best_hypernym, best_parent_from_terms)
                    else:
                        self.add_non_cycle_edge(g, term, best_parent_from_terms)
                    break
            else:
                common_words = self.count_common_words(term)
                if common_words and common_words.most_common()[0][1] >= 45:
                    good_hypernym = common_words.most_common()[0][0]
                    self.add_non_cycle_edge(g, term, good_hypernym)
                    self.add_non_cycle_edge(g, good_hypernym, ROOT_NODE)
                else:
                    self.add_non_cycle_edge(g, term, ROOT_NODE)

        return g

    def build_and_draw_taxonomy_graph(self, terms: Set[str], input_name: str):
        graph = self.build_taxonomy_graph(terms)

        nx.draw(graph, with_labels=True)
        plt.show()

    def count_common_words(self, hyponym: str) -> Dict[str, float]:
        counter = Counter()
        for hypernym, score in self.hyponym_to_hypernym.get(hyponym, dict()).items():
            hypernym_words = hypernym.split()
            for word in hypernym_words:
                if word not in stopwords.words('english') and word != hyponym:
                    counter[word] += 1

        return counter

    def get_top_with_scores_list(self, hyponym: str, n: int = sys.maxsize) -> List[Tuple[str, float]]:
        return sorted(self.hyponym_to_hypernym.get(hyponym, dict()).items(), key=lambda x: -x[1])[:n]

    def get_top_with_scores_dict(self, hyponym: str, n: int = sys.maxsize) -> Dict[str, float]:
        return dict(self.get_top_with_scores_list(hyponym=hyponym, n=n))

    def possible_hypernyms(self, hyponym: str) -> List[str]:
        """
        1. based on scores
        2. based on counts
        3. based on substrings
        """

        score_based = list(self.get_top_with_scores_dict(hyponym=hyponym))
        counter_based = list(self.count_common_words(hyponym=hyponym))

        return counter_based + score_based

    def get_best_hypernym_by_score(self, hyponym: str) -> Optional[Tuple[str, float]]:
        candidates = self.get_top_with_scores_list(hyponym)
        if candidates:
            return [candidate for candidate in candidates if candidate != hyponym][0]

    def predict_hypernym_from_most_common_words(self, hyponym: str) -> Optional[Tuple[str, int]]:
        most_common_words = self.count_common_words(hyponym=hyponym).most_common()
        if most_common_words and most_common_words[0][1] > 10:
            predicted_hypernym, count = most_common_words[0]
            return predicted_hypernym, count

    def predict_hypernym(self, hyponym: str) -> Optional[str]:
        predicted_hypernyms = self.get_top_with_scores_list(hyponym=hyponym)
        if predicted_hypernyms:
            i = 0
            predicted_hypernym = predicted_hypernyms[i][0]
            while predicted_hypernym not in self.hyponym_to_hypernym:
                i += 1
                predicted_hypernym = self.get_top_with_scores_list(hyponym=hyponym)[i][0]

            return predicted_hypernym

    def is_good_hypernym(self, hypernym: str, hynonym: str) -> bool:
        best_scores = self.get_top_with_scores_list(hynonym)
        best_hypernym, best_score = best_scores[0]
        parent_score = self.get_top_with_scores_dict(hynonym).get(hypernym, 0)
        if SCORE_DIFF_THRESHOLD + parent_score >= best_score \
                or self.count_common_words(hypernym).get(hypernym, 0) > 10:
            return True

        return False


def read_terms_from_file(filepath: str) -> Set[str]:
    terms = set()
    with open(filepath) as input_f:
        for line in input_f:
            if line.strip():
                terms.add(line.strip())

    return terms


def add_extra_layer_before_entity(graph: nx.Graph, hyponym: str, hypernym: str, final_layer: bool = False):
    TaxonomyGraphBuilder.add_non_cycle_edge(graph, hyponym, hypernym)
    if final_layer:
        TaxonomyGraphBuilder.add_non_cycle_edge(graph, hypernym, ROOT_NODE)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('Expected exactly 1 argument: input file.')

    input_file = sys.argv[1]
    sample_terms = read_terms_from_file(sys.argv[1])

    taxonomy_builder = TaxonomyGraphBuilder(wedisalod_filepath='data/webisalod-pairs.txt')
    taxonomy_builder.build_and_draw_taxonomy_graph(terms=sample_terms, input_name=input_file)
