# sample_terms = {'samsung', 'tech company', 'entity'}

# g = Graph()
# # g.add_nodes_from(sample_terms)
#
# i = 0
# terms_to_match = sample_terms
#
# while terms_to_match and i < 4:
#     next_level_terms = set()
#     for term in terms_to_match:
#         potential_parents = set(taxonomy_builder.possible_hypernyms(hyponym=term))
#         intersection = potential_parents.intersection(terms_to_match)
#         intersection.discard(term)
#         intersection = list(intersection)
#         if len(intersection) == 1:
#             hypernym = intersection[0]
#         else:
#             choose_next_step_wisely = next_level_terms.intersection(potential_parents)
#             hypernym = taxonomy_builder.count_common_words(term).most_common()[0][0]
#
#         g.add_edge(term, hypernym)
#         next_level_terms.add(hypernym)
#
#     terms_to_match = next_level_terms
#     i += 4

# while terms_to_match:
#     next_level_terms = []
#     for term in terms_to_match:

# hyponyms_num = len(taxonomy_builder.hypernym_to_hyponym.get(term, []))
# print(term, hyponyms_num)

# if hyponyms_num < 10:
#     i = 0
#     cur_term = term
#     while i != 3:
#         #next_term = taxonomy_builder.get_top_with_scores(cur_term)[0]
#         next_term = taxonomy_builder.count_common_words(cur_term).most_common()[0][0]
#         g.add_edge(cur_term, next_term)
#         cur_term = next_term
#         i += 1

#     potential_parents = set(taxonomy_builder.possible_hypernyms(hyponym=term))
#     intersection = potential_parents.intersection(sample_terms)
#     intersection.discard(term)
#     intersection = list(intersection)
#     if len(intersection) == 1:
#         hypernym = intersection[0]
#         g.add_edge(term, hypernym)
#         next_level_terms.append(hypernym)
#
# terms_to_match = next_level_terms

#     else:
#         '''
#         Resolve
#         1. Check top-10 for specific hyponym
#         2. Check number of hypernyms, choose less specific
#         '''
#         # term_hyponyms_count = len(taxonomy_builder.hypernym_to_hyponym.get(term, []))
#         # # child cannot be more generic than parent
#         # less_generic_parents = [
#         #     parent for parent in intersection
#         #     if len(taxonomy_builder.hypernym_to_hyponym.get(parent, [])) < term_hyponyms_count
#         # # ]
#         # intersection = list(set(intersection) - set(less_generic_parents))
#
#         top_with_scores = taxonomy_builder.get_top_with_scores(hyponym=term, n=10)
#         parent_scores = [top_with_scores.get(parent, 0) for parent in intersection]
#
#         top_mentions = taxonomy_builder.count_common_words(hyponym=term)
#         parent_mentions = [top_mentions.get(parent, 0) for parent in intersection]
#
#         # g.add_edge(term, best_score_candidate)
#         t = 1

'''
# my algorithm:
# 1) try every term as possible leaf
#   - look for other nodes as for potential parents: split them to words as match with possible words
# -> if matched, then use it as parent and other as leaf
#   -
# 2)
'''

# words_to_terms = defaultdict(set)
#
#
# for term in sample_terms:
#     term_words = term.split()
#     for word_combination in chain.from_iterable(combinations(term_words, r) for r in range(len(term_words) + 1)):
#         words_to_terms[word_combination].add(term)

# g = Graph()
# root = 'entity'
# g.add_node(root)
# g.add_node(root*2)
# g.add_edge(root, root*2)
# nx.draw(g, with_labels=True)
# plt.show()

# for potential_child in taxonomy_builder.hypernym_to_hyponym[root]:
#     pass
#
# for term in sample_terms:
#     print(term)
#     taxonomy_builder.count_common_words(hyponym=term)
#     print()

# sample_terms = {
#     'samsung',  #
#     'tech company',
#     'company',
#     'organization',
#     'boeing',  #
#     'aircraft manufacturer'  #
# }
#
# sample_terms = {
#     "donald trump",
#     "business people",
#     "person",
# }
#
# sample_terms = {
#     'donald trump',  #
#     'business people',  #
#     'angelia jolie',  #
#     'person',
#     'frodo',  #
#     'character',
#     'gandalf',  #
#     'samsung',  #
#     'tech company',
#     'company',
#     'organization',
#     'boeing',  #
#     'aircraft manufacturer'  #
# }