import ast
import json
import os.path
from typing import Tuple, Generator, List

import spacy
import torch
from spacy import Language
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchtext as tt
from itertools import chain
from tqdm import tqdm

OTHER_TAG = '<O>'
SEP_TOKEN = '<SEP>'
DEV_RATIO = 0.1
BEST_CHECKPOINT = "best-model.pt"
TRAIN_MODEL = True


# BEGIN_TAG = '<B>'
# INSIDE_TAG = '<I>'

class TaggingDataset(Dataset):
    """
    A Pytorch dataset representing a tagged corpus in CoNLL format.
    Each item is a dictionary {'sentence': ..., 'tags': ...} representing one tagged sentence.
    The values under 'sentence' and 'tags' are int tensors of shape (sequence_length,).
    """

    def __init__(self, sentences, tag_sequences):
        self.sentences = sentences
        self.tag_sequences = tag_sequences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sample = {"sentence": self.sentences[idx], "tags": self.tag_sequences[idx]}
        return sample


class NerDataLoader:
    # generic type Generator[yield_type, send_type, return_type]
    @staticmethod
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

    def tagging_collate_fn(self, batch):
        tensors = []
        for instance in batch:
            sent_t = instance["sentence"]
            pos_t = instance["tags"]
            tensors.append(torch.stack([sent_t, pos_t]))

        return torch.stack(tensors)

    def preprocess_token(self, word):
        if word == ".":
            return "</s>"
        else:
            return word.lower()

    # create vocabularies
    def unk_init(self, x):
        return torch.randn_like(x)

    def load_corpus(self, nlp: Language, filename: str, force_reload: bool = False):

        sentences = list()
        tag_sequences = list()

        preprocessed_sents_filename = 'res_sentences_' + filename + '.json'
        preprocessed_tags_filename = 'res_tags_' + filename + '.json'
        if not force_reload and \
                os.path.exists(preprocessed_tags_filename) and os.path.exists(preprocessed_tags_filename):
            with open(preprocessed_sents_filename, encoding='utf-8') as sents_f, \
                    open(preprocessed_tags_filename, encoding='utf-8') as tags_f:
                sentences = json.load(sents_f)
                tag_sequences = json.load(tags_f)
        else:
            all_count = ne_count = 0
            for i, named_entity, types, sentence in self.read_train_file(train_filepath=filename):
                tokens = nlp(sentence)
                tag_sequence = [OTHER_TAG for i in range(len(tokens))]
                # we simplify by selecting only one type for each entity
                entity_type_words = [type_word.lower() for type_word in types[0].split()]
                entity_type = " ".join(entity_type_words)
                named_entity_words = [ne_word for ne_word in named_entity.split()]

                ne_found = False

                for i in range(len(tokens)):
                    if tokens[i].text.lower() == named_entity_words[0].lower():
                        ne_found = True
                        for j in range(len(named_entity_words)):
                            tag_sequence[i + j] = entity_type
                        break

                all_count += 1
                ne_count += ne_found

                if ne_found:
                    sentences.append(named_entity_words + [SEP_TOKEN] + [token.text for token in tokens])
                    tag_sequences.append([entity_type] * len(named_entity_words) + [SEP_TOKEN] + tag_sequence)

            with open(preprocessed_sents_filename, 'w', encoding='utf-8') as sents_f, \
                    open(preprocessed_tags_filename, 'w', encoding='utf-8') as tags_f:
                json.dump(sentences, sents_f)
                json.dump(tag_sequences, tags_f)

        return sentences, tag_sequences

    def make_dataloader(self, sentences, tag_sequences, sents_vocab, tags_vocab):
        train_sent_ts = [torch.tensor(sents_vocab.lookup_indices(sent)) for sent in sentences]
        train_tag_ts = [torch.tensor(tags_vocab.lookup_indices(seq)) for seq in tag_sequences]

        train_dataset = TaggingDataset(train_sent_ts, train_tag_ts)
        train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=self.tagging_collate_fn)

        return train_dataloader

    def load_data(self, train_filename, dev_filename, test_filename, nlp):
        train_dataloader, dev_dataloader, test_dataloader, vocab, tagset, fasttext_embeddings = self.load(
            train_filename, dev_filename, test_filename, nlp=nlp)

        # some exploratory data analysis
        print(len(train_dataloader), len(dev_dataloader), len(test_dataloader))
        print(type(tagset), type(vocab), type(fasttext_embeddings))
        print(tagset.lookup_token(10), vocab.lookup_token(10))
        print(fasttext_embeddings.shape)

        for item in train_dataloader:
            # I convert to numpy because lookup tokens does not accept torch tensors
            tokens, tags = item[0][0].numpy(), item[0][1].numpy()

            print(vocab.lookup_tokens(tokens))
            print(tagset.lookup_tokens(tags))
            break

        return train_dataloader, dev_dataloader, test_dataloader, vocab, tagset, fasttext_embeddings

    def load(self, training_corpus, development_corpus, test_corpus, nlp: Language):
        """
        Loads a tagged corpus in CoNLL format and returns a tuple with the following entries:
        - DataLoader for the training set
        - DataLoader for the development set
        - DataLoader for the test set
        - Vocabulary for the natural-language side (as a Torchtext Vocab object)
        - Vocabulary for the tag side (dito)
        - Pretrained embeddings, as a tensor of shape (vocabulary size, embedding dimension)

        :param training_corpus: filename of the training corpus
        :param development_corpus: filename of the development corpus
        :param test_corpus: filename of the test corpus
        :return:
        """
        # read CoNLL file "de_gsd-ud-train.conllu"
        train_sentences, train_tag_sequences = self.load_corpus(nlp, training_corpus)
        dev_sentences, dev_tag_sequences = self.load_corpus(nlp, development_corpus)
        test_sentences, test_tag_sequences = dev_sentences[:], dev_tag_sequences[:]  # load_corpus(test_corpus)

        # Build vocabulary from pretrained word embeddings.
        # Theoretically, FastText should predict embeddings for unknown words using subword embeddings;
        # but this does not seem to work when using it through Torchtext. Maybe I should fix this sometime.
        fasttext = tt.vocab.FastText(language='en', unk_init=self.unk_init)
        sents_vocab = tt.vocab.build_vocab_from_iterator(chain(train_sentences, dev_sentences, test_sentences),
                                                         specials=["<unk>", "<pad>"])
        sents_vocab.set_default_index(0)
        pretrained_embeddings = fasttext.get_vecs_by_tokens(sents_vocab.get_itos())
        tags_vocab = tt.vocab.build_vocab_from_iterator(
            chain(train_tag_sequences, dev_tag_sequences, test_tag_sequences))

        # Map corpora to datasets
        train_dataloader = self.make_dataloader(train_sentences, train_tag_sequences, sents_vocab, tags_vocab)
        dev_dataloader = self.make_dataloader(dev_sentences, dev_tag_sequences, sents_vocab, tags_vocab)
        test_dataloader = self.make_dataloader(test_sentences, test_tag_sequences, sents_vocab, tags_vocab)

        return train_dataloader, dev_dataloader, test_dataloader, sents_vocab, tags_vocab, pretrained_embeddings


class NeuralPredictor(nn.Module):
    """
    """

    def __init__(self, hidden_size: int, output_size: int, pretrained_embeddings: torch.Tensor,
                 lstm_layers_num: int = 1, bidirectional: bool = False, dropout: float = 0, **kwargs):
        """
        """
        # Initializes internal Module state.
        super().__init__()

        # embedding layers tranforms binary vector to smaller representation with non-binary elements
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)

        # here we create LSTM layer that mostly defines the architecture of the whole RNN
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_size,
                            batch_first=True,
                            num_layers=lstm_layers_num,
                            bidirectional=bidirectional)

        # final layer is used for transforming vector of hidden_size to the vector of the required output length
        self.final_linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

        # dropout layer to regulize training and avoid overfitting
        self.dropout = nn.Dropout(dropout)

        self.nlp = kwargs['nlp']

    # forward function
    def forward(self, raw_input: torch.Tensor):
        """
        Forward path of the model.

        raw_input: torch.Tensor - tensor of shape (sequence_len,)
        """

        sequence_length = raw_input.size(0)

        # transform raw input to dense embedding vector of shape (sequence_len, embedding_size)
        # and add dummy batch_size 1 as a first dimension
        # resulting tensor has shape (1, sequence_len, embedding_size)
        embedded_input = self.embedding(raw_input).view(1, sequence_length, -1)

        # lstm_output has two possible shapes:
        # (1, sequence_len, hidden_size) for unidirectional LSTM model
        # (1, sequence_len, 2*hidden_size) for bidirectional LSTM model
        lstm_output, _ = self.lstm(embedded_input)

        # predictions has shape (1, sequence_len, num_classes)
        predictions = self.final_linear(self.dropout(lstm_output))

        # reshaping to remove artificial batch dimension
        return predictions.view(sequence_length, -1)

    def predict_type(self, named_entity: str, sentence: str):
        sentence = [token.text for token in nlp(named_entity)] + [SEP_TOKEN] + [token.text for token in nlp(sentence)]

        sentence_numerical_vector = torch.LongTensor(vocab.lookup_indices(sentence))

        outputs = model.forward(sentence_numerical_vector)
        _, predicted_tags_indices = torch.max(outputs.data, 1)

        predicted_tags = tagset.lookup_tokens(predicted_tags_indices.numpy())

        print(sentence)
        print(predicted_tags)

        predicted_types = list(set([tag for tag in predicted_tags if tag != OTHER_TAG]))

        return predicted_types

    def get_num_of_correct_tags(self, gold_tags: torch.Tensor, prediction_output: torch.Tensor) -> int:
        """Returns correct tags and total tags based on gold tags and prediction output."""
        # chooses tag with max probability value as predicted tag for input token
        # torch.max returns (value, indices) tuple, we do not need values here.
        _, predicted_tags = torch.max(prediction_output.data, 1)
        # correct tags is a single scalar tensor, then we call item() to get the underlying value
        correct_tags = (predicted_tags == gold_tags.data).sum().item()

        return correct_tags

    def accuracy(self, gold_tags: torch.Tensor, prediction_output: torch.Tensor) -> float:
        """Calculates accuracy based on gold tags and prediction output for the single sentence."""
        # chooses tag with max probability value as predicted tag for input token
        _, predicted_tags = torch.max(prediction_output.data, 1)
        total_tags = gold_tags.size(0)
        # correct tags is a single scalar tensor, then we call item() to get the underlying value
        correct_tags = (predicted_tags == gold_tags.data).sum().item()

        return correct_tags / total_tags

    def train_loop(self, optimizer: torch.optim.Optimizer, loss_function, dataloader: DataLoader) \
            -> Tuple[float, float]:
        """
        Runs one complete training epoch, i. e. trains the model on the whole amount of available training data and updates weights.
        :param model: a pytorch model
        :param optimizer: a pytorch optimizer
        :param loss_function: the type of loss function to use
        :param dataloader: a dataloader for getting the training instances

        :return: (epoch_loss, epoch_accuracy): metrics to evaluate the current state of model
        """
        correct_tags = total_tags = loss_sum = epoch_loss = epoch_accuracy = 0
        # turns model to train mode: activates dropout and some other features that are useful only for training.
        model.train()
        # this context manager only appends 'train' prefixes to all logged metrics
        for i, item in enumerate(dataloader):
            # simple logging to understand the stage of training process inside one epoch
            if (i + 1) % 100 == 0:
                print(i + 1, end=" ")

            # two 1d-vector-like Tensors with the same length
            input_tokens, gold_tags = item[0][0], item[0][1]

            # matrix with vector of tag probabilities as rows for each token
            prediction_output = self.forward(input_tokens)
            loss = loss_function(prediction_output, gold_tags)

            # backpropagation
            # make gradients zero
            optimizer.zero_grad()
            # calculate backward pass
            loss.backward()
            # make optimizer step
            optimizer.step()

            # update loss and accuracy
            loss_sum += loss.item()
            correct_tags += self.get_num_of_correct_tags(gold_tags, prediction_output)
            total_tags += gold_tags.size(0)

        epoch_accuracy = correct_tags / total_tags
        epoch_loss = loss_sum / len(dataloader)

        return epoch_loss, epoch_accuracy

    def validation_loop(self, loss_function, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Runs one complete validation (test) loop,
        i. e. validates(tests) the model on the whole amount of available validation (test) data and reports quality metrics.

        :param model: a pytorch model
        :param loss_function: the type of loss function to use
        :param dataloader: a dataloader for getting the validation (test) instances

        :return: (epoch_loss, epoch_accuracy): metrics to evaluate the current state of model
        """
        correct_tags = total_tags = loss_sum = epoch_loss = epoch_accuracy = 0
        model.eval()
        # we freeze gradient calculation for validation and, therefore, do not update weights
        # in other steps it is the same process as for training
        with torch.no_grad():
            for i, item in enumerate(dataloader):
                input_tokens, gold_tags = item[0][0], item[0][1]
                predicted_tags = self.forward(input_tokens)

                loss_sum += loss_function(predicted_tags, gold_tags).item()
                correct_tags += self.get_num_of_correct_tags(gold_tags, predicted_tags)
                total_tags += gold_tags.size(0)

        epoch_accuracy = correct_tags / total_tags
        epoch_loss = loss_sum / len(dataloader)

        return epoch_loss, epoch_accuracy


def load_default_predictor() -> NeuralPredictor:
    hyperparams = {
        "output_size": len(tagset),
        "pretrained_embeddings": fasttext_embeddings,
        "input_size": len(vocab),
        "hidden_size": 128,
        "num_classes": len(tagset),
        "embeddings_type": 'fasttext',
        "batch_size": 1,
        "num_epochs": 10,
        "learning_rate": 0.001,
        "num_lstm_layers": 2,
        "bidirectional": True,
        "dropout": 0.25,
    }

    model = NeuralPredictor(**hyperparams, nlp=nlp)
    if os.path.isfile(BEST_CHECKPOINT):
        model.load_state_dict(torch.load(BEST_CHECKPOINT))
    return model


def create_dev_file(train_filename: str, dev_filename: str):
    if not os.path.exists(dev_filename):
        with open(train_filename, 'r', encoding='utf-8') as train_f:
            train_lines = train_f.read().splitlines()
            dev_size = round(len(train_lines) * DEV_RATIO)
            dev_lines, train_lines = train_lines[:dev_size], train_lines[dev_size:]

        with open(train_filename, 'w', encoding='utf-8') as train_f, \
                open(dev_filename, 'w', encoding='utf-8') as dev_f:
            dev_f.writelines([line + '\n' for line in dev_lines])
            train_f.writelines([line + '\n' for line in train_lines])


if __name__ == '__main__':
    # data files
    train_filename = 'train.tsv'
    dev_filename = 'dev.tsv'
    test_filename = 'test.tsv'
    gold_test_filename = 'test-groundtruth.tsv'

    nlp = spacy.load('en_core_web_sm')
    ner_data_loader = NerDataLoader()
    train_dataloader, dev_dataloader, test_dataloader, vocab, tagset, fasttext_embeddings = \
        ner_data_loader.load_data(train_filename, dev_filename, test_filename, nlp)

    hyperparams = {
        "output_size": len(tagset),
        "pretrained_embeddings": fasttext_embeddings,
        "input_size": len(vocab),
        "hidden_size": 128,
        "num_classes": len(tagset),
        "embeddings_type": 'fasttext',
        "batch_size": 1,
        "num_epochs": 10,
        "learning_rate": 0.001,
        "num_lstm_layers": 2,
        "bidirectional": True,
        "dropout": 0.25,
    }

    model = NeuralPredictor(**hyperparams, nlp=nlp)

    # check if forward path works
    for item in train_dataloader:
        tokens = item[0][0]
        model.forward(tokens)
        break

    if os.path.isfile(BEST_CHECKPOINT):
        model.load_state_dict(torch.load(BEST_CHECKPOINT))
        model.predict_type(sentence='Francisco Valero (29 May 1906 – 15 September 1982) was a Mexican fencer.',
                           named_entity='Francisco Valero')

    if TRAIN_MODEL:
        OPTIMIZER_CLASS = torch.optim.Adam
        LOSS_FUNCTION = nn.functional.cross_entropy
        best_epoch = best_accuracy = best_val_loss = float('inf')
        # create optimizer instance
        optimizer = OPTIMIZER_CLASS(model.parameters(), lr=hyperparams["learning_rate"])
        # iterates by epochs
        for epoch in range(1, hyperparams['num_epochs'] + 1):
            print(f'Epoch #{epoch}')
            train_loss, train_accuracy = model.train_loop(optimizer, LOSS_FUNCTION, train_dataloader)
            val_loss, val_accuracy = model.validation_loop(LOSS_FUNCTION, dev_dataloader)

            print()
            print(f"Train: Loss {train_loss} | Accuracy {train_accuracy}")
            print(f"Validation: Loss {val_loss} | Accuracy {val_accuracy}")

            # saving best checkpoint
            if val_loss < best_val_loss:
                print(f'SAVING BEST MODEL WITH VAL_LOSS={val_loss}.')
                best_val_loss = val_loss
                best_epoch = epoch
                best_accuracy = val_accuracy
                torch.save(model.state_dict(), 'best-model.pt')
