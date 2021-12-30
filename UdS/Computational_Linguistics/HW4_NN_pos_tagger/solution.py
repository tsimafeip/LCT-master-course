import time

from typing import List, Union, Tuple, Callable, Optional, Dict
from comet_ml import Experiment

import torch
import torch.nn as nn
from torchtext.vocab.vocab import Vocab
from torch.utils.data import DataLoader


class LstmPosTagger(nn.Module):
    """
    """

    def __init__(self,
                 input_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 output_size: int,
                 pretrained_embeddings: torch.Tensor,
                 lstm_layers_num: int = 1,
                 bidirectional: bool = False,
                 dropout: float = 0):
        """
        """
        # Initializes internal Module state.
        super().__init__()

        # input as one sample with dimension len(vocab) with binary (0, 1) elements
        # it is represented in dense form of vector with indices with value 1.
        embedding_layer = nn.Embedding(input_size, embedding_size)
        embedding_layer.weight.data = pretrained_embeddings
        embedding_layer.weight.requires_grad = False

        # embedding layers tranforms binary vector to smaller reprensenation with non-binary elements
        self.embedding = embedding_layer

        # here we create LSTM layer that mostly defines the architecture of the whole RNN
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=lstm_layers_num, bidirectional=bidirectional)

        # final layer is used for transforming vector of hidden_size to the vector of the required output length
        self.final_linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

        # dropout layer to regulize training and avoid overfitting
        self.dropout = nn.Dropout(dropout)

    ## forward function
    def forward(self, raw_input: torch.Tensor):
        """
        """

        # transform raw input to dense embedding vector
        embedded_input = self.embedding(raw_input)
        # add dummy batch dimensionality to match lstm layer input requirements
        embedded_input = embedded_input[None, :, :]
        lstm_output, _ = self.lstm(embedded_input)
        predictions = self.final_linear(self.dropout(lstm_output))

        # remove artificial batch size required for lstm input
        return torch.squeeze(predictions, 0)


class BertPosTagger(nn.Module):
    def __init__(self,
                 bert,
                 output_dim,
                 dropout):
        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']

        self.fc = nn.Linear(embedding_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text: List[int]):
        # text = [batch size, sent len]
        text = text[None, :]

        embedded = self.dropout(self.bert(text)[0])

        # embedded = [batch size, seq len, emb dim]

        embedded = embedded.permute(1, 0, 2)

        # embedded = [sent len, batch size, emb dim]

        predictions = self.fc(self.dropout(embedded))

        # predictions = [sent len, batch size, output dim]

        # remove redundant batch size
        # predictions = [sent len, output dim]
        return torch.squeeze(predictions, 1)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def accuracy(gold_tags, prediction_output) -> float:
    """Calculates accuracy based on gold tags and prediction output."""
    # chooses tag with max probability value as predicted tag for input token
    _, predicted_tags = torch.max(prediction_output.data, 1)
    total_tags = gold_tags.size(0)
    # correct tags is a single scalar tensor, then we call item() to get the underlying value
    correct_tags = (predicted_tags == gold_tags.data).sum().item()

    return correct_tags / total_tags


def train_loop(model: nn.Module, optimizer: torch.optim.Optimizer, loss_function,
               dataloader: DataLoader, epoch: int = 0,
               comet_experiment: Experiment = None, tokens_transformation_func=None) -> Tuple[float, float]:
    """
    Runs one complete training epoch, i. e. trains the model on the whole amount of available training data and updates weights.
    :param model: a pytorch model
    :param optimizer: a pytorch optimizer
    :param loss_function: the type of loss function to use
    :param dataloader: a dataloader for getting the training instances
    :param epoch: optional epoch number for logging purposes
    :param comet_experiment: reference to Comet experiment
    :param

    :return: (epoch_loss, epoch_accuracy): metrics to evaluate the current state of model
    """
    # turns model to train mode: activates dropout and some other features that are useful only for training.
    model.train()

    def _train():
        """Dummy function to handle situation with non-set Comet experiment."""
        accuracy_sum = loss_sum = 0

        for i, item in enumerate(dataloader):
            # simple logging to understand the stage of training process inside one epoch
            if (i + 1) % 1000 == 0:
                print(i + 1, end=" ")

            # two 1d-vector-like Tensors with the same length
            input_tokens, gold_tags = item[0][0], item[0][1]
            if tokens_transformation_func:
                input_tokens = tokens_transformation_func(input_tokens)

            # matrix with vector of tag probabilities as rows for each token
            prediction_output = model.forward(input_tokens)
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
            accuracy_sum += accuracy(gold_tags, prediction_output)

        epoch_accuracy = accuracy_sum / len(dataloader)
        epoch_loss = loss_sum / len(dataloader)
        return epoch_accuracy, epoch_loss

    # this context manager only appends 'train' prefixes to all logged metrics
    if comet_experiment:
        with comet_experiment.train():
            epoch_accuracy, epoch_loss = _train()
            # Log epoch accuracy to Comet.ml
            comet_experiment.log_metric("avg_epoch_accuracy", epoch_accuracy, epoch=epoch)
            comet_experiment.log_metric("avg_epoch_loss", epoch_loss, epoch=epoch)
    else:
        epoch_accuracy, epoch_loss = _train()

    return epoch_loss, epoch_accuracy


def validation_loop(model: nn.Module, loss_function, dataloader: DataLoader, epoch: int = 0,
                    comet_experiment: Experiment = None, tokens_transformation_func=None) -> Tuple[float, float]:
    """
    Runs one complete validation (test) loop, 
    i. e. validates(tests) the model on the whole amount of available validation (test) data and reports quality metrics.

    :param model: a pytorch model
    :param loss_function: the type of loss function to use
    :param dataloader: a dataloader for getting the validation (test) instances

    :return: (epoch_loss, epoch_accuracy): metrics to evaluate the current state of model
    """
    model.eval()

    def _validation():
        # we freeze gradient calculation for validation and, therefore, do not update weights
        # in other steps it is the same process as for training
        with torch.no_grad():
            accuracy_sum = loss_sum = 0
            for i, item in enumerate(dataloader):
                input_tokens, gold_tags = item[0][0], item[0][1]
                if tokens_transformation_func:
                    input_tokens = tokens_transformation_func(input_tokens)
                predicted_tags = model.forward(input_tokens)

                loss_sum += loss_function(predicted_tags, gold_tags).item()
                accuracy_sum += accuracy(gold_tags, predicted_tags)

        epoch_accuracy = accuracy_sum / len(dataloader)
        epoch_loss = loss_sum / len(dataloader)
        return epoch_loss, epoch_accuracy

    if comet_experiment:
        with comet_experiment.validate():
            epoch_loss, epoch_accuracy = _validation()
            # Log epoch accuracy to Comet.ml
            comet_experiment.log_metric("avg_epoch_accuracy", epoch_accuracy, epoch=epoch)
            comet_experiment.log_metric("avg_epoch_loss", epoch_loss, epoch=epoch)
    else:
        epoch_loss, epoch_accuracy = _validation()

    return epoch_loss, epoch_accuracy


def train_model(
        model_to_train: nn.Module, optimizer_class,
        loss_function, train_dataloader: DataLoader, dev_dataloader: DataLoader,
        learning_rate, num_epochs: int,
        save_best_checkpoint: bool = False,
        checkpoint_filename: Optional[str] = None,
        comet_experiment: Optional[Experiment] = None,
        embedding_trasformation_func: Optional[Callable[[torch.LongTensor], torch.LongTensor]] = None,
) -> Tuple[int, float, float, Dict]:
    """"""
    best_epoch = best_accuracy = best_val_loss = float('inf')
    best_model_state = None

    # create optimizer instance
    optimizer = optimizer_class(model_to_train.parameters(), lr=learning_rate)
    # iterates by epochs
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss, train_accuracy = train_loop(model_to_train, optimizer, loss_function, train_dataloader, epoch=epoch,
                                                comet_experiment=comet_experiment,
                                                tokens_transformation_func=embedding_trasformation_func)
        val_loss, val_accuracy = validation_loop(model_to_train, loss_function, dev_dataloader, epoch=epoch,
                                                 comet_experiment=comet_experiment,
                                                 tokens_transformation_func=embedding_trasformation_func)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print()
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f"Train: Loss {train_loss} | Accuracy {train_accuracy}")
        print(f"Validation: Loss {val_loss} | Accuracy {val_accuracy}")

        # saving best checkpoint
        if val_loss < best_val_loss:
            print(f'SAVING BEST MODEL WITH VAL_LOSS={val_loss}.')
            best_val_loss = val_loss
            best_epoch = epoch
            best_accuracy = val_accuracy
            best_model_state = model_to_train.state_dict()
            if save_best_checkpoint:
                torch.save(model_to_train.state_dict(), checkpoint_filename)

    return best_epoch, best_accuracy, best_val_loss, best_model_state


def _predict_pos(trained_model: nn.Module, tagset: Vocab, sentence_numerical_vector: torch.LongTensor) -> List[str]:
    """
    Predicts parts of speech for input German sentence.

    :param trained_model: trained neural model.
    :param sentence: sentence as sequence of tokens or as single string.

    :return: tags: predictes part-of-speech tags
    """
    outputs = trained_model.forward(sentence_numerical_vector)
    _, predicted_tags_indices = torch.max(outputs.data, 1)

    predicted_tags = tagset.lookup_tokens(predicted_tags_indices.numpy())

    return predicted_tags


def predict_pos_lstm(lstm_model: LstmPosTagger, vocab: Vocab, tagset: Vocab,
                     sentence: Union[List[str], str]) -> List[str]:
    """
    Predicts parts of speech for input German sentence.

    :param lstm_model: trained neural model.
    :param sentence: sentence as sequence of tokens or as single string.

    :return: tags: predictes part-of-speech tags
    """
    if isinstance(sentence, str):
        sentence = sentence.split()

    sentence_numerical_vector = torch.LongTensor(vocab.lookup_indices(sentence))
    predicted_tags = _predict_pos(trained_model=lstm_model, tagset=tagset,
                                  sentence_numerical_vector=sentence_numerical_vector)

    return predicted_tags


def predict_pos_bert(bert_model: BertPosTagger, sentence: Union[List[str], str], tagset: Vocab,
                     text_preprocessor: Callable[[List[str]], List[int]]) -> List[str]:
    """
    Predicts parts of speech for input German sentence.

    :param bert_model: trained neural model.
    :param sentence: sentence as sequence of tokens or as single string.

    :return: tags: predictes part-of-speech tags
    """
    if isinstance(sentence, str):
        sentence = sentence.split()

    sentence_numerical_vector = torch.LongTensor(text_preprocessor(sentence))
    predicted_tags = _predict_pos(trained_model=bert_model, tagset=tagset,
                                  sentence_numerical_vector=sentence_numerical_vector)

    return predicted_tags
