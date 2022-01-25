import os
import wget

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from typing import List, Tuple, Union, Dict, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter, defaultdict

PADDING_SYMBOL = '_'
ALPHABET_SIZE = 26
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

alph_to_carka = {'a': 'á', 'e': 'é', 'i': 'í', 'y': 'ý', 'u': 'ú', 'o': 'ó'}
alph_to_hacek = {'e': 'ě', 'c': 'č', 's': 'š', 'z': 'ž', 'r': 'ř'}
alph_to_krouzek = {'u': 'ů'}


def distribution(train_data: List[List[str]], classlabel_to_classname: Dict[int, str],
                 char_to_classlabel: Dict[str, int]):
    accented_words = [row[0] for row in train_data]
    characters = [char for word in accented_words for char in word]

    # dict with char as a key and integer count as value
    char_to_count = Counter(characters)

    # dict with class_label as a key (int) and integer count as value
    classlabel_to_count = {class_label: 0 for class_label in classlabel_to_classname}

    for char, char_count in char_to_count.items():
        # 0 will be returned for non-classified char
        # this default behaviour is handled by defaultclass design
        char_classlabel = char_to_classlabel[char]
        classlabel_to_count[char_classlabel] += char_count

    class_counts = {
        classlabel_to_classname[class_label]: class_count
        for class_label, class_count in classlabel_to_count.items()
    }

    plt.bar(class_counts.keys(), class_counts.values(), 1)
    plt.title('Class Frequency')
    plt.xlabel('Class')
    plt.ylabel('Counts')
    plt.show()

    return class_counts


def download_data_files(filenames: List[str]):
    git_data_path = 'https://raw.githubusercontent.com/tsimafeip/LCT-master-course/main/Neural_Networks/HW9_data/'
    for filename in filenames:
        if not os.path.isfile(filename):
            url = git_data_path + filename
            wget.download(url, filename)


def prepare_sliding_data(data_corpus: List[Union[str, List[str]]],
                         char_to_classlabel: Dict[str, int],
                         padding_symbol=PADDING_SYMBOL):
    """Converts data to sliding window format."""
    window_size = 2
    x_data, y_data = [], []
    # test_data
    if isinstance(data_corpus[0], str):
        # add synthetic second word to reuse code
        data_corpus = [[word, word] for word in data_corpus]

    for accented_word, unaccented_word in data_corpus:
        assert len(accented_word) == len(unaccented_word)
        word_len = len(accented_word)
        for mid in range(word_len):
            word_slice = []
            # left border is inclusive, right border is exclusive
            l, r = mid - window_size, mid + window_size + 1

            # add left padding
            if l < 0:
                word_slice.extend(padding_symbol * abs(l))
            # handle edge case -> add first one first symbol
            if l == -1:
                word_slice.append(unaccented_word[0])
            else:
                word_slice.extend(unaccented_word[l:mid])

            word_slice.extend(unaccented_word[mid:r])
            if r > word_len:
                word_slice.extend(padding_symbol * (r - word_len))

            word_slice_str = "".join(word_slice)
            class_label = char_to_classlabel[accented_word[mid]]

            # yield (word_slice_str, class_label)
            x_data.append(word_slice_str)
            y_data.append(class_label)

    return x_data, y_data


def one_hot_encode(
        word_slice: Union[Tuple[str, int], str],
        padding_symbol: str = PADDING_SYMBOL,
) -> List[int]:
    """
    Makes one-hot vectors (a: (1, 0, 0, ..., 0), z: (0, 0, ...., 1).) 
    and concats them. 

    Parameters
    ----------
    word_slice : Union[Tuple[str, int], str]
        Input in one of the two formats: word_to_class tuple or raw word.
    padding_symbol : str
        Padding symbol which will be encoded as zero-vector for sliding FFNN.

    Returns
    -------
    List[int]
        Vector representation of word_slice.
    """
    res_vector = []
    if isinstance(word_slice, tuple):
        word_slice, _ = word_slice

    for chr in word_slice:
        char_vector = [0] * ALPHABET_SIZE
        if chr != padding_symbol:
            char_vector[ord(chr) - ord('a')] = 1
        res_vector.extend(char_vector)

    return res_vector


class CustomSlidingDataset(Dataset):
    def __init__(self, source_x: List[str], source_y: List[int]):
        self.targets = torch.LongTensor(source_y).to(DEVICE)
        self.source_data = torch.FloatTensor([one_hot_encode(x) for x in source_x]).to(DEVICE)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.source_data[idx], self.targets[idx]


def get_labels_from_predictions(predictions: torch.Tensor) -> np.ndarray:
    softmax_fn = nn.Softmax(dim=1)
    with torch.no_grad():
        softmax_res = softmax_fn(predictions.cpu()).numpy()
        output = np.argmax(softmax_res, axis=1)
        return output


def count_correct_labels(pred_output: torch.Tensor, gold_output: torch.Tensor,
                         predicted_labels_counter: Optional[Dict[int, int]] = None,
                         true_labels_counter: Optional[Dict[int, int]] = None) -> int:
    correct_labels = 0
    pred_labels = get_labels_from_predictions(pred_output)
    for pred_label, gold_label in zip(pred_labels, gold_output.cpu().numpy()):
        correct_labels += (gold_label == pred_label)

        if predicted_labels_counter is not None: predicted_labels_counter[pred_label] += 1
        if true_labels_counter is not None: true_labels_counter[gold_label] += 1

    return correct_labels


def train_loop_ffnn(dataloader: torch.utils.data.DataLoader,
                    model: nn.Module,
                    loss_fn,
                    optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """Train loop function."""
    predicted_labels_counter = defaultdict(int)
    true_labels_counter = defaultdict(int)

    correct_labels = total_labels = 0

    avg_loss = 0
    model.train()
    for batch_id, (x_item, gold_label_tensor) in enumerate(tqdm(dataloader)):
        # Compute prediction and loss
        pred_output = model(x_item)
        loss = loss_fn(pred_output, gold_label_tensor)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        local_correct = count_correct_labels(
            pred_output=pred_output, gold_output=gold_label_tensor,
            predicted_labels_counter=predicted_labels_counter,
            true_labels_counter=true_labels_counter,
        )

        correct_labels += local_correct
        total_labels += gold_label_tensor.size(0)
        avg_loss += loss.item()

    avg_loss /= len(dataloader)
    accuracy = correct_labels / total_labels

    result_metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predicted_labels': predicted_labels_counter,
        'gold_labels': true_labels_counter,
    }

    return result_metrics


def validation_loop_ffnn(dataloader: torch.utils.data.DataLoader,
                         model: nn.Module,
                         loss_fn) -> Dict[str, float]:
    """Validation loop function."""

    correct_labels = total_labels = avg_loss = 0
    dataset_size = len(dataloader)

    predicted_labels_counter = defaultdict(int)
    true_labels_counter = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for batch_id, (x_item, gold_label_tensor) in enumerate(tqdm(dataloader)):
            pred_output = model(x_item)
            avg_loss += loss_fn(pred_output, gold_label_tensor).item()

            local_correct = count_correct_labels(
                pred_output=pred_output, gold_output=gold_label_tensor,
                predicted_labels_counter=predicted_labels_counter,
                true_labels_counter=true_labels_counter,
            )
            correct_labels += local_correct
            total_labels += gold_label_tensor.size(0)

    avg_loss /= dataset_size
    accuracy = correct_labels / total_labels

    result_metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predicted_labels': predicted_labels_counter,
        'gold_labels': true_labels_counter,
    }

    return result_metrics


def run_train_and_val_ffnn(model: nn.Module, train_dataloader: DataLoader, validation_dataloader: DataLoader,
                           epochs: int, best_model_path: str, learning_rate: float = 1e-5):
    best_epoch = best_val_loss = float('inf')
    best_metrics = dict()

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    for cur_epoch in range(epochs):
        print(cur_epoch, end=" ")
        train_metrics = train_loop_ffnn(train_dataloader, model, loss_fn, optimizer)
        val_metrics = validation_loop_ffnn(validation_dataloader, model, loss_fn)

        # saving best checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = cur_epoch
            best_metrics = val_metrics
            # print(f'SAVING BEST MODEL WITH VAL_LOSS={best_val_loss}.')
            torch.save(model.state_dict(), best_model_path)

        if (cur_epoch + 1) % 2 == 0:
            print()
            print(f"Epoch {cur_epoch + 1}\n-------------------------------")
            print('Train metrics: ', train_metrics)
            print('Validation metrics: ', val_metrics)
            print("\n-------------------------------")

    print("Done!")

    return best_model_path, best_epoch, best_metrics


class DiacriticFFNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(130, 72)
        self.bn1 = nn.BatchNorm1d(num_features=72)
        self.linear2 = nn.Linear(72, 4)

        self.dropout = nn.Dropout(0.2)

    def forward(self, input: torch.Tensor):  # Input is a 1D tensor
        y = self.dropout(F.relu(self.bn1(self.linear1(input))))
        y = self.linear2(y)
        return y


def predict_result_fnn(x_test: List[str],
                       model: DiacriticFFNN,
                       path_to_res_file: str = 'sliding_predictions.txt',
                       path_to_best_model: Optional[str] = None):
    def get_testabel_from_ffnn_predictions(predictions: torch.Tensor) -> int:
        softmax_fn = nn.Softmax(dim=1)
        with torch.no_grad():
            output = np.argmax(softmax_fn(predictions)).numpy()
            return int(output)

    if path_to_best_model and os.path.isfile(path_to_best_model):
        model.load_state_dict(torch.load(path_to_best_model))

    model.eval()
    with open(path_to_res_file, 'w') as writefile:
        with torch.no_grad():
            x_word = []
            for x_sliding_item in tqdm(x_test):
                # next word start met, write current word to file
                if x_sliding_item.startswith(PADDING_SYMBOL * 2) and x_word:
                    x_word_str = "".join(x_word)
                    writefile.write(f'{x_word_str}\n')
                    x_word = []

                # put str to vector
                encoded_x = one_hot_encode(x_sliding_item)
                # put vector to Tensor and reshape to create dummy batch
                encoded_x = torch.FloatTensor(encoded_x).view(1, -1).to(DEVICE)
                # pred_ouput is an int denoting class
                pred_output = model(encoded_x).cpu()

                pred_label = get_testabel_from_ffnn_predictions(pred_output)

                mid_char = x_sliding_item[2]
                # If we predict impossible char -> just fall back to original char
                if pred_label == 1:
                    x_word.append(alph_to_carka.get(mid_char, mid_char))
                elif pred_label == 2:
                    x_word.append(alph_to_hacek.get(mid_char, mid_char))
                elif pred_label == 3:
                    x_word.append(alph_to_krouzek.get(mid_char, mid_char))
                else:
                    x_word.append(mid_char)


def prepare_rnn_data(data_corpus: List[Union[str, List[str]]], char_to_classlabel: Dict[str, int]) -> Tuple[
    List[torch.FloatTensor], List[torch.LongTensor]]:
    """Converts data to RNN (cerveny -> HNNNNNANN) format."""
    x_data, y_data = [], []
    # test_data
    if isinstance(data_corpus[0], str):
        # add synthetic second word to reuse code
        data_corpus = [[word, word] for word in data_corpus]

    for accented_word, unaccented_word in data_corpus:
        assert len(accented_word) == len(unaccented_word)

        input_sequence = torch.FloatTensor([one_hot_encode(unaccented_ch) for unaccented_ch in unaccented_word])
        output_sequence = torch.LongTensor([char_to_classlabel[accented_ch] for accented_ch in accented_word])

        x_data.append(input_sequence)
        y_data.append(output_sequence)

    return x_data, y_data


class DiacriticsRNN(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 lstm_layers_num: int = 1,
                 bidirectional: bool = True,
                 dropout: float = 0):
        # Initializes internal Module state.
        super().__init__()

        # here we create LSTM layer that mostly defines the architecture of the whole RNN
        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=True,
                            num_layers=lstm_layers_num,
                            bidirectional=bidirectional)

        # final layer is used for transforming vector of hidden_size to the vector of the required output length
        self.final_linear = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)

        # dropout layer to regulize training and avoid overfitting
        self.dropout = nn.Dropout(dropout)

    # forward function
    def forward(self, embedded_input: torch.FloatTensor):
        """
        Forward path of the model.

        embedded_input: str - one-hot-embedded torch.Tensor of shape (sequence_len, ALPHABET_SIZE)
        """
        sequence_length = embedded_input.size(0)

        # add dummy batch_size 1 as a first dimension 
        # resulting tensor has shape (1, sequence_len, ALPHABET_SIZE)
        embedded_input = embedded_input.view(1, sequence_length, -1)

        # lstm_output has two possible shapes:
        # (1, sequence_len, hidden_size) for unidirectional LSTM model
        # (1, sequence_len, 2*hidden_size) for bidirectional LSTM model
        lstm_output, _ = self.lstm(embedded_input)

        # predictions has shape (1, sequence_len, num_classes)
        predictions = self.final_linear(self.dropout(lstm_output))

        # reshaping to remove artificial batch dimension
        return predictions.view(sequence_length, -1)


def train_loop_rnn(x_data: List[torch.FloatTensor], y_data: List[torch.LongTensor],
                   model: nn.Module,
                   loss_fn,
                   optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    """Train loop function."""
    assert len(x_data) == len(y_data)

    predicted_labels_counter = defaultdict(int)
    true_labels_counter = defaultdict(int)

    correct_labels = total_labels = 0

    avg_loss = 0
    model.train()
    for batch_id, (x_tensor, gold_label_tensor) in enumerate(tqdm(zip(x_data, y_data))):
        # Compute prediction and loss
        pred_output = model(x_tensor)
        loss = loss_fn(pred_output, gold_label_tensor)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        local_correct = count_correct_labels(
            pred_output=pred_output, gold_output=gold_label_tensor,
            predicted_labels_counter=predicted_labels_counter,
            true_labels_counter=true_labels_counter,
        )

        correct_labels += local_correct
        total_labels += gold_label_tensor.size(0)
        avg_loss += loss.item()

    avg_loss /= len(x_data)
    accuracy = correct_labels / total_labels

    result_metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predicted_labels': predicted_labels_counter,
        'gold_labels': true_labels_counter,
    }

    return result_metrics


def validation_loop_rnn(x_data: List[torch.FloatTensor], y_data: List[torch.LongTensor],
                        model: nn.Module,
                        loss_fn) -> Dict[str, float]:
    """Validation loop function."""
    assert len(x_data) == len(y_data)

    correct_labels = total_labels = avg_loss = 0
    dataset_size = len(x_data)

    predicted_labels_counter = defaultdict(int)
    true_labels_counter = defaultdict(int)
    model.eval()
    with torch.no_grad():
        for batch_id, (x_tensor, gold_label_tensor) in enumerate(tqdm(zip(x_data, y_data))):
            pred_output = model(x_tensor)
            avg_loss += loss_fn(pred_output, gold_label_tensor).item()

            local_correct = count_correct_labels(
                pred_output=pred_output, gold_output=gold_label_tensor,
                predicted_labels_counter=predicted_labels_counter,
                true_labels_counter=true_labels_counter,
            )
            correct_labels += local_correct
            total_labels += gold_label_tensor.size(0)

    avg_loss /= dataset_size
    accuracy = correct_labels / total_labels

    result_metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predicted_labels': predicted_labels_counter,
        'gold_labels': true_labels_counter,
    }

    return result_metrics


def run_train_and_val_rnn(model: nn.Module, x_train: List[torch.FloatTensor], y_train: List[torch.LongTensor],
                          x_val: List[torch.FloatTensor], y_val: List[torch.LongTensor],
                          epochs: int, path_to_best_model_file: str = 'best-rnn-model.pt', learning_rate: float = 1e-5):
    best_epoch = best_val_loss = float('inf')
    best_metrics = dict()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    for cur_epoch in range(epochs):
        print(cur_epoch, end=" ")
        train_metrics = train_loop_rnn(x_train, y_train, model, loss_fn, optimizer)
        val_metrics = validation_loop_rnn(x_val, y_val, model, loss_fn)

        # saving best checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_epoch = cur_epoch
            best_metrics = val_metrics
            # print(f'SAVING BEST MODEL WITH VAL_LOSS={best_val_loss}.')
            torch.save(model.state_dict(), path_to_best_model_file)

        if (cur_epoch + 1) % 1 == 0:
            print()
            print(f"Epoch {cur_epoch + 1}\n-------------------------------")
            print('Train metrics: ', train_metrics)
            print('Validation metrics: ', val_metrics)
            print("\n-------------------------------")

    print("Done!")

    return path_to_best_model_file, best_epoch, best_metrics


def predict_result_rnn(raw_x_test: List[str],
                       tensor_x_test: List[torch.FloatTensor],
                       model: DiacriticsRNN,
                       path_to_res_file: str = 'rnn_predictions.txt',
                       path_to_best_model: Optional[str] = None):
    if path_to_best_model and os.path.isfile(path_to_best_model):
        model.load_state_dict(torch.load(path_to_best_model))

    model.eval()
    with open(path_to_res_file, 'w') as writefile:
        with torch.no_grad():
            for input_word, input_tensor in tqdm(zip(raw_x_test, tensor_x_test)):
                # pred_ouput is an int denoting class
                pred_output = model(input_tensor)

                _, pred_labels = torch.max(pred_output.data, 1)

                diacritic_word = []
                for input_ch, pred_label in zip(input_word, pred_labels):
                    if pred_label == 1:
                        diacritic_word.append(alph_to_carka.get(input_ch, input_ch))
                    elif pred_label == 2:
                        diacritic_word.append(alph_to_hacek.get(input_ch, input_ch))
                    elif pred_label == 3:
                        diacritic_word.append(alph_to_krouzek.get(input_ch, input_ch))
                    else:
                        diacritic_word.append(input_ch)

                diacritic_word_str = "".join(diacritic_word)
                writefile.write(f"{diacritic_word_str}\n")
