import os
import wget

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from typing import List, Tuple, Union, Dict, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter, defaultdict

PADDING_SYMBOL = '_'
ALPHABET_SIZE = 26
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def prepare_sliding_data(data_corpus: List[Union[str, List[str]]], char_to_classlabel: Dict[str, int], window_size=2,
                         padding_symbol='_'):
    """Converts data to sliding window format."""
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


def encode_sliding_data(
        word_slice: Union[Tuple[str, int], str],
        padding_symbol: str = PADDING_SYMBOL,
) -> List[int]:
    """
    Makes one-hot vectors (a: (1, 0, 0, ..., 0), z: (0, 0, ...., 1).) 
    and concats them. 
    Final size of every input is ALPHABET_SIZE*len(word_slice) (here, 26*5=130).

    Parameters
    ----------
    word_slice : Union[Tuple[str, int], str]
        Word slice in one of the two formats: slice_to_class tuple or raw slice.
    padding_symbol : str
        Padding symbol which will be encoded as zero-vector.

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

    assert len(res_vector) == ALPHABET_SIZE * 5

    return res_vector


class CustomSlidingDataset(Dataset):
    def __init__(self, source_x: List[str], source_y: List[int]):
        self.targets = torch.LongTensor(source_y).to(DEVICE)
        self.source_data = torch.FloatTensor([encode_sliding_data(x) for x in source_x]).to(DEVICE)

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


def run_train_and_val(model: nn.Module, train_dataloader: DataLoader, validation_dataloader: DataLoader,
                      epochs: int, learning_rate: float = 1e-5):
    best_val_loss = float('inf')
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
            torch.save(model.state_dict(), 'best-model.pt')

        if (cur_epoch + 1) % 2 == 0:
            print()
            print(f"Epoch {cur_epoch + 1}\n-------------------------------")
            print('Train metrics: ', train_metrics)
            print('Validation metrics: ', val_metrics)
            print("\n-------------------------------")

    print("Done!")

    return best_val_loss, best_metrics
