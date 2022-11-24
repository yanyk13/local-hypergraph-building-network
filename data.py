import json
import torch
import numpy as np
import _pickle as pkl

from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer
from pprint import pprint
from math import ceil

from utils import ctext
from utils import check_gpu_device
from utils import compute_classify_performance


def len_statistic(samples):
    sta = {}
    length = [len(sample) for sample in samples]
    for l in length:
        if l in sta.keys():
            sta[l] += 1
        else:
            sta[l] = 1
    pprint(sta)


def load_dict(root_path):
    with open('{}/char2id.pkl'.format(root_path), 'rb') as f:
        char2id = pkl.load(f)
    with open('{}/pos2id.pkl'.format(root_path), 'rb') as f:
        pos2id = pkl.load(f)
    with open('{}/word2id.pkl'.format(root_path), 'rb') as f:
        word2id = pkl.load(f)
    with open('{}/type2id.pkl'.format(root_path), 'rb') as f:
        type2id = pkl.load(f)
    dicts = {
        'char2id': char2id,
        'word2id': word2id,
        'pos2id': pos2id,
        'type2id': type2id
    }
    print('char num: {}'.format(len(list(char2id.keys()))))
    print('pos num: {}'.format(len(list(pos2id.keys()))))
    print('word num: {}'.format(len(list(word2id.keys()))))
    print('type num: {}'.format(len(list(type2id.keys()))))
    return dicts


class ContextData(Dataset):

    def __init__(self, json_path, dicts, tokenizer, name='', test=False, min_len=0, max_len=-1):
        with open(json_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            num_orig = len(samples)
            if max_len != -1:
                self.samples = list(filter(lambda x: min_len < len(x['tokens']) <= max_len, samples))
                num_filtered = len(self.samples)
                print(ctext('length {}~{}: {}% samples'.format(min_len, max_len,
                                                               round(float(num_filtered)/num_orig, 4) * 100), 'blue'))
            else:
                self.samples = samples
        if test:
            self.samples = self.samples[:1]
        # len_statistic([x['tokens'] for x in self.samples])
        self.name = name
        self.word2id = dicts['word2id']
        self.char2id = dicts['char2id']
        self.pos2id = dicts['pos2id']
        self.type2id = dicts['type2id']
        self.tokenizer = tokenizer
        self._wash()
        # print(self.samples[0].keys())

    def convert_tokens_to_ids(self, sample):
        bert_id = self.tokenizer.convert_tokens_to_ids(sample['tokens'])
        char_id = [[self.char2id[char] for char in list(token)] for token in sample['tokens']]
        pos_id = [self.pos2id[pos] if pos in self.pos2id.keys() else self.pos2id['[unk]'] for pos in
                  sample['pos']]
        word_id = [self.word2id[token] if token in self.word2id.keys() else self.word2id['[unk]'] for token in
                   sample['tokens']]
        ids = {
            "bert_id": bert_id,
            "char_id": char_id,
            "pos_id": pos_id,
            "word_id": word_id
        }
        return ids

    def _wash(self):
        for sample_id, sample in tqdm(enumerate(self.samples), desc=ctext('washing', 'cyan'), total=len(self.samples)):
            ids = self.convert_tokens_to_ids(sample)
            entities = sample['entities']
            span2type = {}
            start2end = {}
            end2start = {}
            start_label = np.zeros(sample["sentence_boundary"][1] - sample["sentence_boundary"][0])
            end_label = np.zeros(sample["sentence_boundary"][1] - sample["sentence_boundary"][0])
            for entity in entities:
                start = entity['start']
                end = entity['end']
                start_label[start] = 1.
                end_label[end - 1] = 1.

                if '{}-{}'.format(start, end) in span2type.keys():
                    tokens = sample['tokens']
                    [lb, rb] = sample['sentence_boundary']
                    content = tokens[lb: rb]
                    print(content[start: end])
                    print(entity['type'])
                    print(list(self.type2id.keys())[span2type['{}-{}'.format(start, end)]])
                span2type['{}-{}'.format(start, end)] = self.type2id[entity['type']]

                if start not in start2end.keys():
                    start2end[start] = [end]
                else:
                    start2end[start].append(end)
                if end not in end2start.keys():
                    end2start[end] = [start]
                else:
                    end2start[end].append(start)
            sample['start_label'] = start_label
            sample['end_label'] = end_label
            sample['span2type'] = span2type
            sample['start2end'] = start2end
            sample['end2start'] = end2start
            sample.update(ids)

    def load_samples(self, batch_size):
        num_samples = len(self.samples)
        num_batch = ceil(num_samples/float(batch_size))
        for i in range(num_batch):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            yield context_collate(self.samples[start_idx: end_idx])

    def gen_start(self, start_identifier, max_coe=2, batch_size=128, train=True, extend=1):
        with torch.no_grad():
            start_identifier.eval()
            batches = self.load_samples(batch_size)
            base_num = 0
            predicted_scores = []
            true_labels = []
            for batch in tqdm(batches, total=ceil(len(self.samples)/float(batch_size)),
                              desc="{}: predicting start boundary".format(self.name)):
                scores = torch.sigmoid(start_identifier.forward(batch).squeeze(-1))
                mask = check_gpu_device(batch['content_mask'])
                scores = scores * mask
                base_num += torch.sum(scores.gt(0.5).int())
                predicted_scores.extend([score for score in scores])
                true_labels.extend([start_label for start_label in batch['start_label']])
            predicted_scores = pad_sequence(
                sequences=predicted_scores,
                padding_value=0,
                batch_first=True
            )
            true_labels = pad_sequence(
                sequences=true_labels,
                padding_value=0,
                batch_first=True
            )
            ranked_scores, _ = predicted_scores.view(-1).sort(descending=True)
            if train:
                coes = [max_coe]
            else:
                coes = [round(0.01 * coe, 2) for coe in range(90, int(100 * max_coe) + 1)]
            if ceil(base_num * max_coe) >= len(ranked_scores):
                thresholds = [0]
            else:
                thresholds = [ranked_scores[ceil(base_num * coe)] for coe in coes]

            performances = [compute_classify_performance(predict_scores=predicted_scores.view(-1),
                                                         true_label=check_gpu_device(true_labels).view(-1),
                                                         threshold=threshold) for threshold in thresholds]
            max_threshold = min(thresholds)
            for sample, score in tqdm(zip(self.samples, predicted_scores),
                                      total=len(self.samples),
                                      desc='{}: adding start boundary'.format(self.name)):
                sample['start'] = score.gt(max_threshold).int().nonzero().view(-1).tolist()
                if len(sample['start']) != 0:
                    if max(sample['start']) > sample['sentence_boundary'][1] - sample['sentence_boundary'][0]:
                        print(sample['start'])
                        print(len(sample['tokens']))
                sub_sequence_label = []
                sub_sequence_boundary = []
                start_coe = []
                boundary = sample['sentence_boundary'][1] - sample['sentence_boundary'][0]
                for start in sample['start']:
                    if start in sample['start2end'].keys():
                        max_end = max(sample['start2end'][start])
                        if train:
                            sequence_label = [0] * (min(max_end + extend, boundary) - start)
                            sequence_boundary = [start, min(max_end + extend, boundary)]
                        else:
                            sequence_label = [0] * (boundary - start)
                            sequence_boundary = [start, boundary]
                        sequence_label[: max_end - start] = [self.type2id['Continue']] * (max_end - start)
                        for end in sample['start2end'][start]:
                            span = '{}-{}'.format(start, end)
                            sequence_label[end - 1 - start] = sample['span2type'][span]
                    else:
                        if train:
                            sequence_label = [0] * (min(start + extend, boundary) - start)
                            sequence_boundary = [start, min(start + extend, boundary)]
                        else:
                            sequence_label = [0] * (boundary - start)
                            sequence_boundary = [start, boundary]
                    sub_sequence_label.append(sequence_label)
                    sub_sequence_boundary.append(sequence_boundary)
                    for coe, threshold in zip(coes, thresholds):
                        if threshold < score[start]:
                            start_coe.append(coe)
                            break
                sample['start_coe'] = start_coe
                sample['sub_sequence_label'] = sub_sequence_label
                sample['sub_sequence_boundary'] = sub_sequence_boundary
            info = {}
            for coe, threshold, (precision, recall, f1, _) in zip(coes, thresholds, performances):
                info.update({
                    'coe': coe,
                    'threshold': threshold,
                    'precision': round(precision, 3),
                    'recall': round(recall, 3),
                    'micro_f1': round(f1, 3)
                })
            torch.cuda.empty_cache()
            return info
        
    def gen_end(self, end_identifier, max_coe=2, batch_size=128, train=True, extend=1):
        with torch.no_grad():
            end_identifier.eval()
            batches = self.load_samples(batch_size)
            base_num = 0
            predicted_scores = []
            true_labels = []
            for batch in tqdm(batches, total=ceil(len(self.samples)/float(batch_size)),
                              desc="{}: predicting end boundary".format(self.name)):
                scores = torch.sigmoid(end_identifier.forward(batch).squeeze(-1))
                mask = check_gpu_device(batch['content_mask'])
                scores = scores * mask
                base_num += torch.sum(scores.gt(0.5).int())
                predicted_scores.extend([score for score in scores])
                true_labels.extend([end_label for end_label in batch['end_label']])
            predicted_scores = pad_sequence(
                sequences=predicted_scores,
                padding_value=0,
                batch_first=True
            )
            true_labels = pad_sequence(
                sequences=true_labels,
                padding_value=0,
                batch_first=True
            )
            ranked_scores, _ = predicted_scores.view(-1).sort(descending=True)
            if train:
                coes = [max_coe]
            else:
                coes = [round(0.01 * coe, 2) for coe in range(90, int(100 * max_coe) + 1)]
            thresholds = [ranked_scores[ceil(base_num * coe)] for coe in coes]

            performances = [compute_classify_performance(predict_scores=predicted_scores.view(-1),
                                                         true_label=check_gpu_device(true_labels).view(-1),
                                                         threshold=threshold) for threshold in thresholds]
            max_threshold = min(thresholds)
            for sample, score in tqdm(zip(self.samples, predicted_scores),
                                      total=len(self.samples),
                                      desc='{}: adding end boundary'.format(self.name)):
                sample['end'] = [x + 1 for x in score.gt(max_threshold).int().nonzero().view(-1).tolist()]
                if len(sample['end']) != 0:
                    if max(sample['end']) > sample['sentence_boundary'][1] - sample['sentence_boundary'][0]:
                        print(sample['end'])
                        print(len(sample['tokens']))
                sub_sequence_label = []
                sub_sequence_boundary = []
                end_coe = []
                for end in sample['end']:
                    if end in sample['end2start'].keys():
                        min_start = min(sample['end2start'][end])
                        if train:
                            sequence_label = [0] * (end - max(min_start - extend, 0))
                            sequence_boundary = [max(min_start - extend, 0), end]
                        else:
                            sequence_label = [0] * end
                            sequence_boundary = [0, end]
                        sequence_label[: end - min_start] = [self.type2id['Continue']] * (end - min_start)
                        for start in sample['end2start'][end]:
                            span = '{}-{}'.format(start, end)
                            sequence_label[end - 1 - start] = sample['span2type'][span]  #
                    else:
                        if train:
                            sequence_label = [0] * (end - max(end - extend, 0))
                            sequence_boundary = [max(end - extend, 0), end]
                        else:
                            sequence_label = [0] * end
                            sequence_boundary = [0, end]
                    sub_sequence_label.append(sequence_label)
                    sub_sequence_boundary.append(sequence_boundary)
                    for coe, threshold in zip(coes, thresholds):
                        if threshold < score[end - 1]:
                            end_coe.append(coe)
                            break
                sample['end_coe'] = end_coe
                sample['sub_sequence_label_back'] = sub_sequence_label
                sample['sub_sequence_boundary_back'] = sub_sequence_boundary
            info = {}
            for coe, threshold, (precision, recall, f1, _) in zip(coes, thresholds, performances):
                info.update({
                    'coe': coe,
                    'threshold': threshold,
                    'precision': round(precision, 3),
                    'recall': round(recall, 3),
                    'micro_f1': round(f1, 3)
                })
            return info

    def __add__(self, other):
        self.samples += other.samples

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)


class PassageData(Dataset):

    def __init__(self, json_path, dicts, tokenizer, name='', min_len=0, max_len=-1, test=False):
        with open(json_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        num_orig = len(samples)
        if max_len != -1:
            self.samples = list(filter(lambda x: min_len < len(x['tokens']) <= max_len, samples))
            num_filtered = len(self.samples)
            print(ctext('length {}~{}: {}% samples'.format(min_len, max_len,
                                                           round(float(num_filtered) / num_orig, 4) * 100), 'blue'))
        else:
            self.samples = samples
        if test:
            self.samples = self.samples[:10]
        self.name = name
        self.word2id = dicts['word2id']
        self.char2id = dicts['char2id']
        self.pos2id = dicts['pos2id']
        self.type2id = dicts['type2id']
        self.tokenizer = tokenizer
        self._wash()
        # print(self.samples[0].keys())

    def convert_tokens_to_ids(self, sample):
        bert_id = self.tokenizer.convert_tokens_to_ids(sample['tokens'])
        char_id = [[self.char2id[char] for char in list(token)] for token in sample['tokens']]
        pos_id = [self.pos2id[pos] if pos in self.pos2id.keys() else self.pos2id['[unk]'] for pos in
                   sample['pos']]
        word_id = [self.word2id[token] if token in self.word2id.keys() else self.word2id['[unk]'] for token in
                    sample['tokens']]
        ids = {
            "bert_id": bert_id,
            "char_id": char_id,
            "pos_id": pos_id,
            "word_id": word_id
        }
        return ids

    def _wash(self):
        for sample_id, sample in tqdm(enumerate(self.samples), desc=ctext('washing', 'cyan'), total=len(self.samples)):
            tokens = sample['tokens']
            ids = self.convert_tokens_to_ids(sample)
            entities = sample['entities']
            span2type = {}
            start2end = {}
            end2start = {}
            start_label = np.zeros(len(tokens))
            end_label = np.zeros(len(tokens))
            for entity in entities:
                start = entity['start']
                end = entity['end']
                start_label[start] = 1.
                end_label[end - 1] = 1.
                span2type['{}-{}'.format(start, end)] = self.type2id[entity['type']]
                if start not in start2end.keys():
                    start2end[start] = [end]
                else:
                    start2end[start].append(end)
                if end not in end2start.keys():
                    end2start[end] = [start]
                else:
                    end2start[end].append(start)
            sample['start_label'] = start_label
            sample['end_label'] = end_label
            sample['span2type'] = span2type
            sample['start2end'] = start2end
            sample['end2start'] = end2start
            sample.update(ids)

    def load_samples(self, batch_size):
        num_samples = len(self.samples)
        num_batch = ceil(num_samples/float(batch_size))
        for i in range(num_batch):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            yield passage_collate(self.samples[start_idx: end_idx])

    def gen_start(self, start_identifier, max_coe=2, batch_size=64, train=True, extend=1):
        with torch.no_grad():
            start_identifier.eval()
            batches = self.load_samples(batch_size)
            base_num = 0
            predicted_scores = []
            true_labels = []
            for batch in tqdm(batches, total=ceil(len(self.samples)/float(batch_size)),
                              desc="{}: predicting start boundary".format(self.name)):
                scores = torch.sigmoid(start_identifier.forward(batch).squeeze(-1))
                mask = check_gpu_device(batch['mask'])
                scores = scores * mask
                base_num += torch.sum(scores.gt(0.5).int())
                predicted_scores.extend([score for score in scores])
                true_labels.extend([start_label for start_label in batch['start_label']])
            predicted_scores = pad_sequence(
                sequences=predicted_scores,
                padding_value=0,
                batch_first=True
            )
            true_labels = pad_sequence(
                sequences=true_labels,
                padding_value=0,
                batch_first=True
            )
            ranked_scores, _ = predicted_scores.view(-1).sort(descending=True)
            if train:
                coes = [max_coe]
            else:
                coes = [round(0.01 * coe, 2) for coe in range(90, int(100 * max_coe) + 1)]
            if ceil(base_num * max_coe) >= len(ranked_scores):
                thresholds = [0]
            else:
                thresholds = [ranked_scores[ceil(base_num * coe)] for coe in coes]
            performances = [compute_classify_performance(predict_scores=predicted_scores.view(-1),
                                                         true_label=check_gpu_device(true_labels).view(-1),
                                                         threshold=threshold) for threshold in thresholds]
            max_threshold = min(thresholds)
            for sample, score in tqdm(zip(self.samples, predicted_scores),
                                      total=len(self.samples),
                                      desc='{}: adding start boundary'.format(self.name)):
                sample['start'] = score.gt(max_threshold).int().nonzero().view(-1).tolist()
                start2sen = []
                sub_sequence_label = []
                sub_sequence_boundary = []
                start_coe = []
                sentence_boundary = sample['sentence_boundary']
                for start in sample['start']:
                    for end_boundary in sentence_boundary[1:]:
                        if start < end_boundary:
                            # start2sen.append(end_boundary)
                            break
                    if start in sample['start2end'].keys():
                        max_end = max(sample['start2end'][start])
                        if train:
                            sequence_label = [0] * (min(max_end + extend, end_boundary) - start)
                            sequence_boundary = [start, min(max_end + extend, end_boundary)]
                        else:
                            sequence_label = [0] * (end_boundary - start)
                            sequence_boundary = [start, end_boundary]
                        sequence_label[: max_end - start] = [self.type2id['Continue']] * (max_end - start)
                        for end in sample['start2end'][start]:
                            span = '{}-{}'.format(start, end)
                            sequence_label[end - 1 - start] = sample['span2type'][span]
                    else:
                        if train:
                            sequence_label = [0] * (min(start + extend, end_boundary) - start)
                            sequence_boundary = [start, min(start + extend, end_boundary)]
                        else:
                            sequence_label = [0] * (end_boundary - start)
                            sequence_boundary = [start, end_boundary]
                    sub_sequence_label.append(sequence_label)
                    sub_sequence_boundary.append(sequence_boundary)
                    for coe, threshold in zip(coes, thresholds):
                        if threshold < score[start]:
                            start_coe.append(coe)
                            break
                sample['start_coe'] = start_coe
                # sample['start2sen'] = start2sen
                sample['sub_sequence_label'] = sub_sequence_label
                sample['sub_sequence_boundary'] = sub_sequence_boundary
            info = {}
            for coe, threshold, (precision, recall, f1, _) in zip(coes, thresholds, performances):
                info.update({
                    'coe': coe,
                    'threshold': threshold,
                    'precision': round(precision, 3),
                    'recall': round(recall, 3),
                    'micro_f1': round(f1, 3)
                })
            return info

    def gen_end(self, end_identifier, max_coe=2, batch_size=64, train=True, extend=1):
        with torch.no_grad():
            end_identifier.eval()
            batches = self.load_samples(batch_size)
            base_num = 0
            predicted_scores = []
            true_labels = []
            for batch in tqdm(batches, total=ceil(len(self.samples)/float(batch_size)),
                              desc="{}: predicting end boundary".format(self.name)):
                scores = torch.sigmoid(end_identifier.forward(batch).squeeze(-1))
                mask = check_gpu_device(batch['mask'])
                scores = scores * mask
                base_num += torch.sum(scores.gt(0.5).int())
                predicted_scores.extend([score for score in scores])
                true_labels.extend([end_label for end_label in batch['end_label']])
            predicted_scores = pad_sequence(
                sequences=predicted_scores,
                padding_value=0,
                batch_first=True
            )
            true_labels = pad_sequence(
                sequences=true_labels,
                padding_value=0,
                batch_first=True
            )
            ranked_scores, _ = predicted_scores.view(-1).sort(descending=True)
            if train:
                coes = [max_coe]
            else:
                coes = [round(0.01 * coe, 2) for coe in range(90, int(100 * max_coe) + 1)]
            thresholds = [ranked_scores[ceil(base_num * coe)] for coe in coes]

            performances = [compute_classify_performance(predict_scores=predicted_scores.view(-1),
                                                         true_label=check_gpu_device(true_labels).view(-1),
                                                         threshold=threshold) for threshold in thresholds]
            max_threshold = min(thresholds)
            for sample, score in tqdm(zip(self.samples, predicted_scores),
                                      total=len(self.samples),
                                      desc='{}: adding end boundary'.format(self.name)):
                sample['end'] = [end + 1 for end in score.gt(max_threshold).int().nonzero().view(-1).tolist()]
                end2sen = []
                sub_sequence_label = []
                sub_sequence_boundary = []
                end_coe = []
                sentence_boundary = sample['sentence_boundary']
                for end in sample['end']:
                    for boundary in sentence_boundary[:-1]:
                        if boundary >= end:
                            break
                        start_boundary = boundary
                    end2sen.append(boundary)
                    if end in sample['end2start'].keys():
                        min_start = min(sample['end2start'][end])
                        if train:
                            sequence_label = [0] * (end - max(min_start - extend, start_boundary))
                            sequence_boundary = [max(min_start - extend, start_boundary), end]
                        else:
                            sequence_label = [0] * (end - start_boundary)
                            sequence_boundary = [start_boundary, end]
                        sequence_label[: end - min_start] = [self.type2id['Continue']] * (end - min_start)
                        for start in sample['end2start'][end]:
                            span = '{}-{}'.format(start, end)
                            sequence_label[end - 1 - start] = sample['span2type'][span]
                    else:
                        if train:
                            sequence_label = [0] * (end - max(end - extend, start_boundary))
                            sequence_boundary = [max(end - extend, start_boundary), end]
                        else:
                            sequence_label = [0] * (end - start_boundary)
                            sequence_boundary = [start_boundary, end]
                    sub_sequence_label.append(sequence_label)
                    sub_sequence_boundary.append(sequence_boundary)
                    for coe, threshold in zip(coes, thresholds):
                        if threshold < score[end - 1]:
                            end_coe.append(coe)
                            break
                sample['end_coe'] = end_coe
                sample['end2sen'] = end2sen
                sample['sub_sequence_label_back'] = sub_sequence_label
                sample['sub_sequence_boundary_back'] = sub_sequence_boundary
            info = {}
            for coe, threshold, (precision, recall, f1, _) in zip(coes, thresholds, performances):
                info.update({
                    'coe': coe,
                    'threshold': threshold,
                    'precision': round(precision, 3),
                    'recall': round(recall, 3),
                    'micro_f1': round(f1, 3)
                })
            return info

    def __add__(self, other):
        self.samples += other.samples

    def __getitem__(self, item):
        return self.samples[item]

    def __len__(self):
        return len(self.samples)


def context_collate(batch):
    data = {}
    data_elements = list(batch[0].keys())
    stack_keys = ["id", "tokens", "pos", "chars", "entities", "sentence_boundary",
                  "span2type", "start2end", "end2start",
                  "start", "start_coe", "sub_sequence_label", "sub_sequence_boundary",
                  "end", "end_coe", "sub_sequence_label_back", "sub_sequence_boundary_back"]
    padding_keys = ["bert_id", "pos_id", "word_id", "start_label", "end_label"]
    special_keys = ["length", "mask", "content_mask", "char_id", "char_length"]
    for key in data_elements:
        if key in stack_keys:
            data[key] = [sample[key] for sample in batch]
        elif key in padding_keys:
            data[key] = pad_sequence(
                sequences=[torch.tensor(sample[key]) for sample in batch],
                batch_first=True,
                padding_value=0
            )
        elif key in special_keys:
            pass
        else:
            raise ValueError('unregistered key {}'.format(key))
    data['length'] = [len(sample['tokens']) for sample in batch]
    data["mask"] = pad_sequence(
        sequences=[torch.ones(len(sample['tokens'])) for sample in batch],
        batch_first=True,
        padding_value=0.
    )
    data["content_mask"] = pad_sequence(
        sequences=[torch.ones(sample['sentence_boundary'][1] - sample['sentence_boundary'][0]) for sample in batch],
        batch_first=True,
        padding_value=0.
    )
    batch_chars = []
    char_len = []
    for sample in batch:
        sample_char = sample['char_id']
        batch_chars.extend(sample_char)
        char_len.extend([len(char) for char in sample_char])

    data['char_id'] = pad_sequence(
        sequences=[torch.tensor(char_idx) for char_idx in batch_chars],
        batch_first=True,
        padding_value=0.
    )
    data['char_length'] = char_len
    return data


def passage_collate(batch):
    data = {}
    data_elements = list(batch[0].keys())
    stack_keys = ["id", "tokens", "pos", "entities", "sentences",
                  "span2type", "start2end", "end2start", "sentence_boundary",
                  "start", "start2sen", "sub_sequence_label", "sub_sequence_boundary",
                  "start_coe",
                  "end", "end2sen", "sub_sequence_label_back", "sub_sequence_boundary_back",
                  "end_coe"]
    padding_keys = ["bert_id", "pos_id", "word_id", "start_label", "end_label"]
    special_keys = ["mask", "char_id", "char_length", "length"]
    for key in data_elements:
        if key in stack_keys:
            data[key] = [sample[key] for sample in batch]
        elif key in padding_keys:
            data[key] = pad_sequence(
                sequences=[torch.tensor(sample[key]) for sample in batch],
                batch_first=True,
                padding_value=0
            )
        elif key in special_keys:
            pass
        else:
            raise ValueError('unregistered key {}'.format(key))

    data["length"] = [len(sample["tokens"]) for sample in batch]
    data["mask"] = pad_sequence(
        sequences=[torch.ones(len(sample['tokens'])) for sample in batch],
        batch_first=True,
        padding_value=0.
    )
    data['content_mask'] = data["mask"]
    batch_chars = []
    char_len = []
    for sample in batch:
        sample_char = sample['char_id']
        batch_chars.extend(sample_char)
        char_len.extend([len(char) for char in sample_char])

    data['char_id'] = pad_sequence(
        sequences=[torch.tensor(char_idx) for char_idx in batch_chars],
        batch_first=True,
        padding_value=0.
    )
    data['char_length'] = char_len
    return data


if __name__ == '__main__':
    from utils import set_random_seed
    set_random_seed()
    # tokenizer = BertTokenizer.from_pretrained('./biobert-large-cased-v1.1-squad',
    #                                           do_lower_cased=False)
    # dicts = load_dict(root_path='./dicts/genia')
    # context_data = ContextData(json_path='./data/genia/context_train.json',
    #                            dicts=dicts,
    #                            tokenizer=tokenizer,
    #                            test=True)
    #
    # sample = context_data[1]
    # tokens = sample['tokens']
    # entities = sample['entities']
    # boundary = sample['sentence_boundary']
    # start_label = sample['start_label']
    # content = tokens[boundary[0]: boundary[1]]
    #
    # print(content)
    # for entity in entities:
    #     start = entity['start']
    #     end = entity['end']
    #     cls = entity['type']
    #     print(cls, content[start: end])
    #
    # context_data_loader = DataLoader(context_data,
    #                                  collate_fn=context_collate)

    # passage_data_loader = DataLoader(passage_data,
    #                                  collate_fn=passage_collate)

    # for passage_sample, context_sample in zip(passage_data_loader, context_data_loader):
    #     print(passage_sample['sentence_boundary'])
    #     print(context_sample['sentence_boundary'])
    #     passage_keys = list(passage_sample.keys())
    #     context_keys = list(context_sample.keys())
    #     for key in passage_keys:
    #         if key not in context_keys:
    #             print('not in context: {}'.format(key))
    #     for key in context_keys:
    #         if key not in passage_keys:
    #             print('not in passage: {}'.format(key))
    #     break
    # passage_sample = passage_data[0]
    # context_sample = context_data[0]
    # print(passage_sample.keys())
    # print(context_sample.keys())
    # tokenizer = BertTokenizer.from_pretrained('./bert-large-cased',
    #                                           do_lower_cased=False)
    # dicts = load_dict(root_path='./dicts/kbp17')
    # context_data = PassageData(json_path='./data/kbp17/passage_train.json',
    #                            dicts=dicts,
    #                            tokenizer=tokenizer)
    # dataloader1 = DataLoader(context_data, collate_fn=context_collate)
    # dataloader2 = DataLoader(context_data, collate_fn=context_collate)
    # data_loader = chain.from_iterable([dataloader1, dataloader2])
    # for batch in data_loader:
    #     print(batch)
    # print([len(x['tokens']) for x in context_data])
    # tokenizer = BertTokenizer.from_pretrained('./biobert-large-cased-v1.1-squad',
    #                                           do_lower_cased=False)
    # dicts = load_dict(root_path='./dicts/genia')
    # context_data = ContextData(json_path='./data/genia/context_train.json',
    #                            dicts=dicts,
    #                            tokenizer=tokenizer)
    # sample = context_data[0]
    tokenizer = BertTokenizer.from_pretrained('./bert-large-cased',
                                              do_lower_cased=False)
    dicts = load_dict(root_path='./dicts/genia')
    context_data = ContextData(json_path='./data/genia/context_test.json',
                               dicts=dicts,
                               tokenizer=tokenizer)
    entity_counter = 0
    span_counter = 0
    for sample in context_data:
        entity_counter += len(sample['entities'])
        span_counter += len(sample['span2type'].keys())
    print(entity_counter, span_counter)

    pass
