from os.path import dirname, realpath, join
from os.path import exists
from os import mkdir

import random
import torch
import logging
import re
import numpy as np

"""basic functions"""


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def ctext(src, color='white', bold=False, underline=False):
    """generate text with color
    :param src: text
    :param color: color
    :param bold:
    :param underline:
    :return: colored text
    """
    colors = {'white': '\033[0;m',
              'cyan': '\033[0;36m',
              'grey': '\033[0;37m',
              'red': '\033[0;31m',
              'green': '\033[0;32m',
              'yellow': '\033[0;33m',
              'blue': '\033[0;34m',
              'purple': '\033[0;35m',
              }

    if color not in colors.keys():
        raise ImportError('wrong color')
    elif bold:
        src = '\033[1m' + src
        if underline:
            src = '\033[4m' + src
    return '{}{}{}'.format(colors[color], src, colors['white'])


def cprint(src, color='white', bold=False, underline=False):
    print(ctext(src, color, bold, underline))


def clog(key, value, color, lf='', rt=''):
    key_text = ctext(key, color)
    value_text = ctext('{}'.format(value), 'white')
    text = ctext(lf, 'grey') + key_text + value_text + ctext(rt, 'grey')
    print(text)


def relative_path(path):
    """
    :param path: relative path to current folder
    :return: real path
    """
    folder_real_path = dirname(realpath(__file__))
    return join(folder_real_path, path)


def set_random_seed(seed=9233):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def confirm_directory(directory_path):
    """
    make folder if it doesn't exist
    :param directory_path:
    :return: None
    """
    if not exists(directory_path):
        mkdir(directory_path)
    return None


def check_gpu_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def get_value(tensor):
    if torch.get_device(tensor) == -1:
        return tensor.data.numpy()
    else:
        return tensor.cpu().data.numpy()


"""decoding functions"""


def entity_decode(pred_init_idx, entities_pred_label):
    """trans spans_pred_labels to spans"""
    pred_type_dict = dict()
    for init_idx, entity_pred_label in zip(pred_init_idx, entities_pred_label):
        entity_pred_label = entity_pred_label.tolist()
        counter = 0
        for shift, tag in enumerate(entity_pred_label):
            if tag == 0:
                counter += 1
            elif tag >= 2:
                pred_type_dict['{}-{}'.format(init_idx, init_idx + shift + 1)] = tag
            if counter >= 1:
                break
    return pred_type_dict


def batch_entity_decode(batch_pred_init_idx, batch_entities_pred_label, tolerate=0):
    pred_type_dicts = []
    pred_init_idx_counter = 0
    for idx, pred_init_idx in enumerate(batch_pred_init_idx):
        pred_type_dict = dict()
        for init_idx in pred_init_idx:
            entity_pred_label = batch_entities_pred_label[pred_init_idx_counter].tolist()
            counter = 0
            for shift, tag in enumerate(entity_pred_label):
                if tag == 0:
                    counter += 1
                elif tag >= 2:
                    pred_type_dict['{}-{}'.format(init_idx, init_idx + shift + 1)] = tag
                if counter >= 1 + tolerate:
                    break
            pred_init_idx_counter += 1
        pred_type_dicts.append(pred_type_dict)
    return pred_type_dicts


def batch_entity_decode_backward(batch_pred_end_idx, batch_entities_pred_label):
    pred_type_dicts = []
    pred_end_idx_counter = 0
    for idx, pred_end_idx in enumerate(batch_pred_end_idx):
        pred_type_dict = dict()
        for end_idx in pred_end_idx:
            entity_pred_label = batch_entities_pred_label[pred_end_idx_counter].tolist()
            counter = 0
            for shift, tag in enumerate(entity_pred_label):
                if tag == 0:
                    counter += 1
                elif tag >= 2:
                    pred_type_dict['{}-{}'.format(end_idx - shift, end_idx + 1)] = tag
                if counter >= 1:
                    break
            pred_end_idx_counter += 1
        pred_type_dicts.append(pred_type_dict)
    return pred_type_dicts


def batch_classifier_decode(batch_dict, batch_pred_label):
    batch_pred_type_dict = []
    idx = 0
    for span_dict in batch_dict:
        pred_type_dict = dict()
        for key in span_dict.keys():
            if batch_pred_label[idx] == 7:
                # print('none type')
                idx += 1
                continue
            pred_type_dict[key] = batch_pred_label[idx] + 2
            idx += 1
        batch_pred_type_dict.append(pred_type_dict)
    return batch_pred_type_dict


def batch_prob_decode(batch_pred_idx, batch_entities_pred_probs):
    batch_prob_dicts = []
    counter = 0
    for pred_idx in batch_pred_idx:
        prob_dict = dict()
        for idx in pred_idx:
            prob_dict[idx] = batch_entities_pred_probs[counter]
            counter += 1
        batch_prob_dicts.append(prob_dict)
    return batch_prob_dicts


def batch_prob_decode_backward(batch_pred_idx, batch_entities_pred_probs):
    batch_prob_dicts = []
    counter = 0
    for pred_idx in batch_pred_idx:
        prob_dict = dict()
        for idx in pred_idx:
            prob_dict[idx + 1] = batch_entities_pred_probs[counter]
            counter += 1
        batch_prob_dicts.append(prob_dict)
    return batch_prob_dicts


"""sampling functions"""

"""evaluating functions"""


def merge_pred_spans(forward_pred_spans,
                     backward_pred_spans,
                     forward_pred_span_prob,
                     backward_pred_span_prob,
                     threshold=0.95):
    pred_spans = []
    for f_pred_spans, b_pred_spans, f_pred_span_prob, b_pred_span_prob in zip(
            forward_pred_spans,
            backward_pred_spans,
            forward_pred_span_prob,
            backward_pred_span_prob
    ):
        sample_dict = dict()
        spans = list(set(list(f_pred_spans.keys()) + list(b_pred_spans.keys())))
        spans_score = []
        for span in spans:
            init_idx = int(span.split('-')[0])
            end_idx = int(span.split('-')[1])
            span_score = 0
            if span in f_pred_spans.keys():
                span_score += f_pred_span_prob[init_idx][end_idx - init_idx - 1, 2]
            if span in b_pred_spans.keys():
                span_score += b_pred_span_prob[end_idx][end_idx - init_idx - 1, 2]
            spans_score.append(span_score)
        for span, span_score in zip(spans, spans_score):
            if span_score >= threshold:
                sample_dict[span] = 2
        pred_spans.append(sample_dict)
    return pred_spans


def convert_dict_to_spans(mention_dict):
    """convert mention-dict to span-list"""
    spans = []
    for init_idx in mention_dict.keys():
        for end_idx in mention_dict[init_idx]:
            spans.append([init_idx, end_idx])
    spans.sort(key=lambda x: x[1])
    return spans


def compute_classify_performance(predict_scores, true_label, threshold=0.5):
    predict_label = predict_scores.gt(threshold).int().float()
    true_label = true_label.float()
    tp = get_value(torch.sum(true_label * predict_label).float())
    fp = get_value(torch.sum((1 - true_label) * predict_label).float())
    fn = get_value(torch.sum(true_label * (1 - predict_label)).float())

    if (tp + fn) == 0:
        recall = 0.
    else:
        recall = tp / (tp + fn)

    if (tp + fp) == 0:
        precision = 0.
    else:
        precision = tp / (tp + fp)

    if recall == 0 and precision == 0:
        f1 = 0.
        modified_f1 = 0.
    else:
        f1 = 2 * precision * recall / (precision + recall)
        modified_f1 = 2 * precision * recall / (1.1 * precision + 0.9 * recall)
    return precision * 100, recall * 100, f1 * 100, modified_f1 * 100


def compute_localhypergraph_performance(ground_span2types, predict_span2types, **kwargs):
    if "predict_span2coes" in kwargs.keys():
        predict_span2coes = kwargs['predict_span2coes']
        max_coe = kwargs['max_coe']
        precision = []
        recall = []
        f1 = []
        for i in range(90, int(100 * max_coe + 1)):
            coe = round(i * 0.01, 2)
            tp = 0.
            fp = 0.
            fn = 0.
            for ground_type_dict, pred_type_dict, predict_span2coe in zip(ground_span2types, predict_span2types, predict_span2coes):
                ground_spans = list(ground_type_dict.keys())
                # print(list(pred_type_dict.keys()))
                # print(predict_span2coe)
                pred_spans = list(filter(lambda x: predict_span2coe[x] <= coe, list(pred_type_dict.keys())))
                if len(pred_spans) == 0:
                    fn += len(ground_spans)
                else:
                    for span in ground_spans:
                        if span in pred_spans and ground_type_dict[span] == pred_type_dict[span]:
                            tp += 1
                        else:
                            fn += 1
                    for span in pred_spans:
                        if span not in ground_spans or ground_type_dict[span] != pred_type_dict[span]:
                            fp += 1

            current_precision = tp / (tp + fp) if tp + fp != 0. else 0.
            current_recall = tp / (tp + fn) if tp + fn != 0. else 0.
            current_f1 = 2 * current_precision * current_recall / (
                        current_precision + current_recall) if current_precision + current_recall != 0. else 0.

            current_precision *= 100
            current_recall *= 100
            current_f1 *= 100
            precision.append(current_precision)
            recall.append(current_recall)
            f1.append(current_f1)
    else:
        tp = 0.
        fp = 0.
        fn = 0.
        for ground_type_dict, pred_type_dict in zip(ground_span2types, predict_span2types):
            ground_spans = list(ground_type_dict.keys())
            pred_spans = list(pred_type_dict.keys())
            if len(pred_spans) == 0:
                fn += len(ground_spans)
            else:
                for span in ground_spans:
                    if span in pred_spans and ground_type_dict[span] == pred_type_dict[span]:
                        tp += 1
                    else:
                        fn += 1
                for span in pred_spans:
                    if span not in ground_spans or ground_type_dict[span] != pred_type_dict[span]:
                        fp += 1

        precision = tp / (tp + fp) if tp + fp != 0. else 0.
        recall = tp / (tp + fn) if tp + fn != 0. else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0. else 0.
        precision *= 100
        recall *= 100
        f1 *= 100
    return precision, recall, f1


def merge_dict(dict_list):
    merged_dict = dict()
    for dict_ in dict_list:
        for key in dict_.keys():
            merged_dict[key] = dict_[key]
    return merged_dict


"""log"""


class LocalHyperGraphLogger(object):

    def __init__(self, log_file):
        self.log_file = log_file
        self.counter = 1

    def log(self, key='', value='', color='white', lf='', rt='', toprint=True, bold=False, underline=False):
        key_text = ctext(key, color)
        value_text = ctext('{}'.format(value), 'white', bold, underline)
        text = ctext(lf, 'grey') + key_text + value_text + ctext(rt, 'grey')
        if toprint:
            print(text)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            if not bold:
                f.write('{}{}{}{}\n\n'.format(lf, key, value, rt))
            else:
                f.write('{}**{}{}**{}\n\n'.format(lf, key, value, rt))
        self.counter += 1


if __name__ == '__main__':
    ground_type_dicts = [
        {'1-2': 3,
         '2-4': 4},
        {'1-2': 3,
         '2-3': 4},
    ]
    pred_type_dicts = [
        {'1-2': 3,
         '2-3': 4},
        {'1-2': 0},
    ]

    test_logger = LocalHyperGraphLogger('./test.md')
    test_logger.log(value=[1, 2, 3], color='red')

