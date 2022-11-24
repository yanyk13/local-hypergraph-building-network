import torch
import json
import argparse

from os.path import join
from torch.nn import functional
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import get_cosine_schedule_with_warmup
from transformers import logging
from tensorboardX import SummaryWriter
from itertools import chain
from random import shuffle
from pprint import pprint

from data import ContextData, PassageData, load_dict, context_collate, passage_collate
from model import Config, BoundaryIdentifier, LocalHyperGraphBuilder, BidirectionalLocalHyperGraphBuilder, LocalHyperGraphBuilderSingle, LocalHyperGraphBuilderSingleNone
from model import FocalLoss
from utils import confirm_directory
from utils import check_gpu_device, get_value
from utils import compute_classify_performance, compute_localhypergraph_performance
from utils import ctext, clog, LocalHyperGraphLogger
from utils import set_random_seed


def load_model(benchmark):
    start_config = Config(path='./saved_configs/{}/start.json'.format(benchmark))
    end_config = Config(path='./saved_configs/{}/end.json'.format(benchmark))
    forward_config = Config(path='./saved_configs/{}/forward.json'.format(benchmark))
    backward_config = Config(path='./saved_configs/{}/backward.json'.format(benchmark))

    start_identifier = BoundaryIdentifier(implement_dict=start_config.model)
    end_identifier = BoundaryIdentifier(implement_dict=end_config.model)
    forward_builder = LocalHyperGraphBuilder(implement_dict=forward_config.model)
    backward_builder = LocalHyperGraphBuilder(implement_dict=backward_config.model)

    start_identifier.load('./saved_models/{}/start_identifier.pth'.format(benchmark))
    end_identifier.load('./saved_models/{}/end_identifier.pth'.format(benchmark))
    forward_builder.load('./saved_models/{}/forward_builder.pth'.format(benchmark))
    # backward_builder.load('./saved_models/{}/backward_builder.pth'.format(benchmark))

    # forward_builder.load(
    #     './check_points/ace04/random_seed1/ace04-unidirectional-random_seed1-2022-10-21-10:22:31-best.pth')
    # backward_builder.load(
    #     './check_points/ace04/backward_randomseed1/ace04-unidirectional-backward_randomseed1-2022-10-21-13:58:50-best.pth')

    # forward_builder.load(
    #     './check_points/ace04/random_seed2/ace04-unidirectional-random_seed2-2022-10-21-13:59:21-best.pth')
    # backward_builder.load(
    #     './check_points/ace04/backward_randomseed2/ace04-unidirectional-backward_randomseed2-2022-10-21-13:59:53-best.pth')

    # forward_builder.load(
    #     './check_points/ace04/random_seed5/model.pth')
    backward_builder.load(
        './check_points/ace04/backward_randomseed5/model.pth')
    return start_identifier, end_identifier, forward_builder, backward_builder


def load_model_single(benchmark):
    start_config = Config(path='./saved_configs/{}/start.json'.format(benchmark))
    end_config = Config(path='./saved_configs/{}/end.json'.format(benchmark))
    forward_config = Config(path='./saved_configs/{}/forward.json'.format(benchmark))
    backward_config = Config(path='./saved_configs/{}/backward.json'.format(benchmark))

    start_identifier = BoundaryIdentifier(implement_dict=start_config.model)
    end_identifier = BoundaryIdentifier(implement_dict=end_config.model)
    forward_builder = LocalHyperGraphBuilderSingle(implement_dict=forward_config.model)
    backward_builder = LocalHyperGraphBuilderSingle(implement_dict=backward_config.model)

    start_identifier.load('./saved_models/{}/start_identifier.pth'.format(benchmark))
    end_identifier.load('./saved_models/{}/end_identifier.pth'.format(benchmark))
    forward_builder.load('./saved_models/{}/forward-uniditectional-single.path'.format(benchmark))
    backward_builder.load('./saved_models/{}/backward-unidirectional-single.pth'.format(benchmark))

    return start_identifier, end_identifier, forward_builder, backward_builder


def load_model_single_none(benchmark):
    start_config = Config(path='./saved_configs/{}/start.json'.format(benchmark))
    end_config = Config(path='./saved_configs/{}/end.json'.format(benchmark))
    forward_config = Config(path='./saved_configs/{}/forward.json'.format(benchmark))
    backward_config = Config(path='./saved_configs/{}/backward.json'.format(benchmark))

    start_identifier = BoundaryIdentifier(implement_dict=start_config.model)
    end_identifier = BoundaryIdentifier(implement_dict=end_config.model)
    forward_builder = LocalHyperGraphBuilderSingleNone(implement_dict=forward_config.model)
    backward_builder = LocalHyperGraphBuilderSingleNone(implement_dict=backward_config.model)

    start_identifier.load('./saved_models/{}/start_identifier.pth'.format(benchmark))
    end_identifier.load('./saved_models/{}/end_identifier.pth'.format(benchmark))
    forward_builder.load('./saved_models/{}/forward-uniditectional-single-none.path'.format(benchmark))
    backward_builder.load('./saved_models/{}/backward-unidirectional-single-none.pth'.format(benchmark))

    return start_identifier, end_identifier, forward_builder, backward_builder


def load_uni_model(benchmark):
    start_config = Config(path='./saved_configs/{}/start.json'.format(benchmark))
    forward_config = Config(path='./saved_configs/{}/forward.json'.format(benchmark))

    start_identifier = BoundaryIdentifier(implement_dict=start_config.model)
    forward_builder = LocalHyperGraphBuilder(implement_dict=forward_config.model)

    start_identifier.load('./saved_models/{}/start_identifier.pth'.format(benchmark))
    forward_builder.load('./saved_models/{}/uniencoder2.pth'.format(benchmark))

    return start_identifier, forward_builder


def load_data(benchmark, start_identifier, end_identifier, coe):
    dicts = load_dict(root_path='./dicts/{}'.format(benchmark))
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='./biobert-large-cased-v1.1-squad' if benchmark == 'genia' else './bert-large-cased',
        do_lower_case=False
    )
    if benchmark == 'genia':
        test_data = PassageData('./data/genia/passage_test.json',
                                dicts,
                                tokenizer,
                                name='test_data')
    else:
        test_data = ContextData('./data/{}/context_test.json'.format(benchmark),
                                dicts,
                                tokenizer,
                                name='test_data')

    test_data_info = test_data.gen_start(start_identifier=start_identifier,
                                         max_coe=coe,
                                         train=False)
    for key in test_data_info.keys():
        clog(key='start/{}'.format(key).ljust(30),
             value='{}'.format(test_data_info[key]),
             color='blue')

    test_data_info = test_data.gen_end(end_identifier=end_identifier,
                                       max_coe=coe,
                                       train=False)
    for key in test_data_info.keys():
        clog(key='end/{}'.format(key).ljust(30),
             value='{}'.format(test_data_info[key]),
             color='blue')
    return test_data


def load_uni_data(benchmark, start_identifier, coe):
    dicts = load_dict(root_path='./dicts/{}'.format(benchmark))
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='./biobert-large-cased-v1.1-squad' if benchmark == 'genia' else './bert-large-cased',
        do_lower_case=False
    )
    if benchmark == 'genia':
        test_data = PassageData('./data/genia/passage_test.json',
                                dicts,
                                tokenizer,
                                name='test_data')
    else:
        test_data = ContextData('./data/{}/context_test.json'.format(benchmark),
                                dicts,
                                tokenizer,
                                name='test_data')

    test_data_info = test_data.gen_start(start_identifier=start_identifier,
                                         max_coe=coe,
                                         train=False)
    for key in test_data_info.keys():
        clog(key='end/{}'.format(key).ljust(30),
             value='{}'.format(test_data_info[key]),
             color='blue')
    return test_data


def decode_graph(starts, predicted_labels, predicted_logits):
    counter = 0
    predict_span2types = []
    predict_span2logits = []
    for start in starts:
        sample_span2type = {}
        sample_span2logit = {}
        for s_idx in start:
            predict = predicted_labels[counter].tolist()
            logit = predicted_logits[counter].tolist()
            for shift, (tag, logit) in enumerate(zip(predict, logit)):
                if tag == 0:
                    break
                elif tag > 1:
                    span = "{}-{}".format(s_idx, s_idx + shift + 1)
                    sample_span2type[span] = tag
                    sample_span2logit[span] = logit
                else:
                    continue
            counter += 1
        predict_span2types.append(sample_span2type)
        predict_span2logits.append(sample_span2logit)
    return predict_span2types, predict_span2logits


def decode_graph_back(ends, predicted_labels, predicted_logits):
    counter = 0
    predict_span2types = []
    predict_span2logits = []
    for end in ends:
        sample_span2type = {}
        sample_span2logit = {}
        for e_idx in end:
            predict = predicted_labels[counter].tolist()
            predict_logit = predicted_logits[counter].tolist()
            for shift, (tag, logit) in enumerate(zip(predict, predict_logit)):
                # if tag != 0:
                #     print(tag)
                if tag == 0:
                    break
                elif tag > 1:
                    span = "{}-{}".format(e_idx - shift - 1, e_idx)
                    sample_span2type[span] = tag
                    sample_span2logit[span] = logit
                else:
                    continue
            counter += 1
        predict_span2types.append(sample_span2type)
        predict_span2logits.append(sample_span2logit)
    return predict_span2types, predict_span2logits


def merge_spans(forward_span2types,
                forward_span2logits,
                backward_span2types,
                backward_span2logits,
                threshold=0.5):
    predict_span2types = []
    assert len(forward_span2logits) == len(backward_span2logits)
    for forward_span2type, forward_span2logit, backward_span2type, backward_span2logit in zip(forward_span2types,
                                                                                              forward_span2logits,
                                                                                              backward_span2types,
                                                                                              backward_span2logits):
        sample_span2types = {}
        forward_spans = list(forward_span2type.keys())
        backward_spans = list(backward_span2type.keys())
        spans = list(set(forward_spans + backward_spans))
        if len(spans) == 0:
            predict_span2types.append({})
            continue
        else:
            for span in spans:
                forward_logit = 0 if span not in forward_spans else forward_span2logit[span]
                backward_logit = 0 if span not in backward_spans else backward_span2logit[span]
                mean_score = (forward_logit + backward_logit) / 2.
                if mean_score >= threshold:
                    sample_span2types[span] = forward_span2type[span] if forward_logit > backward_logit \
                        else backward_span2type[span]
        predict_span2types.append(sample_span2types)
    return predict_span2types


def evaluate(model, batches, direction):
    with torch.no_grad():
        model.eval()
        boundaries = []
        boundary_coes = []
        predicted_labels = []
        predicted_logits = []
        ground_span2types = []
        for batch in tqdm(batches, desc=ctext('evaluating', 'grey')):
            ground_span2types.extend(batch['span2type'])
            if direction == 'forward':
                if sum([len(start) for start in batch['start']]) == 0:
                    boundaries.extend([0])
                    boundary_coes.extend([10])
                    predicted_labels.extend([0])
                    predicted_logits.extend([0])
                    continue
            elif direction == 'backward':
                if sum([len(end) for end in batch['end']]) == 0:
                    boundaries.extend([0])
                    boundary_coes.extend([10])
                    predicted_labels.extend([0])
                    predicted_logits.extend([0])
                    continue
            else:
                raise ValueError('illegal direction')
            scores, predicts, targets, masks = model.forward(batch)
            logits = torch.gather(torch.softmax(scores, dim=-1), dim=-1, index=predicts.unsqueeze(-1)).squeeze(-1)
            if direction == 'forward':
                boundaries.extend(batch['start'])
                boundary_coes.extend(batch['start_coe'])
            elif direction == 'backward':
                boundaries.extend(batch['end'])
                boundary_coes.extend(batch['end_coe'])
            else:
                raise ValueError('illegal direction')
            predicted_labels.extend([predict for predict in predicts])
            predicted_logits.extend([logit for logit in logits])

        if direction == 'forward':
            predict_span2types, predict_span2logits = decode_graph(starts=boundaries,
                                                                   predicted_labels=predicted_labels,
                                                                   predicted_logits=predicted_logits)
        elif direction == 'backward':
            predict_span2types, predict_span2logits = decode_graph_back(ends=boundaries,
                                                                        predicted_labels=predicted_labels,
                                                                        predicted_logits=predicted_logits)
        else:
            raise ValueError('illegal direction')
        precision, recall, f1 = compute_localhypergraph_performance(ground_span2types=ground_span2types,
                                                                    predict_span2types=predict_span2types)
        info = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'micro_f1': round(f1, 3)
        }
        return info, ground_span2types, predict_span2types, predict_span2logits


def apply(benchmark, coe=1., gpu=3):
    torch.cuda.set_device(gpu)
    start_identifier, end_identifier, forward_builder, backward_builder = load_model(benchmark)
    test_data = load_data(benchmark, start_identifier, end_identifier, coe)
    batches = DataLoader(test_data,
                         batch_size=32,
                         collate_fn=passage_collate if benchmark == "genia" else context_collate,
                         shuffle=False)
    forward_info, ground_span2types, forward_span2types, forward_spans2logits = evaluate(model=forward_builder,
                                                                                         batches=batches,
                                                                                         direction='forward')
    for key in forward_info.keys():
        clog(key='forward/{}'.format('-'.join(key.split())).ljust(20),
             value='{}'.format(forward_info[key]),
             color='cyan')
    batches = DataLoader(test_data,
                         batch_size=32,
                         collate_fn=passage_collate if benchmark == "genia" else context_collate,
                         shuffle=False)
    backward_info, _, backward_span2types, backward_span2logits = evaluate(model=backward_builder,
                                                                           batches=batches,
                                                                           direction='backward')
    for key in backward_info.keys():
        clog(key='backward/{}'.format('-'.join(key.split())).ljust(20),
             value='{}'.format(backward_info[key]),
             color='blue')
    predict_span2types = merge_spans(forward_span2types=forward_span2types,
                                     forward_span2logits=forward_spans2logits,
                                     backward_span2types=backward_span2types,
                                     backward_span2logits=backward_span2logits,
                                     threshold=0.495)
    precision, recall, f1 = compute_localhypergraph_performance(ground_span2types=ground_span2types,
                                                                predict_span2types=predict_span2types)
    info = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3)
    }
    for key in info.keys():
        clog(key='{}'.format('-'.join(key.split())).ljust(20),
             value='{}'.format(info[key]),
             color='red')

    print('{} & {} & {}'.format(round(precision, 2), round(recall, 2), round(f1, 2)))
    result = []
    for sample, forward_span2type, forward_spans2logit, backward_span2type, backward_span2logit, predict_span2type in zip(test_data,
                                                       forward_span2types,
                                                       forward_spans2logits,
                                                       backward_span2types,
                                                       backward_span2logits,
                                                       predict_span2types):
        sample['forward_span2type'] = forward_span2type
        sample['forward_span2logit'] = forward_spans2logit
        sample['backward_span2type'] = backward_span2type
        sample['backward_span2logit'] = backward_span2logit
        sample['predict_span2type'] = predict_span2type
        sample.pop('start_label')
        sample.pop('end_label')
        result.append(sample)

    # save_path = './saved_results/{}.json'.format(benchmark)
    # with open(save_path, 'w', encoding='utf-8') as f:
    #     json.dump(result, f)


def single_apply(benchmark, coe=1., gpu=3):
    torch.cuda.set_device(gpu)
    start_identifier, end_identifier, forward_builder, backward_builder = load_model_single(benchmark)
    test_data = load_data(benchmark, start_identifier, end_identifier, coe)
    batches = DataLoader(test_data,
                         batch_size=32,
                         collate_fn=passage_collate if benchmark == "genia" else context_collate,
                         shuffle=False)
    forward_info, ground_span2types, forward_span2types, forward_spans2logits = evaluate(model=forward_builder,
                                                                                         batches=batches,
                                                                                         direction='forward')
    for key in forward_info.keys():
        clog(key='forward/{}'.format('-'.join(key.split())).ljust(20),
             value='{}'.format(forward_info[key]),
             color='cyan')
    batches = DataLoader(test_data,
                         batch_size=32,
                         collate_fn=passage_collate if benchmark == "genia" else context_collate,
                         shuffle=False)
    backward_info, _, backward_span2types, backward_span2logits = evaluate(model=backward_builder,
                                                                           batches=batches,
                                                                           direction='backward')
    for key in backward_info.keys():
        clog(key='backward/{}'.format('-'.join(key.split())).ljust(20),
             value='{}'.format(backward_info[key]),
             color='blue')
    predict_span2types = merge_spans(forward_span2types=forward_span2types,
                                     forward_span2logits=forward_spans2logits,
                                     backward_span2types=backward_span2types,
                                     backward_span2logits=backward_span2logits,
                                     threshold=0.495)
    precision, recall, f1 = compute_localhypergraph_performance(ground_span2types=ground_span2types,
                                                                predict_span2types=predict_span2types)
    info = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3)
    }
    for key in info.keys():
        clog(key='{}'.format('-'.join(key.split())).ljust(20),
             value='{}'.format(info[key]),
             color='red')

    print('{} & {} & {}'.format(round(precision, 2), round(recall, 2), round(f1, 2)))
    result = []
    for sample, forward_span2type, forward_spans2logit, backward_span2type, backward_span2logit, predict_span2type in zip(test_data,
                                                       forward_span2types,
                                                       forward_spans2logits,
                                                       backward_span2types,
                                                       backward_span2logits,
                                                       predict_span2types):
        sample['forward_span2type'] = forward_span2type
        sample['forward_span2logit'] = forward_spans2logit
        sample['backward_span2type'] = backward_span2type
        sample['backward_span2logit'] = backward_span2logit
        sample['predict_span2type'] = predict_span2type
        sample.pop('start_label')
        sample.pop('end_label')
        result.append(sample)

    # save_path = './saved_results/{}.json'.format(benchmark)
    # with open(save_path, 'w', encoding='utf-8') as f:
    #     json.dump(result, f)


def single_apply_none(benchmark, coe=1., gpu=3):
    torch.cuda.set_device(gpu)
    start_identifier, end_identifier, forward_builder, backward_builder = load_model_single_none(benchmark)
    test_data = load_data(benchmark, start_identifier, end_identifier, coe)
    batches = DataLoader(test_data,
                         batch_size=32,
                         collate_fn=passage_collate if benchmark == "genia" else context_collate,
                         shuffle=False)
    forward_info, ground_span2types, forward_span2types, forward_spans2logits = evaluate(model=forward_builder,
                                                                                         batches=batches,
                                                                                         direction='forward')
    for key in forward_info.keys():
        clog(key='forward/{}'.format('-'.join(key.split())).ljust(20),
             value='{}'.format(forward_info[key]),
             color='cyan')
    batches = DataLoader(test_data,
                         batch_size=32,
                         collate_fn=passage_collate if benchmark == "genia" else context_collate,
                         shuffle=False)
    backward_info, _, backward_span2types, backward_span2logits = evaluate(model=backward_builder,
                                                                           batches=batches,
                                                                           direction='backward')
    for key in backward_info.keys():
        clog(key='backward/{}'.format('-'.join(key.split())).ljust(20),
             value='{}'.format(backward_info[key]),
             color='blue')
    predict_span2types = merge_spans(forward_span2types=forward_span2types,
                                     forward_span2logits=forward_spans2logits,
                                     backward_span2types=backward_span2types,
                                     backward_span2logits=backward_span2logits,
                                     threshold=0.495)
    precision, recall, f1 = compute_localhypergraph_performance(ground_span2types=ground_span2types,
                                                                predict_span2types=predict_span2types)
    info = {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3)
    }
    for key in info.keys():
        clog(key='{}'.format('-'.join(key.split())).ljust(20),
             value='{}'.format(info[key]),
             color='red')

    print('{} & {} & {}'.format(round(precision, 2), round(recall, 2), round(f1, 2)))

    result = []
    for sample, forward_span2type, forward_spans2logit, backward_span2type, backward_span2logit, predict_span2type in zip(test_data,
                                                       forward_span2types,
                                                       forward_spans2logits,
                                                       backward_span2types,
                                                       backward_span2logits,
                                                       predict_span2types):
        sample['forward_span2type'] = forward_span2type
        sample['forward_span2logit'] = forward_spans2logit
        sample['backward_span2type'] = backward_span2type
        sample['backward_span2logit'] = backward_span2logit
        sample['predict_span2type'] = predict_span2type
        sample.pop('start_label')
        sample.pop('end_label')
        result.append(sample)

    # save_path = './saved_results/{}.json'.format(benchmark)
    # with open(save_path, 'w', encoding='utf-8') as f:
    #     json.dump(result, f)


def uni_apply(benchmark, coe=1., gpu=1):
    torch.cuda.set_device(gpu)
    start_identifier, forward_builder = load_uni_model(benchmark)
    test_data = load_uni_data(benchmark, start_identifier, coe)
    batches = DataLoader(test_data,
                         batch_size=32,
                         collate_fn=passage_collate if benchmark == "genia" else context_collate,
                         shuffle=False)
    forward_info, ground_span2types, forward_span2types, forward_spans2logits = evaluate(model=forward_builder,
                                                                                         batches=batches,
                                                                                         direction='forward')
    for key in forward_info.keys():
        clog(key='forward/{}'.format('-'.join(key.split())).ljust(20),
             value='{}'.format(forward_info[key]),
             color='cyan')
    # batches = DataLoader(test_data,
    #                      batch_size=32,
    #                      collate_fn=passage_collate if benchmark == "genia" else context_collate,
    #                      shuffle=False)
    # backward_info, _, backward_span2types, backward_span2logits = evaluate(model=backward_builder,
    #                                                                        batches=batches,
    #                                                                        direction='backward')
    # for key in backward_info.keys():
    #     clog(key='backward/{}'.format('-'.join(key.split())).ljust(20),
    #          value='{}'.format(backward_info[key]),
    #          color='blue')
    # predict_span2types = merge_spans(forward_span2types=forward_span2types,
    #                                  forward_span2logits=forward_spans2logits,
    #                                  backward_span2types=backward_span2types,
    #                                  backward_span2logits=backward_span2logits,
    #                                  threshold=0.495)
    # precision, recall, f1 = compute_localhypergraph_performance(ground_span2types=ground_span2types,
    #                                                             predict_span2types=predict_span2types)
    # info = {
    #     "precision": round(precision, 3),
    #     "recall": round(recall, 3),
    #     "f1": round(f1, 3)
    # }
    # for key in info.keys():
    #     clog(key='{}'.format('-'.join(key.split())).ljust(20),
    #          value='{}'.format(info[key]),
    #          color='red')
    #
    # result = []
    # for sample, forward_span2type, forward_spans2logit, backward_span2type, backward_span2logit, predict_span2type in zip(test_data,
    #                                                    forward_span2types,
    #                                                    forward_spans2logits,
    #                                                    backward_span2types,
    #                                                    backward_span2logits,
    #                                                    predict_span2types):
    #     sample['forward_span2type'] = forward_span2type
    #     sample['forward_span2logit'] = forward_spans2logit
    #     sample['backward_span2type'] = backward_span2type
    #     sample['backward_span2logit'] = backward_span2logit
    #     sample['predict_span2type'] = predict_span2type
    #     sample.pop('start_label')
    #     sample.pop('end_label')
    #     result.append(sample)
    #
    # save_path = './saved_results/{}.json'.format(benchmark)
    # with open(save_path, 'w', encoding='utf-8') as f:
    #     json.dump(result, f)


if __name__ == '__main__':
    logging.set_verbosity_error()
    set_random_seed()
    benchmarks = ['genia', 'ace04', 'ace05', 'kbp17']
    coes = [1., 1., 1., 1.]
    # for coe in [1., 1.01, 1.05, 1.1]:
    #     clog(key='coe ',
    #          value=coe,
    #          color='yellow')
    for benchmark, coe in zip(benchmarks[1:2], coes[1:2]):
        clog(key='benchmark ',
             value=benchmark,
             color='red')
        apply(benchmark, coe)
        # single_apply(benchmark, coe)
        # single_apply_none(benchmark, coe)
        torch.cuda.empty_cache()
