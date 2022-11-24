import torch
import argparse

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

from data import ContextData, PassageData, load_dict, context_collate, passage_collate
from model import Config, BoundaryIdentifier, LocalHyperGraphBuilder, BidirectionalLocalHyperGraphBuilder
from model import LocalHyperGraphBuilderSingle, LocalHyperGraphBuilderSingleNone
from model import FocalLoss
from utils import confirm_directory
from utils import check_gpu_device, get_value
from utils import compute_classify_performance, compute_localhypergraph_performance
from utils import ctext, LocalHyperGraphLogger
from utils import set_random_seed


def load_data(config, test_summary, logger, get_start=False, get_end=False):
    dicts = load_dict(root_path=config.dict_directory)
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.model['encoder']['bert'],
        do_lower_case=False
    )
    if config.data == 'genia':
        train_data = [PassageData(config.train_data, dicts, tokenizer,
                                  name='train_data', test=config.test,
                                  min_len=min_len, max_len=max_len)
                      for min_len, max_len in zip(config.split_length[:-1], config.split_length[1:])]
        test_data = PassageData(config.test_data, dicts, tokenizer, name='test_data', test=config.test)
    else:
        train_data = [ContextData(config.train_data, dicts, tokenizer,
                                  name='train_data', test=config.test,
                                  min_len=min_len, max_len=max_len)
                      for min_len, max_len in zip(config.split_length[:-1], config.split_length[1:])]
        test_data = ContextData(config.test_data, dicts, tokenizer, name='test_data', test=config.test)

    lines = ''
    if get_start:
        start_identifier_config = Config(path=config.start_identifier['config_path'])
        start_identifier = BoundaryIdentifier(start_identifier_config.model, mode=start_identifier_config.mode)
        start_identifier.load(config.start_identifier['model_path'])
        for split_data, min_len, max_len in zip(train_data, config.split_length[:-1], config.split_length[1:]):
            train_data_info = split_data.gen_start(start_identifier=start_identifier,
                                                   max_coe=config.start_identifier['train_coe'],
                                                   train=False if config.model['mode'] == "attention" else True,
                                                   extend=config.start_identifier['extend'])
            lines += "|key|value|\n|---|-----|\n"
            for key in list(train_data_info.keys())[-3:]:
                logger.log(key='train_data-{}-{}/start/{}'.format(min_len, max_len, key).ljust(50),
                           value='{}'.format(train_data_info[key]),
                           color='blue')
                lines += '|train-{}|{}|\n'.format(key, train_data_info[key])
        test_data_info = test_data.gen_start(start_identifier=start_identifier,
                                             max_coe=config.start_identifier['test_coe'],
                                             train=False)
        print_counter = 0
        for key in test_data_info.keys():
            if print_counter % 3 == 0:
                logger.log(key='=',
                           value='===========')
                lines += '|~|~|\n'
            logger.log(key='test_data/start/{}'.format(key).ljust(30),
                       value='{}'.format(test_data_info[key]),
                       color='blue')
            lines += '|test-{}|{}|\n'.format(key, test_data_info[key])
            print_counter += 1
        test_summary.add_text(tag='start information',
                              text_string=lines,
                              global_step=1)
    if get_end:
        end_identifier_config = Config(path=config.end_identifier['config_path'])
        end_identifier = BoundaryIdentifier(end_identifier_config.model, mode=end_identifier_config.mode)
        end_identifier.load(config.end_identifier['model_path'])
        for split_data, min_len, max_len in zip(train_data, config.split_length[:-1], config.split_length[1:]):
            train_data_info = split_data.gen_end(end_identifier=end_identifier,
                                                 max_coe=config.end_identifier['train_coe'],
                                                 train=False if config.model['mode'] == "attention" else True,
                                                 extend=config.end_identifier['extend'])
            # sample = split_data[0]
            # [start_idx, end_idx] = sample['sentence_boundary']
            # content = sample['tokens'][start_idx: end_idx]
            # print(' '.join(content))
            # for boundary, label in zip(sample['sub_sequence_boundary_back'], sample['sub_sequence_label_back']):
            #     print('boundary', boundary)
            #     print('label', label)
            #     print(' '.join(content[boundary[0]: boundary[1]]))
            #
            # for entity in sample['entities']:
            #     print(entity)
            #     s_idx = entity['start']
            #     e_idx = entity['end']
            #     print('entity', ' '.join(content[s_idx: e_idx]))
            # break
            lines += "|key|value|\n|---|-----|\n"
            for key in list(train_data_info.keys())[-3:]:
                logger.log(key='train_data-{}-{}/end/{}'.format(min_len, max_len, key).ljust(50),
                           value='{}'.format(train_data_info[key]),
                           color='blue')
                lines += '|train-{}|{}|\n'.format(key, train_data_info[key])
        test_data_info = test_data.gen_end(end_identifier=end_identifier,
                                           max_coe=config.end_identifier['test_coe'],
                                           train=False)
        print_counter = 0
        for key in test_data_info.keys():
            if print_counter % 3 == 0:
                logger.log(key='=',
                           value='===========')
                lines += '|~|~|\n'
            logger.log(key='test_data/end/{}'.format(key).ljust(30),
                       value='{}'.format(test_data_info[key]),
                       color='blue')
            lines += '|test-{}|{}|\n'.format(key, test_data_info[key])
            print_counter += 1
        test_summary.add_text(tag='end information',
                              text_string=lines,
                              global_step=1)
    data = {"train": train_data,
            "test": test_data}
    return data


def start_train(model, batches, **kwargs):
    config = kwargs['config']
    optimizer = kwargs['optimizer']
    model.train()
    epoch_loss = 0
    counter = 0
    truth_labels = []
    predict_scores = []
    for batch in tqdm(batches, desc=ctext('training', 'grey')):
        model.zero_grad()
        scores = model.forward(batch)
        targets = check_gpu_device(batch['start_label'])
        masks = check_gpu_device(batch['content_mask'])
        if config.focal:
            focal_loss = FocalLoss(
                num_classes=config.model['fc_layer']['dim_out'],
                alpha=config.alpha
            )
            loss = focal_loss.forward(inputs=scores.view(-1, config.model['fc_layer']['dim_out']),
                                      targets=targets.view(-1),
                                      reduction='none')
        else:
            loss = functional.binary_cross_entropy_with_logits(input=scores.view(-1),
                                                               target=targets.view(-1),
                                                               reduction='none')
        predict_scores.append(scores.view(-1) * masks.view(-1))
        truth_labels.append(targets.view(-1) * masks.view(-1))
        loss = torch.sum(loss * masks.view(-1)) / torch.sum(masks.view(-1))
        loss.backward()
        optimizer.step()
        loss.detach()
        epoch_loss += get_value(loss)
        counter += 1
    epoch_loss = epoch_loss/counter
    # precision, recall, f1, modified_f1 = compute_classify_performance(predict_scores=torch.cat(predict_scores),
    #                                                      true_label=torch.cat(truth_labels))
    # info = {
    #     'loss': round(epoch_loss, 5),
    #     'precision': round(precision, 3),
    #     'recall': round(recall, 3),
    #     'micro_f1': round(f1, 3),
    #     'modified_f1': round(modified_f1, 3)
    # }

    precision, recall, f1, _ = compute_classify_performance(predict_scores=torch.cat(predict_scores),
                                                            true_label=torch.cat(truth_labels))
    info = {
        'loss': round(epoch_loss, 5),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'micro_f1': round(f1, 3)
    }
    return info


def start_evaluate(model, batches, **kwargs):
    model.eval()
    with torch.no_grad():
        truth_labels = []
        predict_scores = []
        for batch in tqdm(batches, desc=ctext('evaluating', 'grey')):
            model.zero_grad()
            scores = model.forward(batch)
            targets = check_gpu_device(batch['start_label'])
            masks = check_gpu_device(batch['content_mask'])
            predict_scores.append(scores.view(-1) * masks.view(-1))  # truth negative have no effect on results
            truth_labels.append(targets.view(-1) * masks.view(-1))
        # precision, recall, f1, modified_f1 = compute_classify_performance(predict_scores=torch.cat(predict_scores),
        #                                                      true_label=torch.cat(truth_labels))
        # info = {
        #     'precision': round(precision, 3),
        #     'recall': round(recall, 3),
        #     'micro_f1': round(f1, 3),
        #     'modified_f1': round(modified_f1, 3)
        # }
        precision, recall, f1, modified_f1 = compute_classify_performance(predict_scores=torch.cat(predict_scores),
                                                                          true_label=torch.cat(truth_labels))
        info = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'micro_f1': round(f1, 3)
        }
        return info
    
    
def end_train(model, batches, **kwargs):
    config = kwargs['config']
    optimizer = kwargs['optimizer']
    model.train()
    epoch_loss = 0
    counter = 0
    truth_labels = []
    predict_scores = []
    for batch in tqdm(batches, desc=ctext('training', 'grey')):
        model.zero_grad()
        scores = model.forward(batch)
        targets = check_gpu_device(batch['end_label'])
        masks = check_gpu_device(batch['content_mask'])
        if config.focal:
            focal_loss = FocalLoss(
                num_classes=config.model['fc_layer']['dim_out'],
                alpha=config.alpha
            )
            loss = focal_loss.forward(inputs=scores.view(-1, config.model['fc_layer']['dim_out']),
                                      targets=targets.view(-1),
                                      reduction='none')
        else:
            loss = functional.binary_cross_entropy_with_logits(input=scores.view(-1),
                                                               target=targets.view(-1),
                                                               reduction='none')
        predict_scores.append(scores.view(-1) * masks.view(-1))
        truth_labels.append(targets.view(-1) * masks.view(-1))
        loss = torch.sum(loss * masks.view(-1)) / torch.sum(masks.view(-1))
        loss.backward()
        optimizer.step()
        loss.detach()
        epoch_loss += get_value(loss)
        counter += 1
    epoch_loss = epoch_loss/counter
    # precision, recall, f1, modified_f1 = compute_classify_performance(predict_scores=torch.cat(predict_scores),
    #                                                      true_label=torch.cat(truth_labels))
    # info = {
    #     'loss': round(epoch_loss, 5),
    #     'precision': round(precision, 3),
    #     'recall': round(recall, 3),
    #     'micro_f1': round(f1, 3),
    #     'modified_f1': round(modified_f1, 3)
    # }
    precision, recall, f1, _ = compute_classify_performance(predict_scores=torch.cat(predict_scores),
                                                                      true_label=torch.cat(truth_labels))
    info = {
        'loss': round(epoch_loss, 5),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'micro_f1': round(f1, 3)
    }
    return info


def end_evaluate(model, batches, **kwargs):
    model.eval()
    with torch.no_grad():
        truth_labels = []
        predict_scores = []
        for batch in tqdm(batches, desc=ctext('evaluating', 'grey')):
            model.zero_grad()
            scores = model.forward(batch)
            targets = check_gpu_device(batch['end_label'])
            masks = check_gpu_device(batch['content_mask'])
            predict_scores.append(scores.view(-1) * masks.view(-1))  # truth negative have no effect on results
            truth_labels.append(targets.view(-1) * masks.view(-1))
    # precision, recall, f1, modified_f1 = compute_classify_performance(predict_scores=torch.cat(predict_scores),
    #                                                      true_label=torch.cat(truth_labels))
    # info = {
    #     'precision': round(precision, 3),
    #     'recall': round(recall, 3),
    #     'micro_f1': round(f1, 3),
    #     'modified_f1': round(modified_f1, 3)
    # }
    precision, recall, f1, _ = compute_classify_performance(predict_scores=torch.cat(predict_scores),
                                                                      true_label=torch.cat(truth_labels))
    info = {
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'micro_f1': round(f1, 3)
    }
    return info


def decode_graph(starts, predicted_labels, **kwargs):
    if "start_coes" in kwargs.keys():
        start_coes = kwargs['start_coes']
        counter = 0
        predict_span2types = []
        predict_span2coes = []
        for start, coe in zip(starts, start_coes):
            sample_span2type = {}
            sample_span2coe = {}
            for s_idx, c in zip(start, coe):
                predict = predicted_labels[counter].tolist()
                for shift, tag in enumerate(predict):
                    if tag == 0:
                        break
                    elif tag > 1:
                        span = "{}-{}".format(s_idx, s_idx + shift + 1)
                        sample_span2type[span] = tag
                        sample_span2coe[span] = c
                    else:
                        continue
                counter += 1
            predict_span2types.append(sample_span2type)
            predict_span2coes.append(sample_span2coe)
        return predict_span2types, predict_span2coes
    else:
        counter = 0
        predict_span2types = []
        for start in starts:
            sample_span2type = {}
            for s_idx in start:
                predict = predicted_labels[counter].tolist()
                for shift, tag in enumerate(predict):
                    if tag == 0:
                        break
                    elif tag > 1:
                        span = "{}-{}".format(s_idx, s_idx + shift + 1)
                        sample_span2type[span] = tag
                    else:
                        continue
                counter += 1
            predict_span2types.append(sample_span2type)
        return predict_span2types


def decode_graph_back(ends, predicted_labels, **kwargs):
    if "end_coes" in kwargs.keys():
        end_coes = kwargs['end_coes']
        counter = 0
        predict_span2types = []
        predict_span2coes = []
        for end, coe in zip(ends, end_coes):
            sample_span2type = {}
            sample_span2coe = {}
            for e_idx, c in zip(end, coe):
                predict = predicted_labels[counter].tolist()
                for shift, tag in enumerate(predict):
                    if tag == 0:
                        break
                    elif tag > 1:
                        span = "{}-{}".format(e_idx - shift - 1, e_idx)
                        sample_span2type[span] = tag
                        sample_span2coe[span] = c
                    else:
                        continue
                counter += 1
            predict_span2types.append(sample_span2type)
            predict_span2coes.append(sample_span2coe)
        return predict_span2types, predict_span2coes
    else:
        counter = 0
        predict_span2types = []
        # print(len(ends))
        # print(len(predicted_labels))
        for end in ends:
            sample_span2type = {}
            for e_idx in end:
                predict = predicted_labels[counter].tolist()
                for shift, tag in enumerate(predict):
                    # if tag != 0:
                    #     print(tag)
                    if tag == 0:
                        break
                    elif tag > 1:
                        span = "{}-{}".format(e_idx - shift - 1, e_idx)
                        sample_span2type[span] = tag
                    else:
                        continue
                counter += 1
            predict_span2types.append(sample_span2type)
        return predict_span2types


def unidirectional_train(model, batches, **kwargs):
    config = kwargs['config']
    optimizer = kwargs['optimizer']
    model.train()
    epoch_loss = 0
    counter = 0
    boundaries = []
    predicted_labels = []
    ground_span2types = []

    for batch in tqdm(batches, desc=ctext('training', 'grey')):
        ground_span2types.extend(batch['span2type'])
        if config.model['direction'] == 'forward':
            if sum([len(start) for start in batch['start']]) == 0:
                boundaries.extend([[0]])
                predicted_labels.extend(([torch.tensor([0])]))
                continue
        elif config.model['direction'] == 'backward':
            if sum([len(end) for end in batch['end']]) == 0:
                boundaries.extend([[0]])
                predicted_labels.extend(([torch.tensor([0])]))
                continue
        else:
            raise ValueError('illegal direction')
        model.zero_grad()
        scores, predicts, targets, masks = model.forward(batch)
        if config.focal:
            focal_loss = FocalLoss(
                num_classes=config.model['fc_layer']['dim_out'],
                alpha=config.alpha
            )
            loss = focal_loss.forward(inputs=scores.view(-1, config.model['fc_layer']['dim_out']),
                                      targets=targets.view(-1),
                                      reduction='none')
        else:
            loss = functional.cross_entropy(input=scores.view(-1, config.model['fc_layer']['dim_out']),
                                            target=targets.view(-1),
                                            reduction='none')
        if config.model['direction'] == 'forward':
            boundaries.extend(batch['start'])
        elif config.model['direction'] == 'backward':
            boundaries.extend(batch['end'])
        else:
            raise ValueError('illegal direction')
        predicted_labels.extend([predict for predict in predicts])
        loss = torch.sum(loss * masks.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += get_value(loss)
        counter += get_value(torch.sum(masks))

    if counter != 0:
        epoch_loss = epoch_loss / counter
        if config.model['direction'] == 'forward':
            predict_span2types = decode_graph(starts=boundaries, predicted_labels=predicted_labels)
        elif config.model['direction'] == 'backward':
            predict_span2types = decode_graph_back(ends=boundaries, predicted_labels=predicted_labels)
        else:
            raise ValueError('illegal direction')
        # print(ground_span2types)
        # print(predict_span2types)

        precision, recall, f1 = compute_localhypergraph_performance(ground_span2types=ground_span2types,
                                                                    predict_span2types=predict_span2types)
    else:
        epoch_loss = 0.
        precision = 0.
        recall = 0.
        f1 = 0.
    info = {
        'loss': round(epoch_loss, 5),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'micro_f1': round(f1, 3)
    }
    return info


def unidirectional_test(model, batches, **kwargs):
    with torch.no_grad():
        config = kwargs['config']
        model.eval()
        boundaries = []
        boundary_coes = []
        predicted_labels = []
        ground_span2types = []
        counter = 0
        for batch in tqdm(batches, desc=ctext('evaluating', 'grey')):
            ground_span2types.extend(batch['span2type'])
            if config.model['direction'] == 'forward':
                if sum([len(start) for start in batch['start']]) == 0:
                    boundaries.extend([0])
                    boundary_coes.extend([10])
                    predicted_labels.extend([0])
                    continue
            elif config.model['direction'] == 'backward':
                if sum([len(end) for end in batch['end']]) == 0:
                    boundaries.extend([0])
                    boundary_coes.extend([10])
                    predicted_labels.extend([0])
                    continue
            else:
                raise ValueError('illegal direction')
            scores, predicts, targets, masks = model.forward(batch)
            if config.model['direction'] == 'forward':
                boundaries.extend(batch['start'])
                boundary_coes.extend(batch['start_coe'])
            elif config.model['direction'] == 'backward':
                boundaries.extend(batch['end'])
                boundary_coes.extend(batch['end_coe'])
            else:
                raise ValueError('illegal direction')

            predicted_labels.extend([predict for predict in predicts])
            counter += 1

        if counter != 0:
            if config.model['direction'] == 'forward':
                max_coe = config.start_identifier['test_coe']
            elif config.model['direction'] == 'backward':
                max_coe = config.end_identifier['test_coe']
            else:
                raise ValueError('illegal direction')
            coes = [round(i * 0.01, 2) for i in range(90, int(100 * max_coe + 1))]
            if config.model['direction'] == 'forward':
                predict_span2types, predict_span2coes = decode_graph(starts=boundaries, start_coes=boundary_coes, predicted_labels=predicted_labels)
            elif config.model['direction'] == 'backward':
                predict_span2types, predict_span2coes = decode_graph_back(ends=boundaries, end_coes=boundary_coes, predicted_labels=predicted_labels)
            else:
                raise ValueError('illegal direction')
            precision, recall, f1 = compute_localhypergraph_performance(ground_span2types=ground_span2types,
                                                                        predict_span2types=predict_span2types,
                                                                        predict_span2coes=predict_span2coes,
                                                                        max_coe=max_coe)
            best_f1 = max(f1)
            best_index = f1.index(best_f1)
            best_coe = coes[best_index]
            best_precision = precision[best_index]
            best_recall = recall[best_index]
        else:
            best_coe = 0.
            best_precision = 0.
            best_recall = 0.
            best_f1 = 0.
        info = {
            'coe': best_coe,
            'precision': round(best_precision, 3),
            'recall': round(best_recall, 3),
            'micro_f1': round(best_f1, 3)
        }
        return info


def bidirectional_train(model, batches, **kwargs):
    config = kwargs['config']
    optimizer = kwargs['optimizer']
    model.train()
    epoch_loss = 0
    counter = 0
    boundaries = []
    predicted_labels = []
    boundaries_back = []
    predicted_labels_back = []
    ground_span2types = []

    for batch in tqdm(batches, desc=ctext('training', 'grey')):
        ground_span2types.extend(batch['span2type'])
        if sum([len(start) for start in batch['start']]) == 0 or sum([len(start) for start in batch['end']]) == 0:
            boundaries.extend([0])
            predicted_labels.extend([torch.tensor([0])])
            boundaries_back.extend([0])
            predicted_labels_back.extend([torch.tensor([0])])
            continue
        model.zero_grad()
        scores, predicts, targets, masks, scores_back, predicts_back, targets_back, masks_back = model.forward(
            batch=batch)
        if config.focal:
            focal_loss = FocalLoss(
                num_classes=config.model['fc_layer']['dim_out'],
                alpha=config.alpha
            )
            loss = focal_loss.forward(inputs=scores.view(-1, config.model['fc_layer']['dim_out']),
                                      targets=targets.view(-1),
                                      reduction='none')
            loss_back = focal_loss.forward(inputs=scores_back.view(-1, config.model['fc_layer']['dim_out']),
                                           targets=targets_back.view(-1),
                                           reduction='none')
        else:
            loss = functional.cross_entropy(input=scores.view(-1, config.model['fc_layer']['dim_out']),
                                            target=targets.view(-1),
                                            reduction='none')
            loss_back = functional.cross_entropy(input=scores_back.view(-1, config.model['fc_layer']['dim_out']),
                                                 target=targets_back.view(-1),
                                                 reduction='none')

        boundaries.extend(batch['start'])
        boundaries_back.extend(batch['end'])
        predicted_labels.extend([predict for predict in predicts])
        predicted_labels_back.extend([predict_back for predict_back in predicts_back])

        loss = torch.sum(loss * masks.view(-1))
        loss_back = torch.sum(loss_back * masks_back.view(-1))
        total_loss = loss + loss_back
        total_loss.backward()
        optimizer.step()
        epoch_loss += get_value(total_loss)
        counter += get_value(torch.sum(masks)) + get_value(torch.sum(masks_back))

    if counter != 0:
        epoch_loss = epoch_loss / counter
        predict_span2types = decode_graph(starts=boundaries, predicted_labels=predicted_labels)
        predict_span2types_back = decode_graph_back(ends=boundaries_back, predicted_labels=predicted_labels_back)
        precision, recall, f1 = compute_localhypergraph_performance(ground_span2types=ground_span2types,
                                                                    predict_span2types=predict_span2types)
        precision_back, recall_back, f1_back = compute_localhypergraph_performance(ground_span2types=ground_span2types,
                                                                                   predict_span2types=predict_span2types_back)
    else:
        epoch_loss = 0.
        precision = 0.
        recall = 0.
        f1 = 0.
        precision_back = 0.
        recall_back = 0.
        f1_back = 0.
    info = {
        'loss': round(epoch_loss, 5),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'micro_f1': round(f1, 3),
        'precision_back': round(precision_back, 3),
        'recall_back': round(recall_back, 3),
        'micro_f1_back': round(f1_back, 3),
    }
    return info


def bidirectional_test(model, batches, **kwargs):
    with torch.no_grad():
        config = kwargs['config']
        model.eval()
        counter = 0
        boundaries = []
        boundary_coes = []
        predicted_labels = []
        boundaries_back = []
        boundary_coes_back = []
        predicted_labels_back = []
        ground_span2types = []

        for batch in tqdm(batches, desc=ctext('training', 'grey')):
            ground_span2types.extend(batch['span2type'])
            if sum([len(start) for start in batch['start']]) == 0 or sum([len(start) for start in batch['end']]) == 0:
                boundaries.extend([0])
                boundary_coes.extend([10])
                predicted_labels.extend([torch.tensor([0])])
                boundaries_back.extend([0])
                boundary_coes_back.extend([10])
                predicted_labels_back.extend([torch.tensor([0])])
                continue
            scores, predicts, targets, masks, scores_back, predicts_back, targets_back, masks_back = model.forward(
                batch=batch)
            boundaries.extend(batch['start'])
            boundary_coes.extend(batch['start_coe'])
            boundaries_back.extend(batch['end'])
            boundary_coes_back.extend(batch['end_coe'])
            predicted_labels.extend([predict for predict in predicts])
            predicted_labels_back.extend([predict_back for predict_back in predicts_back])
            counter += get_value(torch.sum(masks)) + get_value(torch.sum(masks_back))
        if counter != 0:
            max_coe = config.start_identifier['test_coe']
            max_coe_back = config.end_identifier['test_coe']
            coes = [round(i * 0.01, 2) for i in range(90, int(100 * max_coe + 1))]
            coes_back = [round(i * 0.01, 2) for i in range(90, int(100 * max_coe_back + 1))]
            predict_span2types, predict_span2coes = decode_graph(starts=boundaries,
                                                                 start_coes=boundary_coes,
                                                                 predicted_labels=predicted_labels)
            predict_span2types_back, predict_span2coes_back = decode_graph_back(ends=boundaries_back,
                                                                                end_coes=boundary_coes_back,
                                                                                predicted_labels=predicted_labels_back)
            precision, recall, f1 = compute_localhypergraph_performance(
                ground_span2types=ground_span2types,
                predict_span2types=predict_span2types,
                predict_span2coes=predict_span2coes,
                max_coe=max_coe)
            precision_back, recall_back, f1_back = compute_localhypergraph_performance(
                ground_span2types=ground_span2types,
                predict_span2types=predict_span2types_back,
                predict_span2coes=predict_span2coes_back,
                max_coe=max_coe)
            best_f1 = max(f1)
            best_index = f1.index(best_f1)
            best_coe = coes[best_index]
            best_precision = precision[best_index]
            best_recall = recall[best_index]
            best_f1_back = max(f1_back)
            best_index_back = f1_back.index(best_f1_back)
            best_coe_back = coes_back[best_index_back]
            best_precision_back = precision_back[best_index_back]
            best_recall_back = recall_back[best_index_back]
        else:
            best_coe = 0.
            best_precision = 0.
            best_recall = 0.
            best_f1 = 0.
            best_coe_back = 0.
            best_precision_back = 0.
            best_recall_back = 0.
            best_f1_back = 0.
        info = {
            'coe': best_coe,
            'precision': round(best_precision, 3),
            'recall': round(best_recall, 3),
            'micro_f1': round(best_f1, 3),
            'coe_back': best_coe_back,
            'precision_back': round(best_precision_back, 3),
            'recall_back': round(best_recall_back, 3),
            'micro_f1_back': round(best_f1_back, 3)
        }
        return info


def process(config):
    config.generate_tag()
    config.info()
    confirm_directory('./text_logs/{}'.format(config.data))
    confirm_directory('./text_logs/{}/{}'.format(config.data, config.mode))
    logger = LocalHyperGraphLogger(log_file='./text_logs/{}/{}/{}-{}.md'.format(config.data,
                                                                                config.mode,
                                                                                config.name,
                                                                                config.time))
    logger.log(value=config.mark_down(), toprint=False)
    train_summary = SummaryWriter(logdir="./logs/{}/{}/{}-{}/train".format(config.data,
                                                                           config.mode,
                                                                           config.name,
                                                                           config.time),
                                  flush_secs=30)
    test_summary = SummaryWriter(logdir="./logs/{}/{}/{}-{}/test".format(config.data,
                                                                         config.mode,
                                                                         config.name,
                                                                         config.time),
                                 flush_secs=30)
    test_summary.add_text(tag='configuration',
                          text_string=config.mark_down(),
                          global_step=1)

    function_option = {
        "start_train": start_train,
        "start_test": start_evaluate,
        "end_train": end_train,
        "end_test": end_evaluate,
        "unidirectional_train": unidirectional_train,
        "unidirectional_test": unidirectional_test,
        "unidirectional_single_train": unidirectional_train,
        "unidirectional_single_test": unidirectional_test,
        "unidirectional_single_none_train": unidirectional_train,
        "unidirectional_single_none_test": unidirectional_test,
        "bidirectional_train": bidirectional_train,
        "bidirectional_test": bidirectional_test
    }
    colors = {
        "train": 'yellow',
        "devel": 'cyan',
        "test": 'red'
    }
    summaries = {
        "train": train_summary,
        "test": test_summary
    }
    model_option = {
        "start": BoundaryIdentifier,
        "end": BoundaryIdentifier,
        "unidirectional": LocalHyperGraphBuilder,
        "bidirectional": BidirectionalLocalHyperGraphBuilder,
        "unidirectional_single": LocalHyperGraphBuilderSingle,
        "unidirectional_single_none": LocalHyperGraphBuilderSingleNone
    }
    torch.cuda.set_device(config.gpu)
    data = load_data(config, test_summary=test_summary, logger=logger,
                     get_start=True if 'start_identifier' in dir(config) else False,
                     get_end=True if 'end_identifier' in dir(config) else False)
    model = model_option[config.mode](config.model,
                                      name=config.name,
                                      tag=config.tag,
                                      mode=config.mode)
    if config.fine:
        model.load(config.previous_model)
    if config.mode in ['start', 'end']:
        bert_params_ids = list(map(id, model.encoder.bert.parameters()))
        other_params = filter(lambda p: id(p) not in bert_params_ids,
                              model.parameters())
        optimizer = AdamW([
            {"params": model.encoder.bert.parameters(),
             "initial_lr": config.bert_learning_rate},
            {"params": other_params,
             "initial_lr": config.learning_rate}
        ])
    elif "unidirectional" in config.mode:
        bert_params_ids = list(map(id, chain(model.boundary_encoder.bert.parameters(),
                                             model.content_encoder.bert.parameters())))
        other_params = filter(lambda p: id(p) not in bert_params_ids,
                              model.parameters())
        optimizer = AdamW([
            {"params": chain(model.boundary_encoder.bert.parameters(),
                             model.content_encoder.bert.parameters()),
             "initial_lr": config.bert_learning_rate},
            {"params": other_params,
             "initial_lr": config.learning_rate}
        ])
    elif "bidirectional" in config.mode:
        bert_params_ids = list(map(id, chain(model.forward_encoder.bert.parameters(),
                                             model.backward_encoder.bert.parameters())))
        other_params = filter(lambda p: id(p) not in bert_params_ids,
                              model.parameters())
        optimizer = AdamW([
            {"params": chain(model.forward_encoder.bert.parameters(),
                             model.backward_encoder.bert.parameters()),
             "initial_lr": config.bert_learning_rate},
            {"params": other_params,
             "initial_lr": config.learning_rate}
        ])
    else:
        raise ValueError('unregistered name')

    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                num_cycles=0.5,
                                                num_warmup_steps=config.warmup_epoch,
                                                num_training_steps=config.end_epoch)
    previous_best = config.saving_threshold
    for epoch_idx in range(config.start_epoch, config.end_epoch + 1):
        logger.log(key='', value='>epoch {}'.format(epoch_idx), lf='', color='grey')
        scheduler.step()
        lr_dict = {
            "bert_learning_rate": optimizer.state_dict()['param_groups'][0]['lr'],
            "learning_rate": optimizer.state_dict()['param_groups'][1]['lr']
        }
        for key in lr_dict.keys():
            test_summary.add_scalar(tag='learning_rate/{}'.format(key), scalar_value=lr_dict[key],
                                    global_step=epoch_idx)

        for stage in config.stages:
            if stage == 'train':
                split_data_len = list(zip(data['train'], config.train_batch_size))
                shuffle(split_data_len)
                batch_generator = chain.from_iterable([DataLoader(split_data,
                                                                  batch_size=batch_size,
                                                                  collate_fn=passage_collate if config.data == "genia" else context_collate,
                                                                  shuffle=True)
                                                       for split_data, batch_size in split_data_len])
            else:
                batch_generator = DataLoader(data['test'],
                                             batch_size=config.evaluate_batch_size,
                                             collate_fn=passage_collate if config.data == "genia" else context_collate,
                                             shuffle=True)
            info = function_option["{}_{}".format(config.mode, stage)](model=model, batches=batch_generator, config=config, optimizer=optimizer)
            for key in info.keys():
                summaries[stage].add_scalar(tag='{}/{}'.format(config.task, key),
                                            scalar_value=info[key],
                                            global_step=epoch_idx)
                logger.log(key='{}/{}'.format(stage, '-'.join(key.split())).ljust(20),
                           value='{}'.format(info[key]),
                           color=colors[stage],
                           bold=True if 'f1' in key else False)
            if stage == 'test':
                # if config.mode in ['start', 'end']:
                #     current_performance = info['modified_f1']
                # else:
                current_performance = info['micro_f1']
                if current_performance > previous_best:
                    previous_best = current_performance
                    saving_info = model.save(suffix='best')
                    logger.log(key='', value='epoch {} {}'.format(epoch_idx, saving_info), lf='', color='green')


if __name__ == '__main__':
    logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description='configuration')
    parser.add_argument('-config',
                        type=str,
                        default='genia/bidirectional/basic')
    args = parser.parse_args()
    config = Config('./configs/{}.json'.format(args.config))
    set_random_seed(seed=config.randomseed)
    process(config)

