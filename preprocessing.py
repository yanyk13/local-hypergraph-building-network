import json
import re
import torch
import numpy as np
import _pickle as pkl

from functools import reduce
from os.path import join, basename, exists
from glob import glob
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from utils import confirm_directory, ctext, cprint


def add_special_token(special_token_list,
                      bert_path="./scibert_scivocab_cased"):
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertModel.from_pretrained(bert_path)
    if exists(("{}/added_tokens.json".format(bert_path))):
        with open("{}/added_tokens.json".format(bert_path), 'r', encoding='utf-8') as f:
            added_tokens = json.load(f)
        confirmed_special_token_list = list(filter(lambda x: x not in added_tokens, special_token_list))
    else:
        confirmed_special_token_list = special_token_list
    if len(confirmed_special_token_list) != 0:
        tokenizer.add_tokens(confirmed_special_token_list, special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        tokenizer.save_pretrained(bert_path)
        model.save_pretrained(bert_path)

    cprint('add special {} tokens: {}'.format(len(confirmed_special_token_list), ' '.join(confirmed_special_token_list)))


def convert_token(tokens, tokenizer):
    if len(tokens) != 0:
        tokenized_list = [tokenizer.tokenize(token) for token in tokens]
        length = [len(tokenized) for tokenized in tokenized_list]
        tokens = reduce(lambda x, y: x + y, tokenized_list)
        return tokens, length
    else:
        return [], 0


def convert_sentence_data(path, tokenizer, save_path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = []
    for sentence in tqdm(data, desc='converting sentence data'):
        sample_id = sentence['org_id']
        tokens, length = convert_token(sentence['tokens'], tokenizer)
        pos = reduce(lambda x, y: x + y, [[pos] * l for pos, l in zip(sentence['pos'], length)])
        entities = [{'start': sum(length[:entity['start']]),
                     'end': sum(length[:entity['end']]),
                     'type': entity['type']}
                    for entity in sentence['entities']]
        sample = {
            "id": sample_id,
            "tokens": tokens,
            "pos": pos,
            "entities": entities
        }
        samples.append(sample)

    save_basename = basename(save_path)
    with open(save_path.replace(save_basename, 'sentence_{}'.format(save_basename)), 'w', encoding='utf-8') as f:
        json.dump(samples, f)


def convert_context_data(path, tokenizer, save_path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = []
    continue_flag = [previous_sample['tokens'] == sample['ltokens'] for previous_sample, sample in zip(data[:-1], data[1:])]
    l_continue_flag = [False] + continue_flag
    r_continue_flag = continue_flag + [False]
    for idx, (l_flag, r_flag) in tqdm(enumerate(zip(l_continue_flag, r_continue_flag)), desc='converting context data', total=len(data)):
        if not l_flag:
            l_tokens = []
            l_pos = []
        else:
            l_tokens, l_length = convert_token(data[idx - 1]['tokens'], tokenizer)
            l_pos = reduce(lambda x, y: x + y, [[pos] * l for pos, l in zip(data[idx - 1]['pos'], l_length)])
        if not r_flag:
            r_tokens = []
            r_pos = []
        else:
            r_tokens, r_length = convert_token(data[idx + 1]['tokens'], tokenizer)
            r_pos = reduce(lambda x, y: x + y, [[pos] * l for pos, l in zip(data[idx + 1]['pos'], r_length)])

        tokens, length = convert_token(data[idx]['tokens'], tokenizer)
        pos = reduce(lambda x, y: x + y, [[pos] * l for pos, l in zip(data[idx]['pos'], length)])
        entities = [{'start': sum(length[:entity['start']]),
                     'end': sum(length[:entity['end']]),
                     'type': entity['type']}
                    for entity in data[idx]['entities']]
        sample = {
            "id": idx,
            "tokens": l_tokens + tokens + r_tokens,
            "pos": l_pos + pos + r_pos,
            "sentence_boundary": [len(l_tokens), len(l_tokens + tokens)],
            "entities": entities
        }
        samples.append(sample)

    save_basename = basename(save_path)
    with open(save_path.replace(save_basename, 'context_{}'.format(save_basename)), 'w', encoding='utf-8') as f:
        json.dump(samples, f)


def convert_bio_passage_data(path, tokenizer, save_path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = []
    sample = {'id': 'none'}
    shift = 0
    tokens = []
    chars = []
    pos = []
    entities = []
    # sentences = []
    sentence_boundary = [0]
    counter = 0
    previous_ltokens = []

    for sentence in tqdm(data, desc='converting passage data'):
        sentence_id = sentence['org_id']
        if sentence_id != sample['id']:
            if sample['id'] != 'none':
                sample['tokens'] = tokens
                sample['pos'] = pos
                sample['entities'] = entities
                # sample['sentences'] = sentences
                sample['sentence_boundary'] = sentence_boundary
                samples.append(sample)
            sample = {'id': sentence_id}
            shift = 0
            tokens = []
            chars = []
            pos = []
            entities = []
            sentences = []
            sentence_boundary = [0]
            counter = 0
            previous_ltokens = []
        ltokens, _ = convert_token(sentence['ltokens'], tokenizer)
        assert previous_ltokens == ltokens  # assert that the sentences are continuous
        sentence_tokens, length = convert_token(sentence['tokens'], tokenizer)
        tokens.extend(sentence_tokens)
        chars.extend([list(token) for token in sentence_tokens])
        pos.extend(reduce(lambda x, y: x + y, [[pos] * l for pos, l in zip(sentence['pos'], length)]))
        entities.extend([{'start': shift + sum(length[:entity['start']]),
                          'end': shift + sum(length[:entity['end']]),
                          'sen_id': counter,
                          'sen_start': sum(length[:entity['start']]),
                          'sen_end': sum(length[:entity['end']]),
                          'type': entity['type']}
                         for entity in sentence['entities']])

        if False in [''.join(sentence_tokens[sum(length[:entity['start']]): sum(length[:entity['end']])]) ==
               ''.join(tokens[shift + sum(length[:entity['start']]): shift + sum(length[:entity['end']])])
               for entity in sentence['entities']]:
            cprint('Oops', 'red')

        # sentences.append(sentence_tokens)
        shift += len(sentence_tokens)
        sentence_boundary.append(shift)
        counter += 1
        previous_ltokens = sentence_tokens

    save_basename = basename(save_path)
    with open(save_path.replace(save_basename, 'passage_{}'.format(save_basename)), 'w', encoding='utf-8') as f:
        json.dump(samples, f)


def get_bio_vec_dicts(directory):
    with open('./data/bio_vectors/vocab_bio.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    with open('./data/bio_vectors/vocab_embed_bio.npy', 'rb') as f:
        bio_embed = np.load(f)
    with open('./data/bio_vectors/genia_pos.json', 'r', encoding='utf-8') as f:
        genia_pos = json.load(f)

    samples = []
    for file in glob(join(directory, '*.json')):
        with open(file, 'r', encoding='utf-8') as f:
            samples.extend(json.load(f))
    cprint('{} samples loaded'.format(len(samples)), 'cyan')

    words = ['[padding]', '[unk]', '<ENT>', '</ENT>'] + list(set(reduce(lambda x, y: x + y,
                                                                        [sample['tokens'] for sample in samples])))
    chars = list(set(reduce(lambda x, y: x + y, [list(word) for word in words])))

    pos_zip_list = []
    for key in genia_pos.keys():
        pos_zip_list.append([key, genia_pos[key]])
    pos_zip_list.sort(key=lambda x: x[1], reverse=True)
    poses, _ = zip(*pos_zip_list[:20])
    poses = ['[padding]', '[unk]', '<ENT>', '</ENT>'] + list(poses)

    entities = reduce(lambda x, y: x + y, [sample['entities'] for sample in samples])
    types = ["None", "Continue"] + list(set([entity['type'] for entity in entities]))

    char2id = {}
    word2id = {}
    vectors = []
    pos2id = {}
    type2id = {}

    for id, char in enumerate(chars):
        char2id[char] = id

    for id, word in enumerate(['[padding]', '[unk]', '<ENT>', '</ENT>']):
        word2id[word] = id
        vectors.append(np.random.randn(200))

    for id, word in enumerate(list(vocab.keys())):
        word2id[word] = id + 4
        vectors.append(bio_embed[id])

    for id, pos in enumerate(poses):
        pos2id[pos] = id

    for id, cls in enumerate(types):
        type2id[cls] = id

    vectors = np.asarray(vectors)

    with open('./dicts/genia/char2id.pkl', 'wb') as f:
        pkl.dump(char2id, f)

    with open('./dicts/genia/word2id.pkl', 'wb') as f:
        pkl.dump(word2id, f)

    with open('./dicts/genia/word2vec.npy', 'wb') as f:
        np.save(f, vectors)

    with open('./dicts/genia/pos2id.pkl', 'wb') as f:
        pkl.dump(pos2id, f)

    with open('./dicts/genia/type2id.pkl', 'wb') as f:
        pkl.dump(type2id, f)


def get_vec_dicts(directory, prex):
    with open('./Glove/vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        cprint('vocab loaded', 'blue')
    with open('./Glove/word_embedding.npy', 'rb') as f:
        word_embedding = np.load(f)
        cprint('word vector loaded', 'blue')

    samples = []
    for file in glob(join(directory, 'context*.json')):
        with open(file, 'r', encoding='utf-8') as f:
            samples.extend(json.load(f))
    cprint('{} samples loaded'.format(len(samples)), 'cyan')

    words = ['[padding]', '[unk]', '<ENT>', '</ENT>'] + list(set(reduce(lambda x, y: x + y,
                                                                        [sample['tokens'] for sample in samples])))
    chars = list(set(reduce(lambda x, y: x + y, [list(word) for word in words])))
    poses = ['[padding]', '[unk]', '<ENT>', '</ENT>'] + list(set(reduce(lambda x, y: x + y,
                                                                        [sample['pos'] for sample in samples])))

    entities = reduce(lambda x, y: x + y, [sample['entities'] for sample in samples])
    types = ["None", "Continue"] + list(set([entity['type'] for entity in entities]))

    char2id = {}
    word2id = {}
    vectors = []
    pos2id = {}
    type2id = {}

    for id, char in enumerate(chars):
        char2id[char] = id

    for id, word in enumerate(['[padding]', '[unk]', '<ENT>', '</ENT>']):
        word2id[word] = id
        vectors.append(np.random.randn(300))

    word_counter = 4
    for word in tqdm(words[4:], desc="generating word dict"):
        if word in vocab.keys():
            word2id[word] = word_counter
            vectors.append(word_embedding[vocab[word]])
            word_counter += 1
        elif '##' in word:
            word2id[word] = word_counter
            vectors.append(np.random.randn(300))
            word_counter += 1

    for id, pos in enumerate(poses):
        pos2id[pos] = id

    for id, cls in enumerate(types):
        type2id[cls] = id

    vectors = np.asarray(vectors)

    with open('./dicts/{}/char2id.pkl'.format(prex), 'wb') as f:
        pkl.dump(char2id, f)

    with open('./dicts/{}/word2id.pkl'.format(prex), 'wb') as f:
        pkl.dump(word2id, f)

    with open('./dicts/{}/word2vec.npy'.format(prex), 'wb') as f:
        np.save(f, vectors)

    with open('./dicts/{}/pos2id.pkl'.format(prex), 'wb') as f:
        pkl.dump(pos2id, f)

    with open('./dicts/{}/type2id.pkl'.format(prex), 'wb') as f:
        pkl.dump(type2id, f)


def convert_glove():
    with open('Glove/glove.840B.300d.txt', 'r', encoding='utf-8') as f:
        glove = f.readlines()
    vocab = {}
    embedding = []
    for idx, line in tqdm(enumerate(glove), desc=ctext("converting glove_vector", 'cyan'), total=len(list(glove))):
        line = line.split()
        vocab[''.join(line[:-300])] = idx
        embedding.append([float(num) for num in line[-300:]])
    with open('./Glove/vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f)
    with open('./Glove/word_embedding.npy', 'wb') as f:
        np.save(f, embedding)
    cprint('glove word embedding saved with {} words'.format(len(vocab.keys())), color='blue')


if __name__ == '__main__':
    # convert_glove()
    # tokenizer = BertTokenizer.from_pretrained('./biobert-large-cased-v1.1-squad',
    #                                           do_lower_case=False)
    # add_special_token(special_token_list=['<ENT>', '</ENT>'],
    #                   bert_path='./biobert-large-cased-v1.1-squad')
    #
    # train_dev_path = './data/original/genia/genia_train_dev_context.json'
    # test_path = './data/original/genia/genia_test_context.json'
    # paths = [train_dev_path,
    #          test_path]
    # save_paths = ['./data/genia/train.json',
    #               './data/genia/test.json']
    # for path, save_path in zip(paths, save_paths):
    #     convert_sentence_data(path, tokenizer, save_path)
    #     convert_context_data(path, tokenizer, save_path)
    #     convert_bio_passage_data(path, tokenizer, save_path)

    # tokenizer = BertTokenizer.from_pretrained('./bert-large-cased',
    #                                           do_lower_case=False)
    # add_special_token(special_token_list=['<ENT>', '</ENT>'],
    #                   bert_path='./bert-large-cased')
    # train_path = './data/original/ace04/ace04_train_context.json'
    # dev_path = './data/original/ace04/ace04_dev_context.json'
    # test_path = './data/original/ace04/ace04_test_context.json'
    # paths = [train_path,
    #          dev_path,
    #          test_path]
    # confirm_directory('./data/ace04')
    # save_paths = ['./data/ace04/train.json',
    #               './data/ace04/dev.json',
    #               './data/ace04/test.json']
    #
    # for path, save_path in zip(paths, save_paths):
    #     convert_context_data(path, tokenizer, save_path)
    # get_vec_dicts(directory='./data/ace04',
    #               prex='ace04')
    #
    # tokenizer = BertTokenizer.from_pretrained('./bert-large-cased',
    #                                           do_lower_case=False)
    # add_special_token(special_token_list=['<ENT>', '</ENT>'],
    #                   bert_path='./bert-large-cased')
    #
    # train_path = './data/original/ace05/ace05_train_context.json'
    # dev_path = './data/original/ace05/ace05_dev_context.json'
    # test_path = './data/original/ace05/ace05_test_context.json'
    # paths = [train_path,
    #          dev_path,
    #          test_path]
    # confirm_directory('./data/ace05')
    # save_paths = ['./data/ace05/train.json',
    #               './data/ace05/dev.json',
    #               './data/ace05/test.json']
    #
    # for path, save_path in zip(paths, save_paths):
    #     # convert_context_data(path, tokenizer, save_path)
    #     convert_bio_passage_data(path, tokenizer, save_path)
    # get_vec_dicts(directory='./data/ace05',
    #               prex='ace05')
    #
    # tokenizer = BertTokenizer.from_pretrained('./bert-large-cased',
    #                                           do_lower_case=False)
    # add_special_token(special_token_list=['<ENT>', '</ENT>'],
    #                   bert_path='./bert-large-cased')
    #
    # train_path = './data/original/kbp17/kbp17_train_context.json'
    # dev_path = './data/original/kbp17/kbp17_dev_context.json'
    # test_path = './data/original/kbp17/kbp17_test_context.json'
    # paths = [train_path,
    #          dev_path,
    #          test_path]
    # confirm_directory('./data/kbp17')
    # save_paths = ['./data/kbp17/train.json',
    #               './data/kbp17/dev.json',
    #               './data/kbp17/test.json']
    #
    # for path, save_path in zip(paths, save_paths):
    #     convert_context_data(path, tokenizer, save_path)
    #     convert_bio_passage_data(path, tokenizer, save_path)
    get_vec_dicts(directory='./data/kbp17',
                  prex='kbp17')
