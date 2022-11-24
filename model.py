import torch
import json
import time
import numpy as np

from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from math import ceil
from os.path import join, basename, dirname
from prettytable import PrettyTable

from utils import check_gpu_device, confirm_directory, relative_path


class Config(object):

    def __init__(self, path):
        self.config_path = path
        with open(path, 'r') as f:
            config_dict = json.load(f)
        for key in config_dict.keys():
            if isinstance(config_dict[key], str):
                exec("self.{}='{}'".format(key, config_dict[key]))
            else:
                exec("self.{}={}".format(key, config_dict[key]))
        self.name = basename(self.config_path).split('.')[0]
        self.data = basename(dirname(dirname(self.config_path)))
        self.mode = basename(dirname(self.config_path))
        self.time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    def generate_tag(self):
        self.tag = '{}-{}-{}-{}'.format(self.data, self.mode, self.name, self.time)

    def info(self):
        config_table = PrettyTable(['hyper-parameters',
                                    'value1',
                                    'value2',
                                    'value3'
                                    ])
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], dict):
                for idx_1, tiny_key in enumerate(self.__dict__[key].keys()):
                    if isinstance(self.__dict__[key][tiny_key], dict):
                        for idx_2, tiny_tiny_key in enumerate(self.__dict__[key][tiny_key].keys()):
                            config_table.add_row([key if idx_1 == idx_2 == 0 else ' ',
                                                  tiny_key if idx_2 == 0 else ' ',
                                                  tiny_tiny_key,
                                                  self.__dict__[key][tiny_key][tiny_tiny_key]])
                    else:
                        config_table.add_row([key if idx_1 == 0 else ' ', tiny_key, self.__dict__[key][tiny_key], ' '])
            else:
                config_table.add_row([key, self.__dict__[key], ' ', ' '])
        print(config_table)

    def mark_down(self):
        lines = ['|hyper-parameters|value1|value2|value3|\n',
                 '|----------------|------|------|------|\n']
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], dict):
                for idx_1, tiny_key in enumerate(self.__dict__[key].keys()):
                    if isinstance(self.__dict__[key][tiny_key], dict):
                        for idx_2, tiny_tiny_key in enumerate(self.__dict__[key][tiny_key].keys()):
                            lines.append("|{}|{}|{}|{}|\n".format(key if idx_1 == idx_2 == 0 else ' ',
                                                                  tiny_key if idx_2 == 0 else ' ',
                                                                  tiny_tiny_key,
                                                                  self.__dict__[key][tiny_key][tiny_tiny_key]))
                    else:
                        lines.append("|{}|{}|{}|{}|\n".format(key if idx_1 == 0 else ' ',
                                                              tiny_key,
                                                              self.__dict__[key][tiny_key],
                                                              ' '))
            else:
                lines.append("|{}|{}|{}|{}|\n".format(key, self.__dict__[key], ' ', ' '))
        return ''.join(lines)


class FocalLoss(nn.Module):
    def __init__(self, **kwarg):
        """
        focal_loss = -α(1-yi)**γ *ce_loss(xi,yi)
        :param alpha:
        :param gamma: γ, focusing parameter (modulating factor)
        :param num_classes:
        """
        super(FocalLoss, self).__init__()
        if "num_classes" in kwarg.keys():
            num_classes = kwarg["num_classes"]
        else:
            num_classes = 23

        if "alpha" in kwarg.keys():
            alpha = kwarg['alpha']
        else:
            alpha = [1] * num_classes

        if "gamma" in kwarg.keys():
            gamma = kwarg['gamma']
        else:
            gamma = 2

        self.num_classes = torch.tensor(num_classes)
        self.alpha = torch.tensor(alpha, requires_grad=False)
        self.gamma = torch.tensor(gamma, requires_grad=False)
        if torch.cuda.is_available():
            self.alpha = self.alpha.cuda()
            self.gamma = self.gamma.cuda()

    def forward(self, inputs, targets, reduction='none'):
        """
        focal_loss损失计算
        :param inputs: size:[B, N, C] or [B, C]
        :param targets: size:[B, N] or [B]
        :param reduction:
        :return:
        """
        if self.num_classes == 1:
            targets = targets.int()
            prob = torch.sigmoid(inputs)
            prob = torch.clamp(prob, min=1e-7, max=1 - 1e-7).view(-1)
            log_prob = torch.log(prob)
            log_neg_prob = torch.log(torch.tensor(1) - prob)
            loss = - self.alpha * targets * torch.pow(torch.tensor(1) - prob, self.gamma) * log_prob
            loss += - (torch.tensor(1) - self.alpha) * (1 - targets) * torch.pow(prob, self.gamma) * log_neg_prob
        else:
            prob = torch.softmax(inputs, dim=-1)
            prob = torch.clamp(prob, min=1e-7, max=1 - 1e-7)
            # prob = prob.view(-1, self.num_classes)
            prob = prob.gather(1, targets.view(-1, 1)).squeeze(1)
            alpha = self.alpha.gather(0, targets.view(-1))
            log_prob = torch.log(prob)
            loss = - alpha * torch.pow(torch.tensor(1) - prob, self.gamma) * log_prob
        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
        elif reduction == 'none':
            pass
        else:
            raise TypeError('illegal reduction type')
        return loss


class FullyConnected(nn.Module):

    def __init__(self, implement_dict):
        super(FullyConnected, self).__init__()
        self.config = implement_dict
        self._init_components()
        if torch.cuda.is_available():
            self.cuda()

    def _init_components(self):
        self.linear1 = nn.Linear(self.config['dim_in'],
                                 self.config['dim_hid'])
        self.linear2 = nn.Linear(self.config['dim_hid'],
                                 self.config['dim_out'])
        self.dropout = nn.Dropout(self.config['dropout'])

    def forward(self, in_feature):
        hidden_feature = torch.relu(self.linear1.forward(in_feature))
        hidden_feature = self.dropout.forward(hidden_feature)
        out_feature = self.linear2(hidden_feature)
        return out_feature


class Encoder(nn.Module):

    def __init__(self, implement_dict, mode='context'):
        super(Encoder, self).__init__()
        self.config = implement_dict
        self.mode = mode
        self.pretrained_bert = self.config["bert"]
        self._init_components()
        if torch.cuda.is_available():
            self.cuda()

    def _init_components(self):
        self.char_embedding = nn.Embedding(embedding_dim=self.config["dim_char"],
                                           num_embeddings=self.config["num_char"])
        self.char_lstm = nn.LSTM(input_size=self.config["dim_char"],
                                 hidden_size=self.config["dim_char_lstm"],
                                 num_layers=self.config["num_char_layer"],
                                 bidirectional=True)
        self.word_embedding = nn.Embedding(embedding_dim=self.config["dim_word"],
                                           num_embeddings=self.config["num_word"])
        self.pos_embedding = nn.Embedding(embedding_dim=self.config["dim_pos"],
                                          num_embeddings=self.config["num_pos"])
        with open('./dicts/{}/word2vec.npy'.format(self.config['data']), 'rb') as f:
            word2vec = torch.tensor(np.load(f))
        self.word_embedding.from_pretrained(embeddings=word2vec)
        self.word_embedding.requires_grad_(self.config['train_word_embedding'])
        self.bert = BertModel.from_pretrained(self.pretrained_bert)
        if self.config['num_lstm_layer'] != 0:
            dim_lstm_in = 2 * self.config['dim_char_lstm'] + self.config['dim_word'] + self.config['dim_pos'] + self.config['dim_bert']
            self.bilstm = nn.LSTM(input_size=dim_lstm_in,
                                  hidden_size=int(self.config['dim_lstm']/2),
                                  num_layers=self.config['num_lstm_layer'],
                                  bidirectional=True)
        self.char_dropout = nn.Dropout(p=self.config['char_dropout'])
        self.dropout = nn.Dropout(p=self.config["dropout"])

    def char_encode(self, batch):
        char_idx = batch['char_id']
        char_length = batch['char_length']

        char_idx = check_gpu_device(char_idx)
        char_embed = self.char_embedding.forward(char_idx)
        char_pack_padded_embed = pack_padded_sequence(
            input=char_embed,
            lengths=char_length,
            batch_first=True,
            enforce_sorted=False
        )
        char_encoded, (_, _) = self.char_lstm(char_pack_padded_embed)
        char_encoded, _ = pad_packed_sequence(char_encoded, batch_first=True, padding_value=0)
        char_encoded = torch.max(char_encoded, dim=1)[0]
        char_encoded = self.char_dropout(char_encoded)
        return char_encoded

    def forward(self, batch, segment_length=1024):
        char_encoded = self.char_encode(batch)
        word_encoded = self.word_embedding.forward(check_gpu_device(batch['word_id']))
        pos_encoded = self.pos_embedding.forward(check_gpu_device(batch['pos_id']))
        batch_idx = batch['bert_id']
        batch_mask = batch['mask']

        batch_idx = check_gpu_device(batch_idx)
        batch_mask = check_gpu_device(batch_mask)
        batch_length = batch_idx.size(1)
        num_seg = ceil(batch_length/segment_length)
        bert_encoded = []
        for i in range(num_seg):
            init_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, batch_length)
            seg_idx = batch_idx[:, init_idx: end_idx]
            seg_mask = batch_mask[:, init_idx: end_idx]
            # print(seg_idx.size())
            # print(seg_mask.size())
            seg_encoded, _ = self.bert.forward(input_ids=seg_idx,
                                               attention_mask=seg_mask,
                                               return_dict=False)
            bert_encoded.append(seg_encoded)
        bert_encoded = torch.cat(bert_encoded, dim=1)

        shift = 0
        if self.config['data'] == 'genia':
            batch_encoded = []
            for word, bert, pos, length in zip(
                    word_encoded,
                    bert_encoded,
                    pos_encoded,
                    batch['length']):
                bert = bert[:length]
                word = word[:length]
                pos = pos[:length]
                sample_char = char_encoded[shift: shift + length]
                shift += length
                sample_memory_list = [bert, word, pos, sample_char]
                sample_encoded = torch.cat(sample_memory_list, dim=-1)
                batch_encoded.append(sample_encoded)
            content_length = torch.tensor(batch['length'])
        elif self.config['data'] in ['ace04', 'ace05', 'kbp17']:
            batch_encoded = []
            content_length = []
            for word, bert, pos, context_length, [l_boundary, r_boundary] in zip(
                    word_encoded,
                    bert_encoded,
                    pos_encoded,
                    batch['length'],
                    batch['sentence_boundary']):
                bert = bert[l_boundary: r_boundary]
                word = word[l_boundary: r_boundary]
                pos = pos[l_boundary: r_boundary]
                sample_char = char_encoded[shift + l_boundary: shift + r_boundary]
                shift += context_length
                sample_memory_list = [bert, word, pos, sample_char]
                sample_encoded = torch.cat(sample_memory_list, dim=-1)
                batch_encoded.append(sample_encoded)
                content_length.append(r_boundary - l_boundary)
        else:
            raise ValueError("unregistered data {}".format(self.config['data']))

        batch_encoded = pad_sequence(
            sequences=batch_encoded,
            batch_first=True,
            padding_value=0
        )

        if self.config['num_lstm_layer'] != 0:
            packed_padded_batch_w_encoded = pack_padded_sequence(
                input=batch_encoded,
                lengths=torch.tensor(content_length),
                batch_first=True,
                enforce_sorted=False
            )
            batch_encoded, (_, _) = self.bilstm(packed_padded_batch_w_encoded)
            batch_encoded, _ = pad_packed_sequence(batch_encoded, batch_first=True, padding_value=0)
        batch_encoded = self.dropout(batch_encoded)
        return batch_encoded


class SubSequenceLSTMEncoder(nn.Module):

    def __init__(self, implement_dict):
        super(SubSequenceLSTMEncoder, self).__init__()
        self.config = implement_dict
        self._init_component()

    def _init_component(self):
        self.lstm = nn.LSTM(input_size=self.config['dim_lstm'],
                            hidden_size=self.config['dim_lstm'],
                            num_layers=self.config['num_layer'],
                            bidirectional=False,
                            batch_first=True)

    def forward(self, query_subseq, **kwargs):
        encoded, _ = self.lstm.forward(query_subseq)
        return encoded
        # return encoded


class PositionEncoder(nn.Module):

    def __init__(self, dim_position, max_len=5000):
        super(PositionEncoder, self).__init__()
        self.dim_position = dim_position
        self.max_length = max_len
        self._init_component()

    def _init_component(self):
        pe = torch.zeros(size=[self.max_length, self.dim_position])
        position = torch.arange(0, self.max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.dim_position, 2) * -(torch.log(torch.tensor(10000.)) / self.dim_position))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input_tensor):
        output_tensor = input_tensor + check_gpu_device(
            self.pe[:, :input_tensor.size(1)]
        )

        return output_tensor


class SubSequenceAttEncoder(nn.Module):
    def __init__(self, implement_dict):
        super(SubSequenceAttEncoder, self).__init__()
        self.config = implement_dict
        self._init_component()

    def _init_component(self):
        self.position_encoder = PositionEncoder(dim_position=self.config['dim_in'])
        self.query_embedding = nn.Linear(in_features=self.config['dim_in'],
                                         out_features=self.config['dim_model'])
        self.key_embedding = nn.Linear(in_features=self.config['dim_in'],
                                       out_features=self.config['dim_model'])
        self.value_embedding = nn.Linear(in_features=self.config['dim_in'],
                                         out_features=self.config['dim_model'])
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config['dim_model'],
                                                   nhead=self.config['num_head'],
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=self.config['num_layer'])

    def forward(self, query_subseq, subseq_mask):
        query_subseq_with_position = self.position_encoder.forward(query_subseq)
        key_padding_mask = torch.cat([torch.ones_like(subseq_mask[:, 0]).unsqueeze(1), subseq_mask], dim=-1).float()
        key_padding_mask = key_padding_mask.eq(0)
        encoded = self.encoder.forward(src=query_subseq_with_position,
                                       src_key_padding_mask=key_padding_mask)
        return encoded[:, 1:, :]


class BoundaryIdentifier(nn.Module):

    def __init__(self, implement_dict, name='boundary_identifier', tag='', mode="context"):
        super(BoundaryIdentifier, self).__init__()
        self.name = name
        self.tag = tag
        self.mode = mode
        self.config = implement_dict
        self.data = implement_dict['encoder']['data']
        self._init_components()
        if torch.cuda.is_available():
            self.cuda()

    def _init_components(self):
        self.encoder = Encoder(self.config['encoder'], mode=self.mode)
        self.fc_layer = FullyConnected(self.config['fc_layer'])

    def forward(self, batch, segment_length=512):
        encoded = self.encoder.forward(batch, segment_length=segment_length)
        scores = self.fc_layer.forward(encoded)
        return scores

    def save(self, suffix=''):
        confirm_directory('./check_points')
        folder = relative_path('./check_points/{}/{}'.format(self.data, self.name))
        confirm_directory(dirname(folder))
        confirm_directory(folder)
        if len(suffix) == 0:
            path = join(folder, '{}'.format(self.tag) + '.pth')
        else:
            path = join(folder, '{}-{}'.format(self.tag, suffix) + '.pth')
        confirm_directory(folder)
        torch.save(self.state_dict(), path)
        return '{} saved'.format(path)

    def load(self, path=''):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        return '{} loaded'.format(path)


class LocalHyperGraphBuilder(nn.Module):

    def __init__(self, implement_dict, name='local_hyper_graph_builder', tag='', mode='context'):
        super(LocalHyperGraphBuilder, self).__init__()
        self.name = name
        self.mode = mode
        self.tag = tag
        self.config = implement_dict
        self.data = implement_dict['encoder']['data']
        self.subsequence_encoder_options = {"lstm": SubSequenceLSTMEncoder,
                                            "attention": SubSequenceAttEncoder}
        self._init_component()
        if torch.cuda.is_available():
            self.cuda()

    def _init_component(self):
        self.boundary_encoder = Encoder(self.config['encoder'], mode=self.mode)
        self.content_encoder = Encoder(self.config['encoder'], mode=self.mode)
        self.fc_layer = FullyConnected(self.config['fc_layer'])
        self.subsequence_encoder = self.subsequence_encoder_options[self.config['mode']](self.config['sub_sequence_encoder'])

    def generate_subseq(self, batch):
        boundary_encoded = self.boundary_encoder.forward(batch, segment_length=512 if self.data == 'genia' else 1024)
        content_encoded = self.content_encoder.forward(batch, segment_length=512 if self.data == 'genia' else 1024)
        query_sub_sequence = []
        sub_sequence_labels = []
        sub_sequence_masks = []
        if self.config['direction'] == 'forward':
            for content, boundary, start_idx, sub_sequence_label, sub_sequence_boundary in zip(
                content_encoded,
                boundary_encoded,
                batch['start'],
                batch['sub_sequence_label'],
                batch['sub_sequence_boundary']
            ):
                boundary_query = [boundary[idx] for idx in start_idx]
                sub_sequence = [content[boundary[0]: boundary[1]] for boundary in sub_sequence_boundary]
                query_sub_sequence.extend([torch.cat([query.unsqueeze(0),
                                                      sub_seq], dim=0)
                                           for query, sub_seq in zip(boundary_query,
                                                                     sub_sequence)])
                sub_sequence_labels.extend([check_gpu_device(torch.tensor(label)) for label in sub_sequence_label])
                sub_sequence_masks.extend([torch.ones_like(check_gpu_device(torch.tensor(label)))
                                           for label in sub_sequence_label])
        else:
            for content, boundary, end_idx, sub_sequence_label, sub_sequence_boundary in zip(
                content_encoded,
                boundary_encoded,
                batch['end'],
                batch['sub_sequence_label_back'],
                batch['sub_sequence_boundary_back']
            ):
                boundary_query = [boundary[idx - 1] for idx in end_idx]
                sub_sequence = [content[boundary[0]: boundary[1]].flip(dims=(0,)) for boundary in sub_sequence_boundary]
                query_sub_sequence.extend([torch.cat([sub_seq,
                                                      query.unsqueeze(0)
                                                      ], dim=0)
                                           for query, sub_seq in zip(boundary_query,
                                                                     sub_sequence)])
                sub_sequence_labels.extend([check_gpu_device(torch.tensor(label)) for label in sub_sequence_label])
                sub_sequence_masks.extend([torch.ones_like(check_gpu_device(torch.tensor(label)))
                                           for label in sub_sequence_label])
        query_sub_sequence = pad_sequence(
            sequences=query_sub_sequence,
            padding_value=0,
            batch_first=True
        )
        sub_sequence_labels = pad_sequence(
            sequences=sub_sequence_labels,
            padding_value=0,
            batch_first=True
        )
        sub_sequence_mask = pad_sequence(
            sequences=sub_sequence_masks,
            padding_value=0,
            batch_first=True
        )
        return query_sub_sequence, sub_sequence_labels, sub_sequence_mask

    def forward(self, batch):
        query_subseq, subseq_labels, subseq_mask = self.generate_subseq(batch=batch)

        subseq_encoded = self.subsequence_encoder.forward(query_subseq=query_subseq,
                                                          subseq_mask=subseq_mask)[:, 1:, :]
        predicted_scores = self.fc_layer.forward(subseq_encoded)
        predicted_labels = torch.argmax(predicted_scores, dim=-1) * subseq_mask
        return predicted_scores, predicted_labels, subseq_labels, subseq_mask

    def save(self, suffix=''):
        confirm_directory('./check_points')
        folder = relative_path('./check_points/{}/{}'.format(self.data, self.name))
        confirm_directory(dirname(folder))
        confirm_directory(folder)
        if len(suffix) == 0:
            path = join(folder, '{}'.format(self.tag) + '.pth')
        else:
            path = join(folder, '{}-{}'.format(self.tag, suffix) + '.pth')
        confirm_directory(folder)
        torch.save(self.state_dict(), path)
        return '{} saved'.format(path)

    def load(self, path=''):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        return '{} loaded'.format(path)


class LocalHyperGraphBuilderSingle(nn.Module):

    def __init__(self, implement_dict, name='local_hyper_graph_builder', tag='', mode='context'):
        super(LocalHyperGraphBuilderSingle, self).__init__()
        self.name = name
        self.mode = mode
        self.tag = tag
        self.config = implement_dict
        self.data = implement_dict['encoder']['data']
        self.subsequence_encoder_options = {"lstm": SubSequenceLSTMEncoder,
                                            "attention": SubSequenceAttEncoder}
        self._init_component()
        if torch.cuda.is_available():
            self.cuda()

    def _init_component(self):
        self.boundary_encoder = Encoder(self.config['encoder'], mode=self.mode)
        self.content_encoder = Encoder(self.config['encoder'], mode=self.mode)
        self.fc_layer = FullyConnected(self.config['fc_layer'])
        self.subsequence_encoder = self.subsequence_encoder_options[self.config['mode']](self.config['sub_sequence_encoder'])

    def generate_subseq(self, batch):
        # boundary_encoded = self.boundary_encoder.forward(batch, segment_length=512 if self.data == 'genia' else 1024)
        content_encoded = self.content_encoder.forward(batch, segment_length=512 if self.data == 'genia' else 1024)
        query_sub_sequence = []
        sub_sequence_labels = []
        sub_sequence_masks = []
        if self.config['direction'] == 'forward':
            for content, start_idx, sub_sequence_label, sub_sequence_boundary in zip(
                content_encoded,
                batch['start'],
                batch['sub_sequence_label'],
                batch['sub_sequence_boundary']
            ):
                boundary_query = [content[idx] for idx in start_idx]
                sub_sequence = [content[boundary[0]: boundary[1]] for boundary in sub_sequence_boundary]
                query_sub_sequence.extend([torch.cat([query.unsqueeze(0),
                                                      sub_seq], dim=0)
                                           for query, sub_seq in zip(boundary_query,
                                                                     sub_sequence)])
                sub_sequence_labels.extend([check_gpu_device(torch.tensor(label)) for label in sub_sequence_label])
                sub_sequence_masks.extend([torch.ones_like(check_gpu_device(torch.tensor(label)))
                                           for label in sub_sequence_label])
        else:
            for content, end_idx, sub_sequence_label, sub_sequence_boundary in zip(
                content_encoded,
                batch['end'],
                batch['sub_sequence_label_back'],
                batch['sub_sequence_boundary_back']
            ):
                boundary_query = [content[idx - 1] for idx in end_idx]
                sub_sequence = [content[boundary[0]: boundary[1]].flip(dims=(0,)) for boundary in sub_sequence_boundary]
                query_sub_sequence.extend([torch.cat([sub_seq,
                                                      query.unsqueeze(0)
                                                      ], dim=0)
                                           for query, sub_seq in zip(boundary_query,
                                                                     sub_sequence)])
                sub_sequence_labels.extend([check_gpu_device(torch.tensor(label)) for label in sub_sequence_label])
                sub_sequence_masks.extend([torch.ones_like(check_gpu_device(torch.tensor(label)))
                                           for label in sub_sequence_label])
        query_sub_sequence = pad_sequence(
            sequences=query_sub_sequence,
            padding_value=0,
            batch_first=True
        )
        sub_sequence_labels = pad_sequence(
            sequences=sub_sequence_labels,
            padding_value=0,
            batch_first=True
        )
        sub_sequence_mask = pad_sequence(
            sequences=sub_sequence_masks,
            padding_value=0,
            batch_first=True
        )
        return query_sub_sequence, sub_sequence_labels, sub_sequence_mask

    def forward(self, batch):
        query_subseq, subseq_labels, subseq_mask = self.generate_subseq(batch=batch)

        subseq_encoded = self.subsequence_encoder.forward(query_subseq=query_subseq,
                                                          subseq_mask=subseq_mask)[:, 1:, :]
        predicted_scores = self.fc_layer.forward(subseq_encoded)
        predicted_labels = torch.argmax(predicted_scores, dim=-1) * subseq_mask
        return predicted_scores, predicted_labels, subseq_labels, subseq_mask

    def save(self, suffix=''):
        confirm_directory('./check_points')
        folder = relative_path('./check_points/{}/{}'.format(self.data, self.name))
        confirm_directory(dirname(folder))
        confirm_directory(folder)
        if len(suffix) == 0:
            path = join(folder, '{}'.format(self.tag) + '.pth')
        else:
            path = join(folder, '{}-{}'.format(self.tag, suffix) + '.pth')
        confirm_directory(folder)
        torch.save(self.state_dict(), path)
        return '{} saved'.format(path)

    def load(self, path=''):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        return '{} loaded'.format(path)


class LocalHyperGraphBuilderSingleNone(nn.Module):

    def __init__(self, implement_dict, name='local_hyper_graph_builder', tag='', mode='context'):
        super(LocalHyperGraphBuilderSingleNone, self).__init__()
        self.name = name
        self.mode = mode
        self.tag = tag
        self.config = implement_dict
        self.data = implement_dict['encoder']['data']
        self.subsequence_encoder_options = {"lstm": SubSequenceLSTMEncoder,
                                            "attention": SubSequenceAttEncoder}
        self._init_component()
        if torch.cuda.is_available():
            self.cuda()

    def _init_component(self):
        self.boundary_encoder = Encoder(self.config['encoder'], mode=self.mode)
        self.content_encoder = Encoder(self.config['encoder'], mode=self.mode)
        self.fc_layer = FullyConnected(self.config['fc_layer'])
        self.subsequence_encoder = self.subsequence_encoder_options[self.config['mode']](self.config['sub_sequence_encoder'])

    def generate_subseq(self, batch):
        # boundary_encoded = self.boundary_encoder.forward(batch, segment_length=512 if self.data == 'genia' else 1024)
        content_encoded = self.content_encoder.forward(batch, segment_length=512 if self.data == 'genia' else 1024)
        query_sub_sequence = []
        sub_sequence_labels = []
        sub_sequence_masks = []
        if self.config['direction'] == 'forward':
            for content, start_idx, sub_sequence_label, sub_sequence_boundary in zip(
                content_encoded,
                batch['start'],
                batch['sub_sequence_label'],
                batch['sub_sequence_boundary']
            ):
                boundary_query = [content[idx] for idx in start_idx]
                sub_sequence = [content[boundary[0]: boundary[1]] for boundary in sub_sequence_boundary]
                query_sub_sequence.extend(sub_sequence)
                sub_sequence_labels.extend([check_gpu_device(torch.tensor(label)) for label in sub_sequence_label])
                sub_sequence_masks.extend([torch.ones_like(check_gpu_device(torch.tensor(label)))
                                           for label in sub_sequence_label])
        else:
            for content, end_idx, sub_sequence_label, sub_sequence_boundary in zip(
                content_encoded,
                batch['end'],
                batch['sub_sequence_label_back'],
                batch['sub_sequence_boundary_back']
            ):
                boundary_query = [content[idx - 1] for idx in end_idx]
                sub_sequence = [content[boundary[0]: boundary[1]].flip(dims=(0,)) for boundary in sub_sequence_boundary]
                query_sub_sequence.extend(sub_sequence)
                sub_sequence_labels.extend([check_gpu_device(torch.tensor(label)) for label in sub_sequence_label])
                sub_sequence_masks.extend([torch.ones_like(check_gpu_device(torch.tensor(label)))
                                           for label in sub_sequence_label])
        query_sub_sequence = pad_sequence(
            sequences=query_sub_sequence,
            padding_value=0,
            batch_first=True
        )
        sub_sequence_labels = pad_sequence(
            sequences=sub_sequence_labels,
            padding_value=0,
            batch_first=True
        )
        sub_sequence_mask = pad_sequence(
            sequences=sub_sequence_masks,
            padding_value=0,
            batch_first=True
        )
        return query_sub_sequence, sub_sequence_labels, sub_sequence_mask

    def forward(self, batch):
        query_subseq, subseq_labels, subseq_mask = self.generate_subseq(batch=batch)

        subseq_encoded = self.subsequence_encoder.forward(query_subseq=query_subseq,
                                                          subseq_mask=subseq_mask)
        predicted_scores = self.fc_layer.forward(subseq_encoded)
        predicted_labels = torch.argmax(predicted_scores, dim=-1) * subseq_mask
        return predicted_scores, predicted_labels, subseq_labels, subseq_mask

    def save(self, suffix=''):
        confirm_directory('./check_points')
        folder = relative_path('./check_points/{}/{}'.format(self.data, self.name))
        confirm_directory(dirname(folder))
        confirm_directory(folder)
        if len(suffix) == 0:
            path = join(folder, '{}'.format(self.tag) + '.pth')
        else:
            path = join(folder, '{}-{}'.format(self.tag, suffix) + '.pth')
        confirm_directory(folder)
        torch.save(self.state_dict(), path)
        return '{} saved'.format(path)

    def load(self, path=''):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        return '{} loaded'.format(path)


class BidirectionalLocalHyperGraphBuilder(nn.Module):

    def __init__(self, implement_dict, name='bidirectional_local_hyper_graph_builder', tag='', mode='context'):
        super(BidirectionalLocalHyperGraphBuilder, self).__init__()
        self.name = name
        self.mode = mode
        self.tag = tag
        self.config = implement_dict
        self.data = implement_dict['encoder']['data']
        self.subsequence_encoder_options = {"lstm": SubSequenceLSTMEncoder,
                                            "attention": SubSequenceAttEncoder}
        self._init_component()
        if torch.cuda.is_available():
            self.cuda()

    def _init_component(self):
        self.forward_encoder = Encoder(self.config['encoder'], mode=self.mode)
        self.backward_encoder = Encoder(self.config['encoder'], mode=self.mode)
        self.content_encoder = Encoder(self.config['encoder'], mode=self.mode)
        self.fc_layer = FullyConnected(self.config['fc_layer'])
        self.forward_subsequence_encoder = self.subsequence_encoder_options[self.config['mode']](
            self.config['sub_sequence_encoder'])
        self.backward_subsequence_encoder = self.subsequence_encoder_options[self.config['mode']](
            self.config['sub_sequence_encoder'])

    def generate_subseq(self, batch):
        forward_encoded = self.forward_encoder.forward(batch, segment_length=512 if self.data == 'genia' else 1024)
        backward_encoded = self.backward_encoder.forward(batch, segment_length=512 if self.data == 'genia' else 1024)
        query_sub_sequence = []
        sub_sequence_labels = []
        sub_sequence_masks = []
        query_sub_sequence_back = []
        sub_sequence_labels_back = []
        sub_sequence_masks_back = []
        for forward, backward, start_idx, sub_sequence_label, sub_sequence_boundary, end_idx, sub_sequence_label_back, sub_sequence_boundary_back in zip(
                forward_encoded,
                backward_encoded,
                batch['start'],
                batch['sub_sequence_label'],
                batch['sub_sequence_boundary'],
                batch['end'],
                batch['sub_sequence_label_back'],
                batch['sub_sequence_boundary_back']
        ):
            if len(batch['start']) != 0:
                boundary_query = [forward[idx] for idx in start_idx]
                sub_sequence = [backward[boundary[0]: boundary[1]] for boundary in sub_sequence_boundary]
                query_sub_sequence.extend([torch.cat([query.unsqueeze(0),
                                                      sub_seq], dim=0)
                                           for query, sub_seq in zip(boundary_query,
                                                                     sub_sequence)])
                sub_sequence_labels.extend([check_gpu_device(torch.tensor(label)) for label in sub_sequence_label])
                sub_sequence_masks.extend([torch.ones_like(check_gpu_device(torch.tensor(label)))
                                           for label in sub_sequence_label])
            if len(batch['end']) != 0:
                boundary_query_back = [backward[idx - 1] for idx in end_idx]
                sub_sequence_back = [forward[boundary[0]: boundary[1]].flip(dims=(0,)) for boundary in
                                     sub_sequence_boundary_back]
                query_sub_sequence_back.extend([torch.cat([sub_seq,
                                                           query.unsqueeze(0)
                                                           ], dim=0)
                                                for query, sub_seq in zip(boundary_query_back,
                                                                          sub_sequence_back)])
                sub_sequence_labels_back.extend([check_gpu_device(torch.tensor(label)) for label in sub_sequence_label_back])
                sub_sequence_masks_back.extend([torch.ones_like(check_gpu_device(torch.tensor(label)))
                                                for label in sub_sequence_label_back])
        query_sub_sequence = pad_sequence(
            sequences=query_sub_sequence,
            padding_value=0,
            batch_first=True
        )
        sub_sequence_labels = pad_sequence(
            sequences=sub_sequence_labels,
            padding_value=0,
            batch_first=True
        )
        sub_sequence_mask = pad_sequence(
            sequences=sub_sequence_masks,
            padding_value=0,
            batch_first=True
        )
        query_sub_sequence_back = pad_sequence(
            sequences=query_sub_sequence_back,
            padding_value=0,
            batch_first=True
        )
        sub_sequence_labels_back = pad_sequence(
            sequences=sub_sequence_labels_back,
            padding_value=0,
            batch_first=True
        )
        sub_sequence_masks_back = pad_sequence(
            sequences=sub_sequence_masks_back,
            padding_value=0,
            batch_first=True
        )
        return query_sub_sequence, sub_sequence_labels, sub_sequence_mask, \
               query_sub_sequence_back, sub_sequence_labels_back, sub_sequence_masks_back

    def forward(self, batch):
        query_subseq, subseq_labels, subseq_mask, query_subseq_back, subseq_labels_back, subseq_mask_back = self.generate_subseq(
            batch=batch)
        subseq_encoded = self.forward_subsequence_encoder.forward(query_subseq=query_subseq,
                                                                  subseq_mask=subseq_mask)
        predicted_scores = self.fc_layer.forward(subseq_encoded)
        predicted_labels = torch.argmax(predicted_scores, dim=-1) * subseq_mask

        subseq_encoded_back = self.backward_subsequence_encoder.forward(query_subseq=query_subseq_back,
                                                                        subseq_mask=subseq_mask_back)
        predicted_scores_back = self.fc_layer.forward(subseq_encoded_back)
        predicted_labels_back = torch.argmax(predicted_scores_back, dim=-1) * subseq_mask_back
        return predicted_scores, predicted_labels, subseq_labels, subseq_mask, \
               predicted_scores_back, predicted_labels_back, subseq_labels_back, subseq_mask_back

    def save(self, suffix=''):
        confirm_directory('./check_points')
        folder = relative_path('./check_points/{}/{}'.format(self.data, self.name))
        confirm_directory(dirname(folder))
        confirm_directory(folder)
        if len(suffix) == 0:
            path = join(folder, '{}'.format(self.tag) + '.pth')
        else:
            path = join(folder, '{}-{}'.format(self.tag, suffix) + '.pth')
        confirm_directory(folder)
        torch.save(self.state_dict(), path)
        return '{} saved'.format(path)

    def load(self, path=''):
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        return '{} loaded'.format(path)

# if __name__ == '__main__':
#     config = Config(path='./configs/ace04/localhypergraph/version1.json')
#     model = LocalHyperGraphBuilder(implement_dict=config.model)
