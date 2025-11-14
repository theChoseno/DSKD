"""PyTorch BERT-of-Theseus model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP, load_tf_weights_in_bert, gelu, gelu_new, \
    swish, mish, ACT2FN, BertLayerNorm, BertEmbeddings, BertSelfAttention, BertSelfOutput, BertAttention, \
    BertIntermediate, BertOutput, BertLayer, BertPooler, BertPredictionHeadTransform, BertLMPredictionHead, \
    BertOnlyMLMHead, BertOnlyNSPHead, BertPreTrainingHeads

import sys
import os

sys.path.append('..')

import re
import numpy as np

logger = logging.getLogger(__name__)


def DKL(_p, _q):
    return torch.sum(_p * (_p.log() - _q.log()), dim=-1)


def subspace_alignment_loss(Us, Ut):
    return torch.norm(Us.T @ Ut - torch.eye(Us.shape[1], device=Us.device), p='fro') ** 2


def removenan(x):
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    return x


def msym(x):
    return (x + x.T) / 2


def create_svd_with_custom_grad(rank_indeces):
    class SVDWithCustomGrad(torch.autograd.Function):
        ## SVD操作，反向传播计算梯度 ##
        @staticmethod
        def forward(ctx, W):
            U, S, V = torch.svd(W)  # m*r, r, n*r
            # U_k = U[:, rank_indeces]
            # S_k = S[rank_indeces]
            # V_k = V[:, rank_indeces]
            # ctx.save_for_backward(U_k, S_k, V_k)
            ctx.save_for_backward(U, S, V, W)
            return U, S, V  # m*r, r, n*r

        @staticmethod
        def backward(ctx, dU, dS, dV):  # m*k, k, n*k
            U, S, V, W = ctx.saved_tensors  # m*k, k, n*k

            # mask = torch.zeros(U.shape[1], dtype=torch.bool).to(U.device)
            # mask[rank_indeces] = True

            # U[:, ~mask] = 0
            # S[~mask] = 0
            # V[:, ~mask] = 0

            # dU[:, ~mask] = 0
            # dS[~mask] = 0
            # dV[:, ~mask] = 0

            s_f = torch.square(S)
            f = removenan(1.0 / (s_f.reshape(-1, 1) - s_f.reshape(1, -1)))
            F = torch.where(torch.eye(f.size(-1), device=f.device) == 1, torch.zeros_like(f), f)
            S = torch.diag(S)
            s = torch.diag(1 / torch.diag(S))
            # s = 1 / S
            Im = torch.eye(U.shape[0], device=U.device)
            In = torch.eye(V.shape[0], device=V.device)
            Ik = torch.eye(S.shape[0], device=S.device)

            #### v1 ####
            # grad1 = (U @ (F * (U.T @ dU - dU.T @ U)) @ S + (Im - U @ U.T) @ dU @ s) @ V.T
            # grad2 = U @ (Ik * dS) @ V.T
            # grad3 = U @ (S @ (F * (V.T @ dV - dV.T @ V)) @ V.T + s @ dV.T @ (In - V @ V.T))
            # grad = grad1 + grad3 + grad2

            #### v2 ####
            E = dV @ s
            if dU.shape[0] <= dV.shape[0]:
                grad1 = U @ E.T
                grad2 = U @ (E.T @ V) @ V.T
                grad3 = 2 * U @ msym(F * (S @ V.T @ E)) @ S @ V.T
                grad = grad1 - grad2 - grad3
            else:
                grad = 2 * U @ S @ msym(F.T * (V.T @ dV)) @ V.T

            return removenan(grad)

    return SVDWithCustomGrad


def get_param_from_string(module, param_string):
    param_path = param_string.split('.')
    for attr in param_path:
        module = getattr(module, attr)

    return module


class BertEncoder(nn.Module):
    def __init__(self, config, scc_n_layer=6):
        super(BertEncoder, self).__init__()
        self.prd_n_layer = config.num_hidden_layers
        self.scc_n_layer = scc_n_layer
        assert self.prd_n_layer % self.scc_n_layer == 0
        self.compress_ratio = self.prd_n_layer // self.scc_n_layer
        self.bernoulli = None
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(self.prd_n_layer)])
        self.scc_layer = nn.ModuleList([BertLayer(config) for _ in range(self.scc_n_layer)])

    def compute_svd_loss(self, choose_name, teacher_svd_tensor, rank_indeces, J_scores, k=32):
        total_svd_loss = 0
        student_svd_tensor = {}
        for module_name in choose_name:
            ind = int(re.findall(r"\d+", module_name)[0])
            ind_teacher = str(int(re.findall(r"\d+", module_name)[0]) * 2 + 1)

            module_name_path = module_name.replace(f"{ind}", f"[{ind_teacher}]")
            module_name_path = module_name_path.split('.', 1)[1]

            # layer_teacher = self.layer[int(ind_teacher)].state_dict()[module_name_teacher]
            # layer_student = self.scc_layer[int(ind)].state_dict()[module_name_teacher]
            # layer_teacher.requires_grad = True
            # layer_student.requires_grad = True

            layer_student = get_param_from_string(self.scc_layer[int(ind)], module_name_path)
            teacher_module_name = str(ind_teacher) + '.' + module_name_path
            # layer_teacher = self.layer[int(ind_teacher)].state_dict()[module_name_path]

            U_t = teacher_svd_tensor[teacher_module_name]['U']
            S_t = teacher_svd_tensor[teacher_module_name]['S']
            V_t = teacher_svd_tensor[teacher_module_name]['V']

            if len(rank_indeces) == 0 or len(J_scores) == 0:
                r_index = torch.arange(0, k).to(U_t.device)
                score_weight = torch.ones(k).to(U_t.device)
            else:
                r_index = rank_indeces[module_name]
                score_weight = J_scores[module_name]

            with torch.enable_grad():
                svd_Function = create_svd_with_custom_grad(r_index)
                U_s, S_s, V_s = svd_Function.apply(layer_student)

            # U_s, S_s, V_s = torch.svd(layer_student)
            # U_s, S_s, V_s = torch.linalg.svd(layer_student, full_matrices=False)
            # V_s = V_s.T

            if len(rank_indeces) == 0:
                sum_score_weight = score_weight.sum()
                score_weight = score_weight / (sum_score_weight + 1e-6)
            else:
                max_score = score_weight.max()
                min_score = score_weight.min()
                score_weight = (score_weight - min_score) / (max_score - min_score + 1e-6)
                # score_weight = torch.abs(score_weight)
                sum_score_weight = score_weight.sum()
                score_weight = score_weight / (sum_score_weight + 1e-6)

            # all_weight = torch.abs(score_weight)
            # all_weight = torch.sqrt(all_weight)

            # all_weight = torch.full((S_t.shape[0],), 1e-6).to(U_s.device)
            all_weight = torch.zeros(S_t.shape[0]).to(U_s.device)

            all_weight[r_index] = score_weight
            all_weight = torch.abs(all_weight)
            all_weight = torch.sqrt(all_weight)

            student_module_name = str(ind) + '.' + module_name_path
            student_svd_tensor[student_module_name] = {
                'U': U_s.detach(),
                'S': S_s.detach(),
                'V': V_s.detach()
            }

            # all_weight = torch.ones(k).to(U_s.device)
            # all_weight_sum = all_weight.sum()
            # all_weight = all_weight / all_weight_sum
            # all_weight = torch.sqrt(all_weight)
            all_weight = all_weight.to(U_s.device)
            # student_matrix = all_weight * U_sk @ torch.diag(S_sk) @ V_sk.T
            # teacher_matrix = all_weight * U_tk @ torch.diag(S_tk) @ V_tk.T
            student_matrix = all_weight * U_s @ torch.diag(S_s) @ V_s.T
            teacher_matrix = all_weight * U_t @ torch.diag(S_t) @ V_t.T
            svd_loss = torch.norm(student_matrix - teacher_matrix, p='fro') ** 2

            total_svd_loss += svd_loss
        total_svd_loss = total_svd_loss / (len(choose_name))
        return total_svd_loss, student_svd_tensor

    def teacher_layer_with_svd(self, choose_name, rank_indeces, k=32):
        teacher_weight_svd = {}
        for i in range(len(self.layer)):
            for module_name in choose_name:
                module_name_path = module_name.split('.', 1)[1]
                layer_teacher = self.layer[i].state_dict()[module_name_path]
                if len(rank_indeces) == 0:
                    r_index = torch.arange(0, k)
                else:
                    r_index = rank_indeces[module_name]
                U, S, V = torch.svd(layer_teacher)

                teacher_module_name = str(i) + '.' + module_name_path
                teacher_weight_svd[teacher_module_name] = {
                    'U': U,
                    'S': S,
                    'V': V
                }
        return teacher_weight_svd

    def set_replacing_rate(self, replacing_rate):
        if not 0 < replacing_rate <= 1:
            raise Exception('Replace rate must be in the range (0, 1]!')
        self.bernoulli = Bernoulli(torch.tensor([replacing_rate]))

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        scc_layer_index = []

        if self.training:
            inference_layers = []
            for i in range(self.scc_n_layer):
                if self.bernoulli.sample() == 1:  # REPLACE
                    inference_layers.append(self.scc_layer[i])
                    scc_layer_index.append(i)
                else:  # KEEP the original
                    for offset in range(self.compress_ratio):
                        inference_layers.append(self.layer[i * self.compress_ratio + offset])

        else:  # inference with compressed model
            inference_layers = self.scc_layer

        for i, layer_module in enumerate(inference_layers):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                                         encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs, scc_layer_index  # last-layer hidden state, (all hidden states), (all attentions)


class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError("Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(input_shape,
                                                                                                        attention_mask.shape))

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape,
                        encoder_attention_mask.shape))

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs, scc_layer_index = self.encoder(embedding_output,
                                                        attention_mask=extended_attention_mask,
                                                        head_mask=head_mask,
                                                        encoder_hidden_states=encoder_hidden_states,
                                                        encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
                                                      1:]  # add hidden_states and attentions if they are here
        return outputs, scc_layer_index  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                masked_lm_labels=None, next_sentence_label=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[
                                                                 2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


class BertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                masked_lm_labels=None, encoder_hidden_states=None, encoder_attention_mask=None, lm_labels=None, ):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                next_sentence_label=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


class BertForSequenceClassificationSVD(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSequenceClassificationSVD, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs, scc_layer_index = self.bert(input_ids,
                                             attention_mask=attention_mask,
                                             token_type_ids=token_type_ids,
                                             position_ids=position_ids,
                                             head_mask=head_mask,
                                             inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs, scc_layer_index  # (loss), logits, (hidden_states), (attentions)


class BertForMultipleChoice(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                start_positions=None, end_positions=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
