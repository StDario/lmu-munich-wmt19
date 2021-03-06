# Copyright 2017, 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Code for inference/translation
"""
import itertools
import json
import logging
import os
import time
from collections import defaultdict
from functools import lru_cache, partial
from typing import Callable, Dict, Generator, List, NamedTuple, Optional, Tuple, Union, Set

import mxnet as mx
import numpy as np

from . import constants as C
from . import data_io
from . import lexical_constraints as constrained
from . import lexicon
from . import model
from . import utils
from . import vocab
from .log import is_python34
from .transformer import TransformerFeedForward, TransformerFeedForwardTwoAct

logger = logging.getLogger(__name__)


class InferenceModel(model.SockeyeModel):
    """
    InferenceModel is a SockeyeModel that supports three operations used for inference/decoding:

    (1) Encoder forward call: encode source sentence and return initial decoder states.
    (2) Decoder forward call: single decoder step: predict next word.

    :param config: Configuration object holding details about the model.
    :param params_fname: File with model parameters.
    :param context: MXNet context to bind modules to.
    :param beam_size: Beam size.
    :param batch_size: Batch size.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations as safety margin for maximum output length.
    :param decoder_return_logit_inputs: Decoder returns inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Cache weights and biases for logit computation.
    :param skip_softmax: If True, does not compute softmax for greedy decoding.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 params_fname: str,
                 context: mx.context.Context,
                 beam_size: int,
                 batch_size: int,
                 softmax_temperature: Optional[float] = None,
                 max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                 decoder_return_logit_inputs: bool = False,
                 cache_output_layer_w_b: bool = False,
                 forced_max_output_len: Optional[int] = None,
                 skip_softmax: bool = False) -> None:
        super().__init__(config)
        self.params_fname = params_fname
        self.context = context
        self.beam_size = beam_size
        utils.check_condition(beam_size < self.config.vocab_target_size,
                              'The beam size must be smaller than the target vocabulary size.')
        if skip_softmax:
            assert beam_size == 1, 'Skipping softmax does not have any effect for beam size > 1'
        self.batch_size = batch_size
        self.softmax_temperature = softmax_temperature
        self.max_input_length, self.get_max_output_length = models_max_input_output_length([self],
                                                                                           max_output_length_num_stds,
                                                                                           forced_max_output_len=forced_max_output_len)

        self.model_type = config.config_decoder.model_type

        self.skip_softmax = skip_softmax

        self.encoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.encoder_default_bucket_key = None  # type: Optional[int]
        self.decoder_module = None  # type: Optional[mx.mod.BucketingModule]
        self.decoder_default_bucket_key = None  # type: Optional[Tuple[int, int]]
        self.decoder_return_logit_inputs = decoder_return_logit_inputs

        self.cache_output_layer_w_b = cache_output_layer_w_b
        self.output_layer_w = None  # type: Optional[mx.nd.NDArray]
        self.output_layer_b = None  # type: Optional[mx.nd.NDArray]

        self.avg_emb_ff = TransformerFeedForward(
            num_hidden=self.config.config_encoder.feed_forward_num_hidden,
            num_model=self.config.config_encoder.model_size,
            act_type=self.config.config_encoder.avg_act_type,
            dropout=self.config.config_encoder.avg_dropout,
            prefix="avg_emb_ff")


    @property
    def num_source_factors(self) -> int:
        """
        Returns the number of source factors of this InferenceModel (at least 1).
        """
        return self.config.config_data.num_source_factors

    def initialize(self, max_input_length: int, max_ctx_input_length: int, max_doc_input_length: int, get_max_output_length_function: Callable):
        """
        Delayed construction of modules to ensure multiple Inference models can agree on computing a common
        maximum output length.

        :param max_input_length: Maximum input length.
        :param get_max_output_length_function: Callable to compute maximum output length.
        """
        self.max_input_length = max_input_length
        self.max_ctx_input_length = max_ctx_input_length
        self.max_doc_input_length = max_doc_input_length
        if self.max_input_length > self.training_max_seq_len_source:
            logger.warning("Model was only trained with sentences up to a length of %d, "
                           "but a max_input_len of %d is used.",
                           self.training_max_seq_len_source, self.max_input_length)
        self.get_max_output_length = get_max_output_length_function

        # check the maximum supported length of the encoder & decoder:
        if self.max_supported_seq_len_source is not None:
            utils.check_condition(self.max_input_length <= self.max_supported_seq_len_source,
                                  "Encoder only supports a maximum length of %d" % self.max_supported_seq_len_source)
        if self.max_supported_seq_len_target is not None:
            decoder_max_len = self.get_max_output_length(max_input_length)
            utils.check_condition(decoder_max_len <= self.max_supported_seq_len_target,
                                  "Decoder only supports a maximum length of %d, but %d was requested. Note that the "
                                  "maximum output length depends on the input length and the source/target length "
                                  "ratio observed during training." % (self.max_supported_seq_len_target,
                                                                       decoder_max_len))

        self.encoder_module, self.encoder_default_bucket_key = self._get_encoder_module()
        self.decoder_module, self.decoder_default_bucket_key = self._get_decoder_module()

        max_encoder_data_shapes = self._get_encoder_data_shapes(self.encoder_default_bucket_key)
        max_decoder_data_shapes = self._get_decoder_data_shapes(self.decoder_default_bucket_key)
        self.encoder_module.bind(data_shapes=max_encoder_data_shapes, for_training=False, grad_req="null")
        self.decoder_module.bind(data_shapes=max_decoder_data_shapes, for_training=False, grad_req="null")

        self.load_params_from_file(self.params_fname)
        self.encoder_module.init_params(arg_params=self.params, aux_params=self.aux_params, allow_missing=False)
        self.decoder_module.init_params(arg_params=self.params, aux_params=self.aux_params, allow_missing=False)

        if self.cache_output_layer_w_b:
            if self.output_layer.weight_normalization:
                # precompute normalized output layer weight imperatively
                assert self.output_layer.weight_norm is not None
                weight = self.params[self.output_layer.weight_norm.weight.name].as_in_context(self.context)
                scale = self.params[self.output_layer.weight_norm.scale.name].as_in_context(self.context)
                self.output_layer_w = self.output_layer.weight_norm(weight, scale)
            else:
                self.output_layer_w = self.params[self.output_layer.w.name].as_in_context(self.context)
            self.output_layer_b = self.params[self.output_layer.b.name].as_in_context(self.context)

    def _get_encoder_module(self) -> Tuple[mx.mod.BucketingModule, Tuple[int, int, int]]:
        """
        Returns a BucketingModule for the encoder. Given a source sequence, it returns
        the initial decoder states of the model.
        The bucket key for this module is the length of the source sequence.

        :return: Tuple of encoder module and default bucket key.
        """

        def sym_gen(bucket_key: Tuple[int, int, int]):
            source = mx.sym.Variable(C.SOURCE_NAME)
            # source_words = source.split(num_outputs=self.num_source_factors, axis=2, squeeze_axis=True)[0]
            source_length = utils.compute_lengths(source)
            ctx_source = mx.sym.Variable(C.CTX_SOURCE_NAME)
            ctx_source_length = utils.compute_lengths(ctx_source)
            doc_source = mx.sym.Variable(C.DOC_SOURCE_NAME)
            doc_source_length = utils.compute_lengths(doc_source)
            doc_source_pad_indicator = doc_source != C.PAD_ID
            doc_source_pad_indicator = mx.sym.expand_dims(doc_source_pad_indicator, axis=2)
            source_seq_len, ctx_source_seq_len, doc_source_seq_len = bucket_key

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            (ctx_source_embed,
             ctx_source_embed_length,
             ctx_source_embed_seq_len) = self.ctx_embedding_source.encode(ctx_source, ctx_source_length,
                                                                          ctx_source_seq_len)

            (doc_source_embed,
             doc_source_embed_length,
             doc_source_embed_seq_len) = self.ctx_embedding_source.encode(doc_source, doc_source_length,
                                                                          doc_source_seq_len)

            if self.config.config_encoder.model_type == "ctx_dec":
                doc_source_embed = mx.sym.mean(doc_source_embed, axis=1, keepdims=True)

                # doc_source_embed = mx.sym.broadcast_mul(doc_source_embed, doc_source_pad_indicator)
                # ctx_sum = mx.sym.sum(doc_source_embed, axis=1, keepdims=True)
                # doc_source_embed = mx.sym.broadcast_div(ctx_sum, doc_source_length.expand_dims(axis=1).expand_dims(axis=2))

                doc_source_embed = self.avg_emb_ff(doc_source_embed)
                source_embed = mx.sym.broadcast_add(source_embed, doc_source_embed)


            # encoder
            # source_encoded: (source_encoded_length, batch_size, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len,
             ctx_source_encoded,
             ctx_source_encoded_length,
             ctx_source_encoded_seq_len) = self.encoder.encode_with_ctx(source_embed,
                                                                        source_embed_length,
                                                                        source_embed_seq_len,
                                                                        ctx_source_embed,
                                                                        ctx_source_embed_length,
                                                                        ctx_source_embed_seq_len)

            # initial decoder states
            decoder_init_states = self.decoder.init_states(source_encoded,
                                                           source_encoded_length,
                                                           source_encoded_seq_len,
                                                           ctx_source_encoded,
                                                           ctx_source_encoded_length)

            if not self.config.config_encoder.model_type == "ctx_dec":
                data_names = [C.SOURCE_NAME, C.CTX_SOURCE_NAME]
            else:
                data_names = [C.SOURCE_NAME, C.CTX_SOURCE_NAME, C.DOC_SOURCE_NAME]
            label_names = []  # type: List[str]
            return mx.sym.Group(decoder_init_states), data_names, label_names

        default_bucket_key = (self.max_input_length, self.max_ctx_input_length, self.max_doc_input_length)
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

    def _get_decoder_module(self) -> Tuple[mx.mod.BucketingModule, Tuple[int, int, int]]:
        """
        Returns a BucketingModule for a single decoder step.
        Given previously predicted word and previous decoder states, it returns
        a distribution over the next predicted word and the next decoder states.
        The bucket key for this module is the length of the source sequence
        and the current time-step in the inference procedure (e.g. beam search).
        The latter corresponds to the current length of the target sequences.

        :return: Tuple of decoder module and default bucket key.
        """

        def sym_gen(bucket_key: Tuple[int, int, int]):
            """
            Returns either softmax output (probs over target vocabulary) or inputs to logit
            computation, controlled by decoder_return_logit_inputs
            """

            source_seq_len, ctx_source_seq_len, decode_step = bucket_key
            source_embed_seq_len = self.embedding_source.get_encoded_seq_len(source_seq_len)
            source_encoded_seq_len = self.encoder.get_encoded_seq_len(source_embed_seq_len)
            ctx_source_embed_seq_len = self.ctx_embedding_source.get_encoded_seq_len(ctx_source_seq_len)
            ctx_source_encoded_seq_len = self.encoder.get_encoded_seq_len(ctx_source_embed_seq_len)

            self.decoder.reset()
            target_prev = mx.sym.Variable(C.TARGET_NAME)
            states = self.decoder.state_variables(decode_step)
            state_names = [state.name for state in states]

            # embedding for previous word
            # (batch_size, num_embed)
            target_embed_prev, _, _ = self.embedding_target.encode(data=target_prev, data_length=None, seq_len=1)


            # decoder
            # target_decoded: (batch_size, decoder_depth)
            (target_decoded,
             attention_probs,
             states) = self.decoder.decode_step(decode_step,
                                                target_embed_prev,
                                                source_encoded_seq_len,
                                                ctx_source_encoded_seq_len,
                                                *states)

            if self.decoder_return_logit_inputs:
                # skip output layer in graph
                outputs = mx.sym.identity(target_decoded, name=C.LOGIT_INPUTS_NAME)
            else:
                # logits: (batch_size, target_vocab_size)
                logits = self.output_layer(target_decoded)
                if self.softmax_temperature is not None:
                    logits = logits / self.softmax_temperature
                if self.skip_softmax:
                    # skip softmax for greedy decoding
                    outputs = logits
                else:
                    outputs = mx.sym.softmax(data=logits, name=C.SOFTMAX_NAME)

            data_names = [C.TARGET_NAME] + state_names
            label_names = []  # type: List[str]
            return mx.sym.Group([outputs, attention_probs] + states), data_names, label_names

        # pylint: disable=not-callable

        max_input_len = self.max_input_length
        if self.config.config_encoder.model_type == C.MODEL_TYPE_AVG_EMB_TOK or self.config.config_encoder.model_type == C.MODEL_TYPE_MAX_EMB_TOK:
            max_input_len += 1

        default_bucket_key = (max_input_len, self.max_ctx_input_length, self.get_max_output_length(self.max_input_length))
        module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                        default_bucket_key=default_bucket_key,
                                        context=self.context)
        return module, default_bucket_key

    def _get_encoder_data_shapes(self, bucket_key: Tuple[int, int, int]) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the encoder module.

        :param bucket_key: Maximum input length.
        :return: List of data descriptions.
        """

        if not self.config.config_encoder.model_type == "ctx_dec":
            return [mx.io.DataDesc(name=C.SOURCE_NAME,
                                   shape=(self.batch_size, bucket_key[0]),
                                   layout=C.BATCH_MAJOR),
                    mx.io.DataDesc(name='ctx_' + C.SOURCE_NAME,
                                   shape=(self.batch_size, bucket_key[1]),
                                   layout=C.BATCH_MAJOR)
                    ]
        else:
            return [mx.io.DataDesc(name=C.SOURCE_NAME,
                                   shape=(self.batch_size, bucket_key[0]),
                                   layout=C.BATCH_MAJOR),
                    mx.io.DataDesc(name='ctx_' + C.SOURCE_NAME,
                                   shape=(self.batch_size, bucket_key[1]),
                                   layout=C.BATCH_MAJOR),
                    mx.io.DataDesc(name='doc_' + C.SOURCE_NAME,
                                   shape=(self.batch_size, bucket_key[2]),
                                   layout=C.BATCH_MAJOR)
                    ]

    @lru_cache(maxsize=None)
    def _get_decoder_data_shapes(self, bucket_key: Tuple[int, int, int]) -> List[mx.io.DataDesc]:
        """
        Returns data shapes of the decoder module.

        :param bucket_key: Tuple of (maximum input length, maximum target length).
        :return: List of data descriptions.
        """

        source_max_length, ctx_source_max_length, target_max_length = bucket_key

        if self.model_type == "ctx_enc":
            ctx_source_max_length = source_max_length

        # if self.model_type == C.MODEL_TYPE_AVG_EMB_TOK:
        #     source_max_length -= 1

        return [mx.io.DataDesc(name=C.TARGET_NAME, shape=(self.batch_size * self.beam_size,),
                               layout="NT")] + self.decoder.state_shapes(self.batch_size * self.beam_size,
                                                                         target_max_length,
                                                                         self.encoder.get_encoded_seq_len(
                                                                             source_max_length),
                                                                         self.encoder.get_num_hidden(),
                                                                         self.encoder.get_encoded_seq_len(
                                                                             ctx_source_max_length),
                                                                         self.encoder.get_num_hidden()
                                                                         )

    def run_encoder(self,
                    source: mx.nd.NDArray,
                    ctx_source: mx.nd.NDArray,
                    doc_source: mx.nd.NDArray,
                    source_max_length: int, ctx_source_max_length: int, doc_source_max_length: int) -> 'ModelState':
        """
        Runs forward pass of the encoder.
        Encodes source given source length and bucket key.
        Returns encoder representation of the source, source_length, initial hidden state of decoder RNN,
        and initial decoder states tiled to beam size.

        :param source: Integer-coded input tokens. Shape (batch_size, source length, num_source_factors).
        :param source_max_length: Bucket key.
        :return: Initial model state.
        """
        batch = mx.io.DataBatch(data=[source, ctx_source, doc_source],
                                label=None,
                                bucket_key=(source_max_length, ctx_source_max_length, doc_source_max_length),
                                provide_data=self._get_encoder_data_shapes((source_max_length, ctx_source_max_length, doc_source_max_length)))

        self.encoder_module.forward(data_batch=batch, is_train=False)
        decoder_states = self.encoder_module.get_outputs()

        # replicate encoder/init module results beam size times
        decoder_states = [mx.nd.repeat(s, repeats=self.beam_size, axis=0) for s in decoder_states]
        return ModelState(decoder_states)

    def run_decoder(self,
                    prev_word: mx.nd.NDArray,
                    bucket_key: Tuple[int, int, int],
                    model_state: 'ModelState') -> Tuple[mx.nd.NDArray, mx.nd.NDArray, 'ModelState']:
        """
        Runs forward pass of the single-step decoder.

        :return: Decoder stack output (logit inputs or probability distribution), attention scores, updated model state.
        """
        batch = mx.io.DataBatch(
            data=[prev_word.as_in_context(self.context)] + model_state.states,
            label=None,
            bucket_key=bucket_key,
            provide_data=self._get_decoder_data_shapes(bucket_key))
        self.decoder_module.forward(data_batch=batch, is_train=False)
        out, attention_probs, *model_state.states = self.decoder_module.get_outputs()
        return out, attention_probs, model_state

    @property
    def training_max_seq_len_source(self) -> int:
        """ The maximum sequence length on the source side during training. """
        return self.config.config_data.data_statistics.max_observed_len_source

    @property
    def training_max_seq_len_target(self) -> int:
        """ The maximum sequence length on the target side during training. """
        return self.config.config_data.data_statistics.max_observed_len_target

    @property
    def max_supported_seq_len_source(self) -> Optional[int]:
        """ If not None this is the maximally supported source length during inference (hard constraint). """
        return self.encoder.get_max_seq_len()

    @property
    def max_supported_seq_len_target(self) -> Optional[int]:
        """ If not None this is the maximally supported target length during inference (hard constraint). """
        return self.decoder.get_max_seq_len()

    @property
    def length_ratio_mean(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_mean

    @property
    def length_ratio_std(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_std

    @property
    def source_with_eos(self) -> bool:
        return self.config.config_data.source_with_eos


def load_models(context: mx.context.Context,
                max_input_len: Optional[int],
                max_ctx_input_len: Optional[int],
                max_doc_input_len: Optional[int],
                beam_size: int,
                batch_size: int,
                model_folders: List[str],
                checkpoints: Optional[List[int]] = None,
                softmax_temperature: Optional[float] = None,
                max_output_length_num_stds: int = C.DEFAULT_NUM_STD_MAX_OUTPUT_LENGTH,
                decoder_return_logit_inputs: bool = False,
                cache_output_layer_w_b: bool = False,
                forced_max_output_len: Optional[int] = None,
                override_dtype: Optional[str] = None) -> Tuple[List[InferenceModel],
                                                               List[vocab.Vocab],
                                                               vocab.Vocab]:
    """
    Loads a list of models for inference.

    :param context: MXNet context to bind modules to.
    :param max_input_len: Maximum input length.
    :param beam_size: Beam size.
    :param batch_size: Batch size.
    :param model_folders: List of model folders to load models from.
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :param softmax_temperature: Optional parameter to control steepness of softmax distribution.
    :param max_output_length_num_stds: Number of standard deviations to add to mean target-source length ratio
           to compute maximum output length.
    :param decoder_return_logit_inputs: Model decoders return inputs to logit computation instead of softmax over target
                                        vocabulary.  Used when logits/softmax are handled separately.
    :param cache_output_layer_w_b: Models cache weights and biases for logit computation as NumPy arrays (used with
                                   restrict lexicon).
    :param forced_max_output_len: An optional overwrite of the maximum output length.
    :param override_dtype: Overrides dtype of encoder and decoder defined at training time to a different one.
    :return: List of models, source vocabulary, target vocabulary, source factor vocabularies.
    """
    logger.info("Loading %d model(s) from %s ...", len(model_folders), model_folders)
    load_time_start = time.time()
    models = []  # type: List[InferenceModel]
    source_vocabs = []  # type: List[List[vocab.Vocab]]
    target_vocabs = []  # type: List[vocab.Vocab]

    if checkpoints is None:
        checkpoints = [None] * len(model_folders)

    # skip softmax for a single model,
    if len(model_folders) == 1 and beam_size == 1:
        skip_softmax = True
        logger.info("Enabled skipping softmax for a single model and greedy decoding.")
    else:
        # but not for an ensemble or beam search
        skip_softmax = False

    for model_folder, checkpoint in zip(model_folders, checkpoints):
        model_source_vocabs = vocab.load_source_vocabs(model_folder)
        model_target_vocab = vocab.load_target_vocab(model_folder)
        source_vocabs.append(model_source_vocabs)
        target_vocabs.append(model_target_vocab)

        model_version = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
        logger.info("Model version: %s", model_version)
        utils.check_version(model_version)
        model_config = model.SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))
        if override_dtype is not None:
            model_config.config_encoder.dtype = override_dtype
            model_config.config_decoder.dtype = override_dtype

        if not hasattr(model_config.config_encoder, 'ctx_attention_heads'):
            model_config.config_encoder.ctx_attention_heads = model_config.config_encoder.attention_heads
        if not hasattr(model_config.config_encoder, 'ctx_model_size'):
            model_config.config_encoder.ctx_model_size = model_config.config_encoder.model_size
        if not hasattr(model_config.config_decoder, 'ctx_attention_heads'):
            model_config.config_decoder.ctx_attention_heads = model_config.config_decoder.attention_heads
        if not hasattr(model_config.config_decoder, 'ctx_model_size'):
            model_config.config_decoder.ctx_model_size = model_config.config_decoder.model_size

        if checkpoint is None:
            params_fname = os.path.join(model_folder, C.PARAMS_BEST_NAME)
        else:
            params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint)

        inference_model = InferenceModel(config=model_config,
                                         params_fname=params_fname,
                                         context=context,
                                         beam_size=beam_size,
                                         batch_size=batch_size,
                                         softmax_temperature=softmax_temperature,
                                         decoder_return_logit_inputs=decoder_return_logit_inputs,
                                         cache_output_layer_w_b=cache_output_layer_w_b,
                                         skip_softmax=skip_softmax)
        utils.check_condition(inference_model.num_source_factors == len(model_source_vocabs),
                              "Number of loaded source vocabularies (%d) does not match "
                              "number of source factors for model '%s' (%d)" % (len(model_source_vocabs), model_folder,
                                                                                inference_model.num_source_factors))
        models.append(inference_model)

    utils.check_condition(vocab.are_identical(*target_vocabs), "Target vocabulary ids do not match")
    first_model_vocabs = source_vocabs[0]
    for fi in range(len(first_model_vocabs)):
        utils.check_condition(vocab.are_identical(*[source_vocabs[i][fi] for i in range(len(source_vocabs))]),
                              "Source vocabulary ids do not match. Factor %d" % fi)

    source_with_eos = models[0].source_with_eos
    utils.check_condition(all(source_with_eos == m.source_with_eos for m in models),
                          "All models must agree on using source-side EOS symbols or not. "
                          "Did you try combining models trained with different versions?")

    if max_ctx_input_len is None:
        max_ctx_input_len = max_input_len

    # set a common max_output length for all models.
    max_input_len, get_max_output_length = models_max_input_output_length(models,
                                                                          max_output_length_num_stds,
                                                                          max_input_len,
                                                                          forced_max_output_len=forced_max_output_len)

    if model_config.config_encoder.model_type == C.MODEL_TYPE_AVG_EMB_TOK or model_config.config_encoder.model_type == C.MODEL_TYPE_MAX_EMB_TOK:
        max_input_len -= 1

    for inference_model in models:
        inference_model.initialize(max_input_len, max_ctx_input_len, max_doc_input_len, get_max_output_length)

    load_time = time.time() - load_time_start
    logger.info("%d model(s) loaded in %.4fs", len(models), load_time)
    return models, source_vocabs[0], target_vocabs[0]


def models_max_input_output_length(models: List[InferenceModel],
                                   num_stds: int,
                                   forced_max_input_len: Optional[int] = None,
                                   forced_max_output_len: Optional[int] = None) -> Tuple[int, Callable]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length.
    Mean and std are taken from the model with the largest values to allow proper ensembling of models
    trained on different data sets.

    :param models: List of models.
    :param num_stds: Number of standard deviations to add as a safety margin. If -1, returned maximum output lengths
                     will always be 2 * input_length.
    :param forced_max_input_len: An optional overwrite of the maximum input length.
    :param forced_max_output_len: An optional overwrite of the maximum output length.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    max_mean = max(model.length_ratio_mean for model in models)
    max_std = max(model.length_ratio_std for model in models)

    supported_max_seq_len_source = min((model.max_supported_seq_len_source for model in models
                                        if model.max_supported_seq_len_source is not None),
                                       default=None)
    supported_max_seq_len_target = min((model.max_supported_seq_len_target for model in models
                                        if model.max_supported_seq_len_target is not None),
                                       default=None)
    training_max_seq_len_source = min(model.training_max_seq_len_source for model in models)

    return get_max_input_output_length(supported_max_seq_len_source,
                                       supported_max_seq_len_target,
                                       training_max_seq_len_source,
                                       length_ratio_mean=max_mean,
                                       length_ratio_std=max_std,
                                       num_stds=num_stds,
                                       forced_max_input_len=forced_max_input_len,
                                       forced_max_output_len=forced_max_output_len)


def get_max_input_output_length(supported_max_seq_len_source: Optional[int],
                                supported_max_seq_len_target: Optional[int],
                                training_max_seq_len_source: Optional[int],
                                length_ratio_mean: float,
                                length_ratio_std: float,
                                num_stds: int,
                                forced_max_input_len: Optional[int] = None,
                                forced_max_output_len: Optional[int] = None) -> Tuple[int, Callable]:
    """
    Returns a function to compute maximum output length given a fixed number of standard deviations as a
    safety margin, and the current input length. It takes into account optional maximum source and target lengths.

    :param supported_max_seq_len_source: The maximum source length supported by the models.
    :param supported_max_seq_len_target: The maximum target length supported by the models.
    :param training_max_seq_len_source: The maximum source length observed during training.
    :param length_ratio_mean: The mean of the length ratio that was calculated on the raw sequences with special
           symbols such as EOS or BOS.
    :param length_ratio_std: The standard deviation of the length ratio.
    :param num_stds: The number of standard deviations the target length may exceed the mean target length (as long as
           the supported maximum length allows for this).
    :param forced_max_input_len: An optional overwrite of the maximum input length.
    :param forced_max_output_len: An optional overwrite of the maximum out length.
    :return: The maximum input length and a function to get the output length given the input length.
    """
    space_for_bos = 1
    space_for_eos = 1

    if num_stds < 0:
        factor = C.TARGET_MAX_LENGTH_FACTOR  # type: float
    else:
        factor = length_ratio_mean + (length_ratio_std * num_stds)

    if forced_max_input_len is None:
        # Make sure that if there is a hard constraint on the maximum source or target length we never exceed this
        # constraint. This is for example the case for learned positional embeddings, which are only defined for the
        # maximum source and target sequence length observed during training.
        if supported_max_seq_len_source is not None and supported_max_seq_len_target is None:
            max_input_len = supported_max_seq_len_source
        elif supported_max_seq_len_source is None and supported_max_seq_len_target is not None:
            max_output_len = supported_max_seq_len_target - space_for_bos - space_for_eos
            if np.ceil(factor * training_max_seq_len_source) > max_output_len:
                max_input_len = int(np.floor(max_output_len / factor))
            else:
                max_input_len = training_max_seq_len_source
        elif supported_max_seq_len_source is not None or supported_max_seq_len_target is not None:
            max_output_len = supported_max_seq_len_target - space_for_bos - space_for_eos
            if np.ceil(factor * supported_max_seq_len_source) > max_output_len:
                max_input_len = int(np.floor(max_output_len / factor))
            else:
                max_input_len = supported_max_seq_len_source
        else:
            # Any source/target length is supported and max_input_len was not manually set, therefore we use the
            # maximum length from training.
            max_input_len = training_max_seq_len_source
    else:
        max_input_len = forced_max_input_len

    def get_max_output_length(input_length: int):
        """
        Returns the maximum output length for inference given the input length.
        Explicitly includes space for BOS and EOS sentence symbols in the target sequence, because we assume
        that the mean length ratio computed on the training data do not include these special symbols.
        (see data_io.analyze_sequence_lengths)
        """
        if forced_max_output_len is not None:
            return forced_max_output_len
        else:
            return int(np.ceil(factor * input_length)) + space_for_bos + space_for_eos

    return max_input_len, get_max_output_length


BeamHistory = Dict[str, List]
Tokens = List[str]
SentenceId = Union[int, str]


class TranslatorInput:
    """
    Object required by Translator.translate().

    :param sentence_id: Sentence id.
    :param tokens: List of input tokens.
    :param factors: Optional list of additional factor sequences.
    :param constraints: Optional list of target-side constraints.
    """

    __slots__ = ('sentence_id', 'tokens', 'ctx_tokens', 'doc_tokens', 'factors', 'constraints', 'avoid_list')

    def __init__(self,
                 sentence_id: SentenceId,
                 tokens: Tokens,
                 ctx_tokens: Tokens,
                 doc_tokens: Tokens,
                 factors: Optional[List[Tokens]] = None,
                 constraints: Optional[List[Tokens]] = None,
                 avoid_list: Optional[List[Tokens]] = None) -> None:
        self.sentence_id = sentence_id
        self.tokens = tokens
        self.ctx_tokens = ctx_tokens
        self.doc_tokens = doc_tokens
        self.factors = factors
        self.constraints = constraints
        self.avoid_list = avoid_list

    def __str__(self):
        return 'TranslatorInput(%s, %s, %s %s, factors=%s, constraints=%s, avoid=%s)' \
            % (self.sentence_id, self.tokens, self.ctx_tokens, self.doc_tokens, self.factors, self.constraints, self.avoid_list)

    def __len__(self):
        return len(self.tokens)

    @property
    def num_factors(self) -> int:
        """
        Returns the number of factors of this instance.
        """
        return 1 + (0 if not self.factors else len(self.factors))

    def chunks(self, chunk_size: int) -> Generator['TranslatorInput', None, None]:
        """
        Takes a TranslatorInput (itself) and yields TranslatorInputs for chunks of size chunk_size.

        :param chunk_size: The maximum size of a chunk.
        :return: A generator of TranslatorInputs, one for each chunk created.
        """

        if len(self.tokens) > chunk_size and self.constraints is not None:
            logger.warning(
                'Input %s has length (%d) that exceeds max input length (%d), '
                'triggering internal splitting. Placing all target-side constraints '
                'with the first chunk, which is probably wrong.',
                self.sentence_id, len(self.tokens), chunk_size)

        for chunk_id, i in enumerate(range(0, len(self), chunk_size)):
            factors = [factor[i:i + chunk_size] for factor in self.factors] if self.factors is not None else None
            # Constrained decoding is not supported for chunked TranslatorInputs. As a fall-back, constraints are
            # assigned to the first chunk
            constraints = self.constraints if chunk_id == 0 else None
            yield TranslatorInput(sentence_id=self.sentence_id,
                                  tokens=self.tokens[i:i + chunk_size],
                                  ctx_tokens=[C.CTX_SYMBOL] + self.ctx_tokens,
                                  doc_tokens=[C.CTX_SYMBOL] + self.doc_tokens,
                                  factors=factors,
                                  constraints=constraints,
                                  avoid_list=self.avoid_list)

    def with_eos(self) -> 'TranslatorInput':
        """
        :return: A new translator input with EOS appended to the tokens and factors.
        """
        return TranslatorInput(sentence_id=self.sentence_id,
                               tokens=self.tokens + [C.EOS_SYMBOL],
                               ctx_tokens=[C.CTX_SYMBOL] + self.ctx_tokens,
                               doc_tokens=[C.CTX_SYMBOL] + self.doc_tokens,
                               factors=[factor + [C.EOS_SYMBOL] for factor in
                                        self.factors] if self.factors is not None else None,
                               constraints=self.constraints,
                               avoid_list=self.avoid_list)


class BadTranslatorInput(TranslatorInput):

    def __init__(self, sentence_id: SentenceId, tokens: Tokens) -> None:
        super().__init__(sentence_id=sentence_id, tokens=tokens, factors=None)


def _bad_input(sentence_id: SentenceId, reason: str = '') -> BadTranslatorInput:
    logger.warning("Bad input (%s): '%s'. Will return empty output.", sentence_id, reason.strip())
    return BadTranslatorInput(sentence_id=sentence_id, tokens=[])


def make_input_from_plain_string(sentence_id: SentenceId, string: str) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a plain string.

    :param sentence_id: Sentence id.
    :param string: An input string.
    :return: A TranslatorInput.
    """

    tokens, ctx_tokens = utils.get_ctx_tokens(string)

    return TranslatorInput(sentence_id, tokens=tokens, ctx_tokens=ctx_tokens, factors=None)
    # return TranslatorInput(sentence_id, tokens=list(data_io.get_tokens(string)), factors=None)


def make_input_from_json_string(sentence_id: SentenceId, json_string: str) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a JSON object, serialized as a string.

    :param sentence_id: Sentence id.
    :param json_string: A JSON object serialized as a string that must contain a key "text", mapping to the input text,
           and optionally a key "factors" that maps to a list of strings, each of which representing a factor sequence
           for the input text.
    :return: A TranslatorInput.
    """
    try:
        jobj = json.loads(json_string, encoding=C.JSON_ENCODING)
        tokens = jobj[C.JSON_TEXT_KEY]
        tokens = list(data_io.get_tokens(tokens))
        factors = jobj.get(C.JSON_FACTORS_KEY)
        if isinstance(factors, list):
            factors = [list(data_io.get_tokens(factor)) for factor in factors]
            lengths = [len(f) for f in factors]
            if not all(length == len(tokens) for length in lengths):
                logger.error("Factors have different length than input text: %d vs. %s", len(tokens), str(lengths))
                return _bad_input(sentence_id, reason=json_string)

        # List of phrases to prevent from occuring in the output
        avoid_list = jobj.get(C.JSON_AVOID_KEY)

        # List of phrases that must appear in the output
        constraints = jobj.get(C.JSON_CONSTRAINTS_KEY)

        # If there is overlap between positive and negative constraints, assume the user wanted
        # the words, and so remove them from the avoid_list (negative constraints)
        if constraints is not None and avoid_list is not None:
            avoid_set = set(avoid_list)
            overlap = set(constraints).intersection(avoid_set)
            if len(overlap) > 0:
                avoid_list = list(avoid_set.difference(overlap))

        # Convert to a list of tokens
        if isinstance(avoid_list, list):
            avoid_list = [list(data_io.get_tokens(phrase)) for phrase in avoid_list]
        if isinstance(constraints, list):
            constraints = [list(data_io.get_tokens(constraint)) for constraint in constraints]

        return TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors, constraints=constraints, avoid_list=avoid_list)

    except Exception as e:
        logger.exception(e, exc_info=True) if not is_python34() else logger.error(e)  # type: ignore
        return _bad_input(sentence_id, reason=json_string)


def make_input_from_factored_string(sentence_id: SentenceId,
                                    factored_string: str,
                                    translator: 'Translator',
                                    delimiter: str = C.DEFAULT_FACTOR_DELIMITER) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a string with factor annotations on a token level, separated by delimiter.
    If translator does not require any source factors, the string is parsed as a plain token string.

    :param sentence_id: Sentence id.
    :param factored_string: An input string with additional factors per token, separated by delimiter.
    :param translator: A translator object.
    :param delimiter: A factor delimiter. Default: '|'.
    :return: A TranslatorInput.
    """
    utils.check_condition(bool(delimiter) and not delimiter.isspace(),
                          "Factor delimiter can not be whitespace or empty.")

    model_num_source_factors = translator.num_source_factors

    if model_num_source_factors == 1:
        return make_input_from_plain_string(sentence_id=sentence_id, string=factored_string)

    tokens = []  # type: Tokens
    factors = [[] for _ in range(model_num_source_factors - 1)]  # type: List[Tokens]
    for token_id, token in enumerate(data_io.get_tokens(factored_string)):
        pieces = token.split(delimiter)

        if not all(pieces) or len(pieces) != model_num_source_factors:
            logger.error("Failed to parse %d factors at position %d ('%s') in '%s'" % (model_num_source_factors,
                                                                                       token_id, token,
                                                                                       factored_string.strip()))
            return _bad_input(sentence_id, reason=factored_string)

        tokens.append(pieces[0])
        for i, factor in enumerate(factors):
            factors[i].append(pieces[i + 1])

    return TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors)

def make_input_from_factored_string_ctx_doc(sentence_id: SentenceId,
                                    factored_string: str,
                                    ctx_factored_string: str,
                                    doc_factored_string: str,
                                    translator: 'Translator',
                                    delimiter: str = C.DEFAULT_FACTOR_DELIMITER) -> TranslatorInput:
    """
    Returns a TranslatorInput object from a string with factor annotations on a token level, separated by delimiter.
    If translator does not require any source factors, the string is parsed as a plain token string.

    :param sentence_id: Sentence id.
    :param factored_string: An input string with additional factors per token, separated by delimiter.
    :param translator: A translator object.
    :param delimiter: A factor delimiter. Default: '|'.
    :return: A TranslatorInput.
    """
    utils.check_condition(bool(delimiter) and not delimiter.isspace(),
                          "Factor delimiter can not be whitespace or empty.")

    # logger.info("Ignoring factors if any")

    tokens = utils.get_tokens(factored_string)
    ctx_tokens = utils.get_tokens(ctx_factored_string)
    doc_tokens = utils.get_tokens(doc_factored_string)

    return TranslatorInput(sentence_id, tokens=list(tokens), ctx_tokens=list(ctx_tokens), doc_tokens=list(doc_tokens), factors=None)


def make_input_from_multiple_strings(sentence_id: SentenceId, strings: List[str]) -> TranslatorInput:
    """
    Returns a TranslatorInput object from multiple strings, where the first element corresponds to the surface tokens
    and the remaining elements to additional factors. All strings must parse into token sequences of the same length.

    :param sentence_id: Sentence id.
    :param strings: A list of strings representing a factored input sequence.
    :return: A TranslatorInput.
    """
    if not bool(strings):
        return TranslatorInput(sentence_id=sentence_id, tokens=[], factors=None)

    tokens = list(data_io.get_tokens(strings[0]))
    factors = [list(data_io.get_tokens(factor)) for factor in strings[1:]]
    if not all(len(factor) == len(tokens) for factor in factors):
        logger.error("Length of string sequences do not match: '%s'", strings)
        return _bad_input(sentence_id, reason=str(strings))
    return TranslatorInput(sentence_id=sentence_id, tokens=tokens, factors=factors)


class TranslatorOutput:
    """
    Output structure from Translator.

    :param sentence_id: Sentence id.
    :param translation: Translation string without sentence boundary tokens.
    :param tokens: List of translated tokens.
    :param attention_matrix: Attention matrix. Shape: (target_length, source_length).
    :param score: Negative log probability of generated translation.
    :param beam_histories: List of beam histories. The list will contain more than one
           history if it was split due to exceeding max_length.
    """
    __slots__ = ('sentence_id', 'translation', 'tokens', 'attention_matrix', 'score', 'scores', 'beam_histories')

    def __init__(self,
                 sentence_id: SentenceId,
                 translation: str,
                 tokens: List[str],
                 attention_matrix: np.ndarray,
                 score: float,
                 scores: Optional[List[float]] = None,
                 beam_histories: Optional[List[BeamHistory]] = None) -> None:
        self.sentence_id = sentence_id
        self.translation = translation
        self.tokens = tokens
        self.attention_matrix = attention_matrix
        self.score = score
        self.beam_histories = beam_histories
        self.scores = scores


TokenIds = List[int]


class Translation:
    __slots__ = ('target_ids', 'attention_matrix', 'score', 'beam_histories')

    def __init__(self,
                 target_ids: TokenIds,
                 attention_matrix: np.ndarray,
                 score: float,
                 beam_history: List[BeamHistory] = None) -> None:
        self.target_ids = target_ids
        self.attention_matrix = attention_matrix
        self.score = score
        self.beam_histories = beam_history if beam_history is not None else []


def empty_translation() -> Translation:
    return Translation(target_ids=[], attention_matrix=np.asarray([[0]]), score=-np.inf)


IndexedTranslatorInput = NamedTuple('IndexedTranslatorInput', [
    ('input_idx', int),
    ('chunk_idx', int),
    ('translator_input', TranslatorInput)
])
"""
Translation of a chunk of a sentence.

:param input_idx: Internal index of translation requests to keep track of the correct order of translations.
:param chunk_idx: The index of the chunk. Used when TranslatorInputs get split across multiple chunks.
:param input: The translator input.
"""


IndexedTranslation = NamedTuple('IndexedTranslation', [
    ('input_idx', int),
    ('chunk_idx', int),
    ('translation', Translation),
])
"""
Translation of a chunk of a sentence.

:param input_idx: Internal index of translation requests to keep track of the correct order of translations.
:param chunk_idx: The index of the chunk. Used when TranslatorInputs get split across multiple chunks.
:param translation: The translation of the input chunk.
"""


class ModelState:
    """
    A ModelState encapsulates information about the decoder states of an InferenceModel.
    """

    def __init__(self, states: List[mx.nd.NDArray]) -> None:
        self.states = states

    def sort_state(self, best_hyp_indices: mx.nd.NDArray):
        """
        Sorts states according to k-best order from last step in beam search.
        """
        self.states = [mx.nd.take(ds, best_hyp_indices) for ds in self.states]


class LengthPenalty(mx.gluon.HybridBlock):
    """
    Calculates the length penalty as:
    (beta + len(Y))**alpha / (beta + 1)**alpha

    See Wu et al. 2016 (note that in the paper beta has a different meaning,
    and a fixed value 5 was used for this parameter)

    :param alpha: The alpha factor for the length penalty (see above).
    :param beta: The beta factor for the length penalty (see above).
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.denominator = (self.beta + 1.) ** self.alpha

    def hybrid_forward(self, F, lengths):
        if self.alpha == 0.0:
            if F is None:
                return 1.0
            else:
                return F.ones_like(lengths)
        else:
            numerator = self.beta + lengths if self.beta != 0.0 else lengths
            numerator = numerator ** self.alpha if self.alpha != 1.0 else numerator
            return numerator / self.denominator

    def get(self, lengths: Union[mx.nd.NDArray, int, float]) -> Union[mx.nd.NDArray, float]:
        """
        Calculate the length penalty for the given vector of lengths.

        :param lengths: A scalar or a matrix of sentence lengths of dimensionality (batch_size, 1).
        :return: The length penalty. A scalar or a matrix (batch_size, 1) depending on the input.
        """
        return self.hybrid_forward(None, lengths)


def _concat_translations(translations: List[Translation], stop_ids: Set[int],
                         length_penalty: LengthPenalty) -> Translation:
    """
    Combine translations through concatenation.

    :param translations: A list of translations (sequence starting with BOS symbol, attention_matrix), score and length.
    :param translations: The EOS symbols.
    :return: A concatenation if the translations with a score.
    """
    # Concatenation of all target ids without BOS and EOS
    target_ids = []
    attention_matrices = []
    beam_histories = []  # type: List[BeamHistory]
    for idx, translation in enumerate(translations):
        if idx == len(translations) - 1:
            target_ids.extend(translation.target_ids)
            attention_matrices.append(translation.attention_matrix)
        else:
            if translation.target_ids[-1] in stop_ids:
                target_ids.extend(translation.target_ids[:-1])
                attention_matrices.append(translation.attention_matrix[:-1, :])
            else:
                target_ids.extend(translation.target_ids)
                attention_matrices.append(translation.attention_matrix)
        beam_histories.extend(translation.beam_histories)

    # Combine attention matrices:
    attention_shapes = [attention_matrix.shape for attention_matrix in attention_matrices]
    attention_matrix_combined = np.zeros(np.sum(np.asarray(attention_shapes), axis=0))
    pos_t, pos_s = 0, 0
    for attention_matrix, (len_t, len_s) in zip(attention_matrices, attention_shapes):
        attention_matrix_combined[pos_t:pos_t + len_t, pos_s:pos_s + len_s] = attention_matrix
        pos_t += len_t
        pos_s += len_s

    # Unnormalize + sum and renormalize the score:
    score = sum(translation.score * length_penalty.get(len(translation.target_ids))
                for translation in translations)
    score = score / length_penalty.get(len(target_ids))
    return Translation(target_ids, attention_matrix_combined, score, beam_histories)


class Translator:
    """
    Translator uses one or several models to translate input.
    The translator holds a reference to vocabularies to convert between word ids and text tokens for input and
    translation strings.

    :param context: MXNet context to bind modules to.
    :param ensemble_mode: Ensemble mode: linear or log_linear combination.
    :param length_penalty: Length penalty instance.
    :param beam_prune: Beam pruning difference threshold.
    :param beam_search_stop: The stopping criterium.
    :param models: List of models.
    :param source_vocabs: Source vocabularies.
    :param target_vocab: Target vocabulary.
    :param restrict_lexicon: Top-k lexicon to use for target vocabulary restriction.
    :param avoid_list: Global list of phrases to exclude from the output.
    :param store_beam: If True, store the beam search history and return it in the TranslatorOutput.
    :param strip_unknown_words: If True, removes any <unk> symbols from outputs.
    :param skip_topk: If True, uses argmax instead of topk for greedy decoding.
    """

    def __init__(self,
                 context: mx.context.Context,
                 ensemble_mode: str,
                 bucket_source_width: int,
                 length_penalty: LengthPenalty,
                 beam_prune: float,
                 beam_search_stop: str,
                 models: List[InferenceModel],
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab,
                 restrict_lexicon: Optional[lexicon.TopKLexicon] = None,
                 avoid_list: Optional[str] = None,
                 store_beam: bool = False,
                 strip_unknown_words: bool = False,
                 skip_topk: bool = False,
                 use_previous_translation: bool = False,
                 ctx_step_size: int = None,
                 doc_step_size: int = None) -> None:
        self.context = context
        self.length_penalty = length_penalty
        self.beam_prune = beam_prune
        self.beam_search_stop = beam_search_stop
        self.source_vocabs = source_vocabs
        self.vocab_target = target_vocab
        self.vocab_target_inv = vocab.reverse_vocab(self.vocab_target)
        self.restrict_lexicon = restrict_lexicon
        self.store_beam = store_beam

        self.use_previous_translation = use_previous_translation

        self.start_id = self.vocab_target[C.BOS_SYMBOL]
        assert C.PAD_ID == 0, "pad id should be 0"
        self.stop_ids = {self.vocab_target[C.EOS_SYMBOL], C.PAD_ID}  # type: Set[int]
        self.strip_ids = self.stop_ids.copy()  # ids to strip from the output
        self.unk_id = self.vocab_target[C.UNK_SYMBOL]
        if strip_unknown_words:
            self.strip_ids.add(self.unk_id)
        self.models = models
        utils.check_condition(all(models[0].source_with_eos == m.source_with_eos for m in models),
                              "The source_with_eos property must match across models.")
        self.source_with_eos = models[0].source_with_eos
        self.interpolation_func = self._get_interpolation_func(ensemble_mode)
        self.beam_size = self.models[0].beam_size
        self.batch_size = self.models[0].batch_size
        # skip softmax for a single model, but not for an ensemble
        self.skip_softmax = self.models[0].skip_softmax
        if self.skip_softmax:
            utils.check_condition(len(self.models) == 1 and self.beam_size == 1, "Skipping softmax cannot be enabled for several models, or a beam size > 1.")

        self.skip_topk = skip_topk
        # after models are loaded we ensured that they agree on max_input_length, max_output_length and batch size
        self._max_input_length = self.models[0].max_input_length
        self.max_ctx_input_length = self.models[0].max_ctx_input_length
        self.max_doc_input_length = self.models[0].max_doc_input_length

        # max_seq_len_source: int,
        # max_seq_len_target: int,
        # max_seq_len_ctx_source: int,
        # bucket_width: int = 10,
        # length_ratio: float = 1.0,
        # length_ratio_ctx: float = 1.0,
        # is_ctx_trg: bool = False,
        # num_ctx_sentences: int = 0
        # training_buckets = data_io.define_parallel_buckets(self.models[0].training_max_seq_len_source, self.models[0].training_max_seq_len_target,
        #                                                    17770, 10, 0.9, 1.4, False, 10)

        if bucket_source_width > 0:
            self.buckets_source = data_io.define_ctx_doc_buckets(self._max_input_length, step=bucket_source_width, ctx_max_seq_len=self.max_ctx_input_length, ctx_step=ctx_step_size,
                                                                 doc_max_seq_len=self.max_doc_input_length, doc_step=doc_step_size)
        else:
            self.buckets_source = [self._max_input_length]


        self.pad_dist = mx.nd.full((self.batch_size * self.beam_size, len(self.vocab_target) - 1), val=np.inf,
                                   ctx=self.context)

        self.use_doc_pool = False
        self.pool_window = 0
        self.pool_stride = 0

        # These are constants used for manipulation of the beam and scores (particularly for pruning)
        self.zeros_array = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')
        self.inf_array = mx.nd.full((self.batch_size * self.beam_size, 1), val=np.inf,
                                    ctx=self.context, dtype='float32')

        # offset for hypothesis indices in batch decoding
        self.offset = mx.nd.array(np.repeat(np.arange(0, self.batch_size * self.beam_size, self.beam_size), self.beam_size),
                                  dtype='int32', ctx=self.context)

        self._update_scores = UpdateScores()
        self._update_scores.initialize(ctx=self.context)
        self._update_scores.hybridize(static_alloc=True, static_shape=True)

        # Vocabulary selection leads to different vocabulary sizes across requests. Hence, we cannot use a
        # statically-shaped HybridBlock for the topk operation in this case; resorting to imperative topk
        # function in this case.
        if self.restrict_lexicon:
            if self.skip_topk:
                self._top = partial(utils.top1, offset=self.offset)  # type: Callable
            else:
                self._top = partial(utils.topk,
                                    k=self.beam_size,
                                    offset=self.offset,
                                    use_mxnet_topk=True)  # type: Callable
        else:
            if self.skip_topk:
                self._top = Top1(k=self.beam_size,
                                 batch_size=self.batch_size)  # type: mx.gluon.HybridBlock
                self._top.initialize(ctx=self.context)
                self._top.hybridize(static_alloc=True, static_shape=True)
            else:
                self._top = TopK(k=self.beam_size,
                                 batch_size=self.batch_size,
                                 vocab_size=len(self.vocab_target))  # type: mx.gluon.HybridBlock
                self._top.initialize(ctx=self.context)
                self._top.hybridize(static_alloc=True, static_shape=True)

        self._sort_by_index = SortByIndex()
        self._sort_by_index.initialize(ctx=self.context)
        self._sort_by_index.hybridize(static_alloc=True, static_shape=True)

        self._update_finished = NormalizeAndUpdateFinished(pad_id=C.PAD_ID,
                                                           eos_id=self.vocab_target[C.EOS_SYMBOL],
                                                           length_penalty_alpha=self.length_penalty.alpha,
                                                           length_penalty_beta=self.length_penalty.beta)
        self._update_finished.initialize(ctx=self.context)
        self._update_finished.hybridize(static_alloc=True, static_shape=True)

        self._prune_hyps = PruneHypotheses(threshold=self.beam_prune, beam_size=self.beam_size)
        self._prune_hyps.initialize(ctx=self.context)
        self._prune_hyps.hybridize(static_alloc=True, static_shape=True)

        self.global_avoid_trie = None
        if avoid_list is not None:
            self.global_avoid_trie = constrained.AvoidTrie()
            for phrase in data_io.read_content(avoid_list):
                phrase_ids = data_io.tokens2ids(phrase, self.vocab_target)
                if self.unk_id in phrase_ids:
                    logger.warning("Global avoid phrase '%s' contains an %s; this may indicate improper preprocessing.", ' '.join(phrase), C.UNK_SYMBOL)
                self.global_avoid_trie.add_phrase(phrase_ids)

        logger.info("Translator (%d model(s) beam_size=%d beam_prune=%s beam_search_stop=%s "
                    "ensemble_mode=%s batch_size=%d buckets_source=%s avoiding=%d)",
                    len(self.models),
                    self.beam_size,
                    'off' if not self.beam_prune else "%.2f" % self.beam_prune,
                    self.beam_search_stop,
                    "None" if len(self.models) == 1 else ensemble_mode,
                    self.batch_size,
                    self.buckets_source,
                    0 if self.global_avoid_trie is None else len(self.global_avoid_trie))

    @property
    def max_input_length(self) -> int:
        """
        Returns maximum input length for TranslatorInput objects passed to translate()
        """
        if self.source_with_eos:
            return self._max_input_length - C.SPACE_FOR_XOS
        else:
            return self._max_input_length

    @property
    def num_source_factors(self) -> int:
        return self.models[0].num_source_factors

    @staticmethod
    def _get_interpolation_func(ensemble_mode):
        if ensemble_mode == 'linear':
            return Translator._linear_interpolation
        elif ensemble_mode == 'log_linear':
            return Translator._log_linear_interpolation
        else:
            raise ValueError("unknown interpolation type")

    @staticmethod
    def _linear_interpolation(predictions):
        # pylint: disable=invalid-unary-operand-type
        return -mx.nd.log(utils.average_arrays(predictions))

    @staticmethod
    def _log_linear_interpolation(predictions):
        """
        Returns averaged and re-normalized log probabilities
        """
        log_probs = utils.average_arrays([p.log() for p in predictions])
        # pylint: disable=invalid-unary-operand-type
        return -log_probs.log_softmax()

    def translate(self, trans_inputs: List[TranslatorInput]) -> List[TranslatorOutput]:
        """
        Batch-translates a list of TranslatorInputs, returns a list of TranslatorOutputs.
        Splits oversized sentences to sentence chunks of size less than max_input_length.

        :param trans_inputs: List of TranslatorInputs as returned by make_input().
        :return: List of translation results.
        """
        translated_chunks = []  # type: List[IndexedTranslation]

        # split into chunks
        input_chunks = []  # type: List[IndexedTranslatorInput]
        for trans_input_idx, trans_input in enumerate(trans_inputs):
            # bad input
            if isinstance(trans_input, BadTranslatorInput):
                translated_chunks.append(IndexedTranslation(input_idx=trans_input_idx, chunk_idx=0,
                                                            translation=empty_translation()))
            # empty input
            elif len(trans_input.tokens) == 0:
                translated_chunks.append(IndexedTranslation(input_idx=trans_input_idx, chunk_idx=0,
                                                            translation=empty_translation()))
            else:
                # TODO(tdomhan): Remove branch without EOS with next major version bump, as future models will always be trained with source side EOS symbols
                if self.source_with_eos:
                    max_input_length_without_eos = self.max_input_length
                    # oversized input
                    if len(trans_input.tokens) > max_input_length_without_eos:
                        logger.debug(
                            "Input %s has length (%d) that exceeds max input length (%d). "
                            "Splitting into chunks of size %d.",
                            trans_input.sentence_id, len(trans_input.tokens),
                            self.buckets_source[-1], max_input_length_without_eos)
                        chunks = [trans_input_chunk.with_eos()
                                  for trans_input_chunk in trans_input.chunks(max_input_length_without_eos)]
                        input_chunks.extend([IndexedTranslatorInput(trans_input_idx, chunk_idx, chunk_input)
                                             for chunk_idx, chunk_input in enumerate(chunks)])
                    # regular input
                    else:
                        input_chunks.append(IndexedTranslatorInput(trans_input_idx,
                                                                   chunk_idx=0,
                                                                   translator_input=trans_input.with_eos()))
                else:
                    if len(trans_input.tokens) > self.max_input_length:
                        # oversized input
                        logger.debug(
                            "Input %s has length (%d) that exceeds max input length (%d). "
                            "Splitting into chunks of size %d.",
                            trans_input.sentence_id, len(trans_input.tokens),
                            self.buckets_source[-1], self.max_input_length)
                        chunks = [trans_input_chunk
                                  for trans_input_chunk in
                                  trans_input.chunks(self.max_input_length)]
                        input_chunks.extend([IndexedTranslatorInput(trans_input_idx, chunk_idx, chunk_input)
                                             for chunk_idx, chunk_input in enumerate(chunks)])
                    else:
                        # regular input
                        input_chunks.append(IndexedTranslatorInput(trans_input_idx,
                                                                   chunk_idx=0,
                                                                   translator_input=trans_input))

            if trans_input.constraints is not None:
                logger.info("Input %s has %d %s: %s", trans_input.sentence_id,
                            len(trans_input.constraints),
                            "constraint" if len(trans_input.constraints) == 1 else "constraints",
                            ", ".join(" ".join(x) for x in trans_input.constraints))

        # Sort longest to shortest (to rather fill batches of shorter than longer sequences)
        input_chunks = sorted(input_chunks, key=lambda chunk: len(chunk.translator_input.tokens), reverse=True)

        # translate in batch-sized blocks over input chunks
        for batch_id, batch in enumerate(utils.grouper(input_chunks, self.batch_size)):
            logger.debug("Translating batch %d", batch_id)
            # underfilled batch will be filled to a full batch size with copies of the 1st input
            rest = self.batch_size - len(batch)
            if rest > 0:
                logger.debug("Extending the last batch to the full batch size (%d)", self.batch_size)
                batch = batch + [batch[0]] * rest
            translator_inputs = [indexed_translator_input.translator_input for indexed_translator_input in batch]
            batch_translations = self._translate_nd(*self._get_inference_input(translator_inputs))
            # truncate to remove filler translations
            if rest > 0:
                batch_translations = batch_translations[:-rest]
            for chunk, translation in zip(batch, batch_translations):
                translated_chunks.append(IndexedTranslation(chunk.input_idx, chunk.chunk_idx, translation))
        # Sort by input idx and then chunk id
        translated_chunks = sorted(translated_chunks)

        # Concatenate results
        results = []  # type: List[TranslatorOutput]
        chunks_by_input_idx = itertools.groupby(translated_chunks, key=lambda translation: translation.input_idx)
        for trans_input, (input_idx, translations_for_input_idx) in zip(trans_inputs, chunks_by_input_idx):
            translations_for_input_idx = list(translations_for_input_idx)  # type: ignore
            if len(translations_for_input_idx) == 1:  # type: ignore
                translation = translations_for_input_idx[0].translation  # type: ignore
            else:
                translations_to_concat = [translated_chunk.translation
                                          for translated_chunk in translations_for_input_idx]
                translation = self._concat_translations(translations_to_concat)

            results.append(self._make_result(trans_input, translation))

        return results

    def _get_inference_input(self,
                             trans_inputs: List[TranslatorInput]) -> Tuple[mx.nd.NDArray, int, mx.nd.NDArray,
                                                                           int,
                                                                           List[Optional[constrained.RawConstraintList]],
                                                                           List[Optional[constrained.RawConstraintList]],
                                                                           mx.nd.NDArray]:
        """
        Assembles the numerical data for the batch.
        This comprises an NDArray for the source sentences, the bucket key (padded source length), and a list of
        raw constraint lists, one for each sentence in the batch, an NDArray of maximum output lengths for each sentence in the batch.
        Each raw constraint list contains phrases in the form of lists of integers in the target language vocabulary.

        :param trans_inputs: List of TranslatorInputs.
        :return NDArray of source ids (shape=(batch_size, bucket_key, num_factors)),
                bucket key, list of raw constraint lists, and list of phrases to avoid,
                and an NDArray of maximum output lengths.
        """

        max_tok = 0
        max_ctx_tok = 0
        max_doc_tok = 0
        for t in trans_inputs:

            if len(t.tokens) > max_tok:
                max_tok = len(t.tokens)
            if len(t.ctx_tokens) > max_ctx_tok - 1:
                max_ctx_tok = len(t.ctx_tokens)
            if len(t.doc_tokens) > max_doc_tok - 1:
                max_doc_tok = len(t.doc_tokens)
                # max_ctx_tok += 1

        bucket_key = data_io.get_ctx_doc_bucket(max_tok, max_ctx_tok, max_doc_tok, self.buckets_source)

        # bucket_key = data_io.get_bucket(max(len(inp.tokens) for inp in trans_inputs), self.buckets_source)


        source = mx.nd.zeros((len(trans_inputs), bucket_key[0]), ctx=self.context)
        ctx_source = mx.nd.zeros((len(trans_inputs), bucket_key[1]), ctx=self.context)
        doc_source = mx.nd.zeros((len(trans_inputs), bucket_key[2]), ctx=self.context)
        # ctx_source = mx.nd.full((len(trans_inputs), bucket_key[1]), np.nan, ctx=self.context)
        # np.full((num_samples, ctx_source_len), np.nan, dtype=self.dtype)

        raw_constraints = [None for x in range(self.batch_size)]  # type: List[Optional[constrained.RawConstraintList]]
        raw_avoid_list = [None for x in range(self.batch_size)]  # type: List[Optional[constrained.RawConstraintList]]

        max_output_lengths = []  # type: List[int]
        for j, trans_input in enumerate(trans_inputs):
            num_tokens = len(trans_input)
            num_ctx_tokens = len(trans_input.ctx_tokens)
            num_doc_tokens = len(trans_input.doc_tokens)

            max_output_lengths.append(self.models[0].get_max_output_length(data_io.get_ctx_doc_bucket(num_tokens, num_ctx_tokens, num_doc_tokens, self.buckets_source)[0]))

            source[j, :num_tokens] = data_io.tokens2ids(trans_input.tokens, self.source_vocabs[0])
            if num_ctx_tokens > 0:
                ctx_source[j, :num_ctx_tokens] = data_io.tokens2ids(trans_input.ctx_tokens, self.source_vocabs[0])
            if num_doc_tokens > 0:
                doc_source[j, :num_doc_tokens] = data_io.tokens2ids(trans_input.doc_tokens, self.source_vocabs[0])

            factors = trans_input.factors if trans_input.factors is not None else []
            num_factors = 1 + len(factors)
            if num_factors != self.num_source_factors:
                logger.warning("Input %d factors, but model(s) expect %d", num_factors,
                               self.num_source_factors)
            # for i, factor in enumerate(factors[:self.num_source_factors - 1], start=1):
            #     # fill in as many factors as there are tokens
            #
            #     source[j, :num_tokens, i] = data_io.tokens2ids(factor, self.source_vocabs[i])[:num_tokens]

            if trans_input.constraints is not None:
                raw_constraints[j] = [data_io.tokens2ids(phrase, self.vocab_target) for phrase in
                                      trans_input.constraints]

            if trans_input.avoid_list is not None:
                raw_avoid_list[j] = [data_io.tokens2ids(phrase, self.vocab_target) for phrase in
                                     trans_input.avoid_list]
                if any(self.unk_id in phrase for phrase in raw_avoid_list[j]):
                    logger.warning("Sentence %s: %s was found in the list of phrases to avoid; "
                                   "this may indicate improper preprocessing.", trans_input.sentence_id, C.UNK_SYMBOL)

        return source, bucket_key[0], ctx_source, bucket_key[1], doc_source, bucket_key[2], raw_constraints, raw_avoid_list, mx.nd.array(max_output_lengths, ctx=self.context, dtype='int32')

    def _make_result(self,
                     trans_input: TranslatorInput,
                     translation: Translation) -> TranslatorOutput:
        """
        Returns a translator result from generated target-side word ids, attention matrix, and score.
        Strips stop ids from translation string.

        :param trans_input: Translator input.
        :param translation: The translation + attention and score.
        :return: TranslatorOutput.
        """
        target_ids = translation.target_ids
        attention_matrix = translation.attention_matrix

        target_tokens = [self.vocab_target_inv[target_id] for target_id in target_ids]
        target_string = C.TOKEN_SEPARATOR.join(data_io.ids2tokens(target_ids, self.vocab_target_inv, self.strip_ids))

        attention_matrix = attention_matrix[:, :len(trans_input.tokens)]

        return TranslatorOutput(sentence_id=trans_input.sentence_id,
                                translation=target_string,
                                tokens=target_tokens,
                                attention_matrix=attention_matrix,
                                score=translation.score,
                                beam_histories=translation.beam_histories)

    def _concat_translations(self, translations: List[Translation]) -> Translation:
        """
        Combine translations through concatenation.

        :param translations: A list of translations (sequence, attention_matrix), score and length.
        :return: A concatenation if the translations with a score.
        """
        return _concat_translations(translations, self.stop_ids, self.length_penalty)

    def _translate_nd(self,
                      source: mx.nd.NDArray,
                      source_length: int,
                      ctx_source: mx.nd.NDArray,
                      ctx_source_length: int,
                      doc_source: mx.nd.NDArray,
                      doc_source_length: int,
                      raw_constraints: List[Optional[constrained.RawConstraintList]],
                      raw_avoid_list: List[Optional[constrained.RawConstraintList]],
                      max_output_lengths: mx.nd.NDArray) -> List[Translation]:
        """
        Translates source of source_length, given a bucket_key.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Bucket key.
        :param raw_constraints: A list of optional constraint lists.

        :return: Sequence of translations.
        """

        return self._get_best_from_beam(*self._beam_search(source, source_length, ctx_source, ctx_source_length, doc_source, doc_source_length, raw_constraints, raw_avoid_list, max_output_lengths))

    def _encode(self, sources: mx.nd.NDArray, source_length: int, ctx_sources: mx.nd.NDArray, ctx_source_length: int,
                doc_source: mx.nd.NDArray, doc_source_length: int) -> List[ModelState]:
        """
        Returns a ModelState for each model representing the state of the model after encoding the source.

        :param sources: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Bucket key.
        :return: List of ModelStates.
        """
        return [model.run_encoder(sources, ctx_sources, doc_source, source_length, ctx_source_length, doc_source_length) for model in self.models]

    def _decode_step(self,
                     prev_word: mx.nd.NDArray,
                     step: int,
                     source_length: int,
                     ctx_source_length: int,
                     states: List[ModelState],
                     models_output_layer_w: List[mx.nd.NDArray],
                     models_output_layer_b: List[mx.nd.NDArray]) \
            -> Tuple[mx.nd.NDArray, mx.nd.NDArray, List[ModelState]]:
        """
        Returns decoder predictions (combined from all models), attention scores, and updated states.

        :param prev_word: Previous words of hypotheses. Shape: (batch_size * beam_size,).
        :param step: Beam search iteration.
        :param source_length: Length of the input sequence.
        :param states: List of model states.
        :param models_output_layer_w: Custom model weights for logit computation (empty for none).
        :param models_output_layer_b: Custom model biases for logit computation (empty for none).
        :return: (probs, attention scores, list of model states)
        """
        bucket_key = (source_length, ctx_source_length, step)

        model_probs, model_attention_probs, model_states = [], [], []
        # We use zip_longest here since we'll have empty lists when not using restrict_lexicon
        for model, out_w, out_b, state in itertools.zip_longest(
                self.models, models_output_layer_w, models_output_layer_b, states):
            decoder_outputs, attention_probs, state = model.run_decoder(prev_word, bucket_key, state)
            # Compute logits and softmax with restricted vocabulary
            if self.restrict_lexicon:
                logits = model.output_layer(decoder_outputs, out_w, out_b)
                if self.skip_softmax:
                    # skip softmax for greedy decoding and single model
                    probs = logits
                else:
                    probs = mx.nd.softmax(logits)
            else:
                # Otherwise decoder outputs are already target vocab probs,
                # or logits if beam size is 1
                probs = decoder_outputs
            model_probs.append(probs)
            model_attention_probs.append(attention_probs)
            model_states.append(state)

        # model_probs[1] = mx.nd.concatenate([model_probs[1], mx.nd.zeros((self.beam_size, 1), ctx=self.context)], axis=1)

        neg_logprobs, attention_probs = self._combine_predictions(model_probs, model_attention_probs)
        return neg_logprobs, attention_probs, model_states

    def _combine_predictions(self,
                             probs: List[mx.nd.NDArray],
                             attention_probs: List[mx.nd.NDArray]) -> Tuple[mx.nd.NDArray, mx.nd.NDArray]:
        """
        Returns combined predictions of models as negative log probabilities and averaged attention prob scores.

        :param probs: List of Shape(beam_size, target_vocab_size).
        :param attention_probs: List of Shape(beam_size, bucket_key).
        :return: Combined negative log probabilities, averaged attention scores.
        """
        # average attention prob scores. TODO: is there a smarter way to do this?
        attention_prob_score = utils.average_arrays(attention_probs)

        # combine model predictions and convert to neg log probs
        if len(self.models) == 1:
            if self.skip_softmax:
                neg_probs = -probs[0]
            else:
                neg_probs = -mx.nd.log(probs[0])  # pylint: disable=invalid-unary-operand-type
        else:
            neg_probs = self.interpolation_func(probs)
        return neg_probs, attention_prob_score

    def _beam_search(self,
                     source: mx.nd.NDArray,
                     source_length: int,
                     ctx_source: mx.nd.NDArray,
                     ctx_source_length: int,
                     doc_source: mx.nd.NDArray,
                     doc_source_length: int,
                     raw_constraint_list: List[Optional[constrained.RawConstraintList]],
                     raw_avoid_list: List[Optional[constrained.RawConstraintList]],
                     max_output_lengths: mx.nd.NDArray) -> Tuple[np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray,
                                                                 List[Optional[constrained.ConstrainedHypothesis]],
                                                                 Optional[List[BeamHistory]]]:
        """
        Translates multiple sentences using beam search.

        :param source: Source ids. Shape: (batch_size, bucket_key, num_factors).
        :param source_length: Max source length.
        :param raw_constraint_list: A list of optional lists containing phrases (as lists of target word IDs)
               that must appear in each output.
        :param raw_avoid_list: A list of optional lists containing phrases (as lists of target word IDs)
               that must NOT appear in each output.
        :return List of best hypotheses indices, list of best word indices, list of attentions,
                array of accumulated length-normalized negative log-probs, hypotheses lengths, constraints (if any),
                beam histories (if any).
        """

        # Length of encoded sequence (may differ from initial input length)
        encoded_source_length = self.models[0].encoder.get_encoded_seq_len(source_length)
        utils.check_condition(all(encoded_source_length ==
                                  model.encoder.get_encoded_seq_len(source_length) for model in self.models),
                              "Models must agree on encoded sequence length")
        # Maximum output length
        max_output_length = self.models[0].get_max_output_length(source_length)

        # General data structure: each row has batch_size * beam blocks for the 1st sentence, with a full beam,
        # then the next block for the 2nd sentence and so on

        best_word_indices = mx.nd.full((self.batch_size * self.beam_size,), val=self.start_id, ctx=self.context,
                                       dtype='int32')

        # Best word and hypotheses indices across beam search steps from topk operation.
        best_hyp_indices_list = []  # type: List[mx.nd.NDArray]
        best_word_indices_list = []  # type: List[mx.nd.NDArray]

        # Beam history
        beam_histories = None  # type: Optional[List[BeamHistory]]
        if self.store_beam:
            beam_histories = [defaultdict(list) for _ in range(self.batch_size)]

        lengths = mx.nd.zeros((self.batch_size * self.beam_size, 1), ctx=self.context)
        finished = mx.nd.zeros((self.batch_size * self.beam_size,), ctx=self.context, dtype='int32')

        # Extending max_output_lengths to shape (batch_size * beam_size,)
        max_output_lengths = mx.nd.repeat(max_output_lengths, self.beam_size)

        # Attention distributions across beam search steps
        attentions = []  # type: List[mx.nd.NDArray]

        # scores_accumulated: chosen smallest scores in scores (ascending).
        scores_accumulated = mx.nd.zeros((self.batch_size * self.beam_size, 1), ctx=self.context)

        # reset all padding distribution cells to np.inf
        self.pad_dist[:] = np.inf

        # If using a top-k lexicon, select param rows for logit computation that correspond to the
        # target vocab for this sentence.
        models_output_layer_w = list()
        models_output_layer_b = list()
        pad_dist = self.pad_dist
        vocab_slice_ids = None  # type: mx.nd.NDArray
        if self.restrict_lexicon:
            source_words = utils.split(source, num_outputs=self.num_source_factors, axis=2, squeeze_axis=True)[0]
            # TODO: See note in method about migrating to pure MXNet when set operations are supported.
            #       We currently convert source to NumPy and target ids back to NDArray.
            vocab_slice_ids = self.restrict_lexicon.get_trg_ids(source_words.astype("int32").asnumpy())
            if any(raw_constraint_list):
                # Add the constraint IDs to the list of permissibled IDs, and then project them into the reduced space
                constraint_ids = np.array([word_id for sent in raw_constraint_list for phr in sent for word_id in phr])
                vocab_slice_ids = np.lib.arraysetops.union1d(vocab_slice_ids, constraint_ids)
                full_to_reduced = dict((val, i) for i, val in enumerate(vocab_slice_ids))
                raw_constraint_list = [[[full_to_reduced[x] for x in phr] for phr in sent] for sent in
                                       raw_constraint_list]

            vocab_slice_ids = mx.nd.array(vocab_slice_ids, ctx=self.context, dtype='int32')

            if vocab_slice_ids.shape[0] < self.beam_size + 1:
                # This fixes an edge case for toy models, where the number of vocab ids from the lexicon is
                # smaller than the beam size.
                logger.warning("Padding vocab_slice_ids (%d) with EOS to have at least %d+1 elements to expand",
                               vocab_slice_ids.shape[0], self.beam_size)
                n = self.beam_size - vocab_slice_ids.shape[0] + 1
                vocab_slice_ids = mx.nd.concat(vocab_slice_ids,
                                               mx.nd.full((n,), val=self.vocab_target[C.EOS_SYMBOL],
                                                          ctx=self.context, dtype='int32'),
                                               dim=0)

            pad_dist = mx.nd.full((self.batch_size * self.beam_size, vocab_slice_ids.shape[0] - 1),
                                  val=np.inf, ctx=self.context)
            for m in self.models:
                models_output_layer_w.append(m.output_layer_w.take(vocab_slice_ids))
                models_output_layer_b.append(m.output_layer_b.take(vocab_slice_ids))

        # (0) encode source sentence, returns a list
        model_states = self._encode(source, source_length, ctx_source, ctx_source_length, doc_source, doc_source_length)

        # Initialize the beam to track constraint sets, where target-side lexical constraints are present
        constraints = constrained.init_batch(raw_constraint_list, self.beam_size, self.start_id,
                                             self.vocab_target[C.EOS_SYMBOL])

        if self.global_avoid_trie or any(raw_avoid_list):
            avoid_states = constrained.AvoidBatch(self.batch_size, self.beam_size,
                                                  avoid_list=raw_avoid_list,
                                                  global_avoid_trie=self.global_avoid_trie)
            avoid_states.consume(best_word_indices)

        # Records items in the beam that are inactive. At the beginning (t==1), there is only one valid or active
        # item on the beam for each sentence
        inactive = mx.nd.ones((self.batch_size * self.beam_size), dtype='int32', ctx=self.context)
        inactive[::self.beam_size] = 0
        t = 1

        if self.models[0].model_type == C.MODEL_TYPE_AVG_EMB_TOK or self.models[0].model_type == C.MODEL_TYPE_MAX_EMB_TOK:
            source_length += 1

        if self.use_doc_pool:
            ctx_source_length = utils.compute_max_length_pool(ctx_source_length - 1, self.pool_window,
                                                              self.pool_stride) + 1

        for t in range(1, max_output_length):
            # (1) obtain next predictions and advance models' state
            # scores: (batch_size * beam_size, target_vocab_size)
            # attention_scores: (batch_size * beam_size, bucket_key)

            # ctx_source_length = ctx_source_length



            scores, attention_scores, model_states = self._decode_step(prev_word=best_word_indices,
                                                                       step=t,
                                                                       source_length=source_length,
                                                                       ctx_source_length=ctx_source_length,
                                                                       states=model_states,
                                                                       models_output_layer_w=models_output_layer_w,
                                                                       models_output_layer_b=models_output_layer_b)

            # (2) Update scores. Special treatment for finished and inactive rows. Inactive rows are inf everywhere;
            # finished rows are inf everywhere except column zero, which holds the accumulated model score
            scores = self._update_scores.forward(scores, finished, inactive, scores_accumulated, self.inf_array, pad_dist)

            # Mark entries that should be blocked as having a score of np.inf
            if self.global_avoid_trie or any(raw_avoid_list):
                block_indices = avoid_states.avoid()
                if len(block_indices) > 0:
                    scores[block_indices] = np.inf

            # (3) Get beam_size winning hypotheses for each sentence block separately. Only look as
            # far as the active beam size for each sentence.
            best_hyp_indices, best_word_indices, scores_accumulated = self._top(scores)

            # Constraints for constrained decoding are processed sentence by sentence
            if any(raw_constraint_list):
                best_hyp_indices, best_word_indices, scores_accumulated, constraints, inactive = constrained.topk(
                    self.batch_size,
                    self.beam_size,
                    inactive,
                    scores,
                    constraints,
                    best_hyp_indices,
                    best_word_indices,
                    scores_accumulated,
                    self.context)

            else:
                # All rows are now active (after special treatment of start state at t=1)
                inactive[:] = 0

            # Map from restricted to full vocab ids if needed
            if self.restrict_lexicon:
                best_word_indices = vocab_slice_ids.take(best_word_indices)

            # (4) Reorder fixed-size beam data according to best_hyp_indices (ascending)
            finished, lengths, attention_scores = self._sort_by_index.forward(best_hyp_indices,
                                                                              finished,
                                                                              lengths,
                                                                              attention_scores)

            # (5) Normalize the scores of newly finished hypotheses. Note that after this until the
            # next call to topk(), hypotheses may not be in sorted order.
            finished, scores_accumulated, lengths = self._update_finished.forward(best_word_indices,
                                                                                  max_output_lengths,
                                                                                  finished,
                                                                                  scores_accumulated,
                                                                                  lengths)

            # (6) Prune out low-probability hypotheses. Pruning works by setting entries `inactive`.
            if self.beam_prune > 0.0:
                inactive, best_word_indices, scores_accumulated = self._prune_hyps.forward(best_word_indices,
                                                                                           scores_accumulated,
                                                                                           finished,
                                                                                           self.inf_array,
                                                                                           self.zeros_array)

            # (7) update negative constraints
            if self.global_avoid_trie or any(raw_avoid_list):
                avoid_states.reorder(best_hyp_indices)
                avoid_states.consume(best_word_indices)

            # (8) optionally save beam history
            if self.store_beam:
                finished_or_inactive = mx.nd.clip(data=finished + inactive, a_min=0, a_max=1)
                unnormalized_scores = mx.nd.where(finished_or_inactive,
                                                  scores_accumulated * self.length_penalty(lengths),
                                                  scores_accumulated)
                normalized_scores = mx.nd.where(finished_or_inactive,
                                                scores_accumulated,
                                                scores_accumulated / self.length_penalty(lengths))
                for sent in range(self.batch_size):
                    rows = slice(sent * self.beam_size, (sent + 1) * self.beam_size)

                    best_word_indices_sent = best_word_indices[rows].asnumpy().tolist()
                    # avoid adding columns for finished sentences
                    if any(x for x in best_word_indices_sent if x != C.PAD_ID):
                        beam_histories[sent]["predicted_ids"].append(best_word_indices_sent)
                        beam_histories[sent]["predicted_tokens"].append([self.vocab_target_inv[x] for x in
                                                                         best_word_indices_sent])
                        # for later sentences in the matrix, shift from e.g. [5, 6, 7, 8, 6] to [0, 1, 3, 4, 1]
                        shifted_parents = best_hyp_indices[rows] - (sent * self.beam_size)
                        beam_histories[sent]["parent_ids"].append(shifted_parents.asnumpy().tolist())

                        beam_histories[sent]["scores"].append(unnormalized_scores[rows].asnumpy().flatten().tolist())
                        beam_histories[sent]["normalized_scores"].append(
                            normalized_scores[rows].asnumpy().flatten().tolist())

            # Collect best hypotheses, best word indices, and attention scores
            best_hyp_indices_list.append(best_hyp_indices)
            best_word_indices_list.append(best_word_indices)
            attentions.append(attention_scores)

            if self.beam_search_stop == C.BEAM_SEARCH_STOP_FIRST:
                at_least_one_finished = finished.reshape((self.batch_size, self.beam_size)).sum(axis=1) > 0
                if at_least_one_finished.sum().asscalar() == self.batch_size:
                    break
            else:
                if finished.sum().asscalar() == self.batch_size * self.beam_size:  # all finished
                    break

            # (9) update models' state with winning hypotheses (ascending)
            for ms in model_states:
                ms.sort_state(best_hyp_indices)

        logger.debug("Finished after %d / %d steps.", t + 1, max_output_length)

        # (9) Sort the hypotheses within each sentence (normalization for finished hyps may have unsorted them).
        folded_accumulated_scores = scores_accumulated.reshape((self.batch_size,
                                                                self.beam_size * scores_accumulated.shape[-1]))
        indices = mx.nd.cast(mx.nd.argsort(folded_accumulated_scores, axis=1), dtype='int32').reshape((-1,))
        best_hyp_indices, _ = mx.nd.unravel_index(indices, scores_accumulated.shape) + self.offset
        best_hyp_indices_list.append(best_hyp_indices)
        lengths = lengths.take(best_hyp_indices)
        scores_accumulated = scores_accumulated.take(best_hyp_indices)
        constraints = [constraints[x] for x in best_hyp_indices.asnumpy()]

        all_best_hyp_indices = mx.nd.stack(*best_hyp_indices_list, axis=1)
        all_best_word_indices = mx.nd.stack(*best_word_indices_list, axis=1)
        all_attentions = mx.nd.stack(*attentions, axis=1)

        return all_best_hyp_indices.asnumpy(), \
               all_best_word_indices.asnumpy(), \
               all_attentions.asnumpy(), \
               scores_accumulated.asnumpy(), \
               lengths.asnumpy().astype('int32'), \
               constraints, \
               beam_histories

    def _get_best_from_beam(self,
                            best_hyp_indices: np.ndarray,
                            best_word_indices: np.ndarray,
                            attentions: np.ndarray,
                            seq_scores: np.ndarray,
                            lengths: np.ndarray,
                            constraints: List[Optional[constrained.ConstrainedHypothesis]],
                            beam_histories: Optional[List[BeamHistory]] = None) -> List[Translation]:
        """
        Return the best (aka top) entry from the n-best list.

        :param best_hyp_indices: Array of best hypotheses indices ids. Shape: (batch * beam, num_beam_search_steps + 1).
        :param best_word_indices: Array of best hypotheses indices ids. Shape: (batch * beam, num_beam_search_steps).
        :param attentions: Array of attentions over source words.
                           Shape: (batch * beam, num_beam_search_steps, encoded_source_length).
        :param seq_scores: Array of length-normalized negative log-probs. Shape: (batch * beam, 1)
        :param lengths: The lengths of all items in the beam. Shape: (batch * beam). Dtype: int32.
        :param constraints: The constraints for all items in the beam. Shape: (batch * beam).
        :param beam_histories: The beam histories for each sentence in the batch.
        :return: List of Translation objects containing all relevant information.
        """
        # Initialize the best_ids to the first item in each batch
        best_ids = np.arange(0, self.batch_size * self.beam_size, self.beam_size, dtype='int32')

        if any(constraints):
            # For constrained decoding, select from items that have met all constraints (might not be finished)
            unmet = np.array([c.num_needed() if c is not None else 0 for c in constraints])
            filtered = np.where(unmet == 0, seq_scores.flatten(), np.inf)
            filtered = filtered.reshape((self.batch_size, self.beam_size))
            best_ids += np.argmin(filtered, axis=1).astype('int32')

        # Obtain sequences for all best hypotheses in the batch
        indices = self._get_best_word_indeces_for_kth_hypotheses(best_ids, best_hyp_indices)

        histories = beam_histories if beam_histories is not None else [None] * self.batch_size  # type: List
        return [self._assemble_translation(*x) for x in zip(best_word_indices[indices, np.arange(indices.shape[1])],
                                                            lengths[best_ids],
                                                            attentions[best_ids],
                                                            seq_scores[best_ids],
                                                            histories)]

    @staticmethod
    def _get_best_word_indeces_for_kth_hypotheses(ks: np.ndarray, all_hyp_indices: np.ndarray) -> np.ndarray:
        """
        Traverses the matrix of best hypotheses indices collected during beam search in reversed order by
        by using the kth hypotheses index as a backpointer.
        Returns and array containing the indices into the best_word_indices collected during beam search to extract
        the kth hypotheses.

        :param ks: The kth-best hypotheses to extract. Supports multiple for batch_size > 1. Shape: (batch,).
        :param all_hyp_indices: All best hypotheses indices list collected in beam search. Shape: (batch * beam, steps).
        :return: Array of indices into the best_word_indices collected in beam search
            that extract the kth-best hypothesis. Shape: (batch,).
        """
        batch_size = ks.shape[0]
        num_steps = all_hyp_indices.shape[1]
        result = np.zeros((batch_size, num_steps - 1), dtype=all_hyp_indices.dtype)
        # first index into the history of the desired hypotheses.
        pointer = all_hyp_indices[ks, -1]
        # for each column/step follow the pointer, starting from the penultimate column/step
        num_steps = all_hyp_indices.shape[1]
        for step in range(num_steps - 2, -1, -1):
            result[:, step] = pointer
            pointer = all_hyp_indices[pointer, step]
        return result

    @staticmethod
    def _assemble_translation(sequence: np.ndarray,
                              length: np.ndarray,
                              attention_lists: np.ndarray,
                              seq_score: np.ndarray,
                              beam_history: Optional[BeamHistory]) -> Translation:
        """
        Takes a set of data pertaining to a single translated item, performs slightly different
        processing on each, and merges it into a Translation object.

        :param sequence: Array of word ids. Shape: (batch_size, bucket_key).
        :param length: The length of the translated segment.
        :param attention_lists: Array of attentions over source words.
                                Shape: (batch_size * self.beam_size, max_output_length, encoded_source_length).
        :param seq_score: Array of length-normalized negative log-probs.
        :param beam_history: The optional beam histories for each sentence in the batch.
        :return: A Translation object.
        """
        length = int(length)
        sequence = sequence[:length].tolist()
        attention_matrix = attention_lists[:length, :]
        score = float(seq_score)
        beam_history_list = [beam_history] if beam_history is not None else []
        return Translation(sequence, attention_matrix, score, beam_history_list)

    def _print_beam(self,
                    sequences: mx.nd.NDArray,
                    accumulated_scores: mx.nd.NDArray,
                    finished: mx.nd.NDArray,
                    inactive: mx.nd.NDArray,
                    constraints: List[Optional[constrained.ConstrainedHypothesis]],
                    timestep: int) -> None:
        """
        Prints the beam for debugging purposes.

        :param sequences: The beam histories (shape: batch_size * beam_size, max_output_len).
        :param accumulated_scores: The accumulated scores for each item in the beam.
               Shape: (batch_size * beam_size, target_vocab_size).
        :param finished: Indicates which items are finished (shape: batch_size * beam_size).
        :param inactive: Indicates any inactive items (shape: batch_size * beam_size).
        :param timestep: The current timestep.
        """
        logger.info('BEAM AT TIMESTEP %d', timestep)
        for i in range(self.batch_size * self.beam_size):
            # for each hypothesis, print its entire history
            score = accumulated_scores[i].asscalar()
            word_ids = [int(x.asscalar()) for x in sequences[i]]
            unmet = constraints[i].num_needed() if constraints[i] is not None else -1
            hypothesis = '----------' if inactive[i] else ' '.join(
                [self.vocab_target_inv[x] for x in word_ids if x != 0])
            logger.info('%d %d %d %d %.2f %s', i + 1, finished[i].asscalar(), inactive[i].asscalar(), unmet, score,
                        hypothesis)


class PruneHypotheses(mx.gluon.HybridBlock):
    """
    A HybridBlock that returns an array of shape (batch*beam,) indicating which hypotheses are inactive due to pruning.

    :param threshold: Pruning threshold.
    :param beam_size: Beam size.
    """

    def __init__(self, threshold: float, beam_size: int) -> None:
        super().__init__()
        self.threshold = threshold
        self.beam_size = beam_size

    def hybrid_forward(self, F, best_word_indices, scores, finished, inf_array, zeros_array):
        scores_2d = F.reshape(scores, shape=(-1, self.beam_size))
        finished_2d = F.reshape(finished, shape=(-1, self.beam_size))
        inf_array_2d = F.reshape(inf_array, shape=(-1, self.beam_size))

        # best finished scores. Shape: (batch, 1)
        best_finished_scores = F.min(F.where(finished_2d, scores_2d, inf_array_2d), axis=1, keepdims=True)
        difference = F.broadcast_minus(scores_2d, best_finished_scores)
        inactive = F.cast(difference > self.threshold, dtype='int32')
        inactive = F.reshape(inactive, shape=(-1))

        best_word_indices = F.where(inactive, zeros_array, best_word_indices)
        scores = F.where(inactive, inf_array, scores)

        return inactive, best_word_indices, scores


class SortByIndex(mx.gluon.HybridBlock):
    """
    A HybridBlock that sorts args by the given indices.
    """

    def hybrid_forward(self, F, indices, *args):
        return [F.take(arg, indices) for arg in args]


class TopK(mx.gluon.HybridBlock):
    """
    A HybridBlock for a statically-shaped batch-wise topk operation.
    """

    def __init__(self, k: int, batch_size: int, vocab_size: int) -> None:
        """
        :param k: The number of smallest scores to return.
        :param batch_size: Number of sentences being decoded at once.
        :param vocab_size: Vocabulary size.
        """
        super().__init__()
        self.k = k
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        with self.name_scope():
            offset = mx.nd.repeat(mx.nd.arange(0, batch_size * k, k, dtype='int32'), k)
            self.offset = self.params.get_constant(name='offset', value=offset)

    def hybrid_forward(self, F, scores, offset):
        """
        Get the lowest k elements per sentence from a `scores` matrix.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :param offset: Array to add to the hypothesis indices for offsetting in batch decoding.
        :return: The row indices, column indices and values of the k smallest items in matrix.
        """
        folded_scores = F.reshape(scores, shape=(self.batch_size, self.k * self.vocab_size))
        values, indices = F.topk(folded_scores, axis=1, k=self.k, ret_typ='both', is_ascend=True)
        indices = F.reshape(F.cast(indices, 'int32'), shape=(-1,))
        unraveled = F.unravel_index(indices, shape=(self.batch_size * self.k, self.vocab_size))
        best_hyp_indices, best_word_indices = F.split(unraveled, axis=0, num_outputs=2, squeeze_axis=True)
        best_hyp_indices = best_hyp_indices + offset
        values = F.reshape(values, shape=(-1, 1))
        return best_hyp_indices, best_word_indices, values


class Top1(mx.gluon.HybridBlock):
    """
    A HybridBlock for a statically-shaped batch-wise first-best operation.

    Get the single lowest element per sentence from a `scores` matrix. Expects that
    beam size is 1, for greedy decoding.

    NOTE(mathmu): The current implementation of argmin in MXNet much slower than topk with k=1.
    """
    def __init__(self, k: int, batch_size: int) -> None:
        """
        :param k: The number of smallest scores to return.
        :param batch_size: Number of sentences being decoded at once.
        :param vocab_size: Vocabulary size.
        """
        super().__init__()
        with self.name_scope():
            offset = mx.nd.repeat(mx.nd.arange(0, batch_size * k, k, dtype='int32'), k)
            self.offset = self.params.get_constant(name='offset', value=offset)

    def hybrid_forward(self, F, scores, offset):
        """
        Get the single lowest element per sentence from a `scores` matrix. Expects that
        beam size is 1, for greedy decoding.

        :param scores: Vocabulary scores for the next beam step. (batch_size * beam_size, target_vocabulary_size)
        :param offset: Array to add to the hypothesis indices for offsetting in batch decoding.
        :return: The row indices, column indices and values of the smallest items in matrix.
        """
        best_word_indices = F.cast(F.argmin(scores, axis=1), dtype='int32')
        values = F.pick(scores, best_word_indices, axis=1)
        values = F.reshape(values, shape=(-1, 1))

        # for top1, the best hyp indices are equal to the plain offset
        best_hyp_indices = offset

        return best_hyp_indices, best_word_indices, values


class NormalizeAndUpdateFinished(mx.gluon.HybridBlock):
    """
    A HybridBlock for normalizing newly finished hypotheses scores with LengthPenalty.
    """

    def __init__(self, pad_id: int,
                 eos_id: int,
                 length_penalty_alpha: float = 1.0,
                 length_penalty_beta: float = 0.0) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.eos_id = eos_id
        with self.name_scope():
            self.length_penalty = LengthPenalty(alpha=length_penalty_alpha, beta=length_penalty_beta)

    def hybrid_forward(self, F, best_word_indices, max_output_lengths, finished, scores_accumulated, lengths):
        all_finished = F.broadcast_logical_or(best_word_indices == self.pad_id, best_word_indices == self.eos_id)
        newly_finished = F.broadcast_logical_xor(all_finished, finished)
        scores_accumulated = F.where(newly_finished,
                                     scores_accumulated / self.length_penalty(lengths),
                                     scores_accumulated)

        # Update lengths of all items, except those that were already finished. This updates
        # the lengths for inactive items, too, but that doesn't matter since they are ignored anyway.
        lengths = lengths + F.cast(1 - F.expand_dims(finished, axis=1), dtype='float32')

        # Now, recompute finished. Hypotheses are finished if they are
        # - extended with <pad>, or
        # - extended with <eos>, or
        # - at their maximum length.
        finished = F.broadcast_logical_or(F.broadcast_logical_or(best_word_indices == self.pad_id,
                                                                 best_word_indices == self.eos_id),
                                          (F.cast(F.reshape(lengths, shape=(-1,)), 'int32') >= max_output_lengths))

        return finished, scores_accumulated, lengths


class UpdateScores(mx.gluon.HybridBlock):
    """
    A HybridBlock that updates the scores from the decoder step with acumulated scores.
    Inactive hypotheses receive score inf. Finished hypotheses receive their accumulated score for C.PAD_ID.
    All other options are set to infinity.
    """

    def __init__(self):
        super().__init__()
        assert C.PAD_ID == 0, "This blocks only works with PAD_ID == 0"

    def hybrid_forward(self, F, scores, finished, inactive, scores_accumulated, inf_array, pad_dist):
        # Special treatment for finished and inactive rows. Inactive rows are inf everywhere;
        # finished rows are inf everywhere except column zero (pad_id), which holds the accumulated model score.
        # Items that are finished (but not inactive) get their previous accumulated score for the <pad> symbol,
        # infinity otherwise.
        scores = F.broadcast_add(scores, scores_accumulated)
        # pylint: disable=invalid-sequence-index
        pad_id_scores = F.where(F.broadcast_logical_and(finished, F.logical_not(inactive)), scores_accumulated, inf_array)
        # pad_dist. Shape: (batch*beam, vocab_size)
        pad_dist = F.concat(pad_id_scores, pad_dist)
        scores = F.where(F.broadcast_logical_or(finished, inactive), pad_dist, scores)
        return scores
