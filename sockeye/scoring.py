# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Code for scoring.
"""
import logging
import os
import time
from typing import List, Optional, Tuple

import mxnet as mx

from . import constants as C
from . import data_io
from . import inference
from . import model
from . import utils
from . import vocab
from .inference import TranslatorInput, TranslatorOutput
from .output_handler import OutputHandler

logger = logging.getLogger(__name__)


class ScoringModel(model.SockeyeModel):
    """
    ScoringModel is a TrainingModel (which is in turn a SockeyeModel) that scores a pair of sentences.
    That is, it full unrolls over source and target sequences, running the encoder and decoder,
    but stopping short of computing a loss and backpropagating.
    It is analogous to TrainingModel, but more limited.

    :param config: Configuration object holding details about the model.
    :param model_dir: Directory containing the trained model.
    :param context: The context(s) that MXNet will be run in (GPU(s)/CPU).
    :param provide_data: List of input data descriptions.
    :param provide_label: List of label descriptions.
    :param default_bucket_key: Default bucket key.
    :param score_type: The type of score to output (negative logprob or logprob).
    :param length_penalty: The length penalty class to use.
    """

    def __init__(self,
                 config: model.ModelConfig,
                 model_dir: str,
                 context: List[mx.context.Context],
                 provide_data: List[mx.io.DataDesc],
                 provide_label: List[mx.io.DataDesc],
                 bucketing: bool,
                 default_bucket_key: Tuple[int, int],
                 score_type: str,
                 length_penalty: inference.LengthPenalty,
                 checkpoint: str = None,
                 softmax_temperature: Optional[float] = None) -> None:
        super().__init__(config)
        self.context = context
        self.bucketing = bucketing
        self.score_type = score_type
        self.length_penalty = length_penalty
        self.softmax_temperature = softmax_temperature

        # Create the computation graph
        self._initialize(provide_data, provide_label, default_bucket_key)

        # Load model parameters into graph
        if checkpoint is None:
            params_fname = os.path.join(model_dir, C.PARAMS_BEST_NAME)
        else:
            params_fname = os.path.join(model_dir, C.PARAMS_NAME % checkpoint)
        # params_fname = os.path.join(model_dir, C.PARAMS_BEST_NAME)
        super().load_params_from_file(params_fname)
        self.module.set_params(arg_params=self.params,
                               aux_params=self.aux_params,
                               allow_missing=False)

    def _initialize(self,
                    provide_data: List[mx.io.DataDesc],
                    provide_label: List[mx.io.DataDesc],
                    default_bucket_key: Tuple[int, int]) -> None:
        """
        Initializes model components, creates scoring symbol and module, and binds it.

        :param provide_data: List of data descriptors.
        :param provide_label: List of label descriptors.
        :param default_bucket_key: The default maximum (source, target) lengths.
        """
        source = mx.sym.Variable(C.SOURCE_NAME)
        # source_words = source.split(num_outputs=self.config.config_embed_source.num_factors,
        #                             axis=2, squeeze_axis=True)[0]
        source_length = utils.compute_lengths(source)
        ctx_source = mx.sym.Variable(C.CTX_SOURCE_NAME)
        ctx_source_length = utils.compute_lengths(ctx_source)
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)

        # labels shape: (batch_size, target_length) (usually the maximum target sequence length)
        labels = mx.sym.Variable(C.TARGET_LABEL_NAME)

        data_names = [C.SOURCE_NAME, C.CTX_SOURCE_NAME, C.TARGET_NAME]
        label_names = [C.TARGET_LABEL_NAME]

        # check provide_{data,label} names
        provide_data_names = [d[0] for d in provide_data]
        utils.check_condition(provide_data_names == data_names,
                              "incompatible provide_data: %s, names should be %s" % (provide_data_names, data_names))
        provide_label_names = [d[0] for d in provide_label]
        utils.check_condition(provide_label_names == label_names,
                              "incompatible provide_label: %s, names should be %s" % (provide_label_names, label_names))

        def sym_gen(seq_lens):
            """
            Returns a (grouped) symbol containing the summed score for each sentence, as well as the entire target
            distributions for each word.
            Also returns data and label names for the BucketingModule.
            """
            source_seq_len, target_seq_len, ctx_source_seq_len = seq_lens

            # source embedding
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # ctx source embedding
            (ctx_source_embed,
             ctx_source_embed_length,
             ctx_source_embed_seq_len) = self.ctx_embedding_source.encode(ctx_source, ctx_source_length,
                                                                          ctx_source_seq_len)

            # target embedding
            (target_embed,
             target_embed_length,
             target_embed_seq_len) = self.embedding_target.encode(target, target_length, target_seq_len)

            # encoder
            # source_encoded: (batch_size, source_encoded_length, encoder_depth)
            (source_encoded,
             source_encoded_length,
             source_encoded_seq_len,
             ctx_source_encoded,
             ctx_source_encoded_length,
             ctx_source_encoded_seq_len) = self.encoder.encode_with_ctx(source_embed,
                                                                        source_embed_length,
                                                                        source_embed_seq_len, ctx_source_embed,
                                                                        ctx_source_embed_length,
                                                                        ctx_source_embed_seq_len)

            # decoder
            # target_decoded: (batch-size, target_len, decoder_depth)
            target_decoded = self.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                          target_embed, target_embed_length, target_embed_seq_len,
                                                          ctx_source_encoded=ctx_source_encoded,
                                                          ctx_source_encoded_length=ctx_source_encoded_length,
                                                          ctx_source_encoded_max_length=ctx_source_encoded_seq_len)

            # output layer
            # logits: (batch_size * target_seq_len, target_vocab_size)
            logits = self.output_layer(mx.sym.reshape(data=target_decoded, shape=(-3, 0)))
            # logits after reshape: (batch_size, target_seq_len, target_vocab_size)
            logits = mx.sym.reshape(data=logits, shape=(-4, -1, target_embed_seq_len, 0))

            if self.softmax_temperature is not None:
                logits = logits / self.softmax_temperature

            # Compute the softmax along the final dimension.
            # target_dists: (batch_size, target_seq_len, target_vocab_size)
            target_dists = mx.sym.softmax(data=logits, axis=2, name=C.SOFTMAX_NAME)

            # Select the label probability, then take their logs.
            # probs and scores: (batch_size, target_seq_len)
            probs = mx.sym.pick(target_dists, labels)
            scores = mx.sym.log(probs)
            if self.score_type == C.SCORING_TYPE_NEGLOGPROB:
                scores = -1 * scores

            # Sum, then apply length penalty. The call to `mx.sym.where` masks out invalid values from scores.
            # zeros and sums: (batch_size,)
            zeros = mx.sym.zeros_like(scores)
            sums = mx.sym.sum(mx.sym.where(labels != 0, scores, zeros), axis=1) / (self.length_penalty(target_length - 1))

            # Return the sums and the target distributions
            # sums: (batch_size,) target_dists: (batch_size, target_seq_len, target_vocab_size)
            return mx.sym.Group([sums, target_dists]), data_names, label_names

        if self.bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", default_bucket_key)
            self.module = mx.mod.BucketingModule(sym_gen=sym_gen,
                                                 logger=logger,
                                                 default_bucket_key=default_bucket_key,
                                                 context=self.context)
        else:
            symbol, _, __ = sym_gen(default_bucket_key)
            self.module = mx.mod.Module(symbol=symbol,
                                        data_names=data_names,
                                        label_names=label_names,
                                        logger=logger,
                                        context=self.context)

        self.module.bind(data_shapes=provide_data,
                         label_shapes=provide_label,
                         for_training=False,
                         force_rebind=False,
                         grad_req='null')

    def run(self, batch: mx.io.DataBatch) -> List[mx.nd.NDArray]:
        """
        Runs the forward pass and returns the outputs.

        :param batch: The batch to run.
        :return: The grouped symbol (probs and target dists) and lists containing the data names and label names.
        """
        self.module.forward(batch, is_train=False)
        return self.module.get_outputs()


class Scorer:
    """
    Scorer class takes a ScoringModel and uses it to score a stream of parallel sentences.
    It also takes the vocabularies so that the original sentences can be printed out, if desired.

    :param model: The model to score with.
    :param source_vocabs: The source vocabularies.
    :param target_vocab: The target vocabulary.
    """
    def __init__(self,
                 models: List[ScoringModel],
                 source_vocabs: List[vocab.Vocab],
                 target_vocab: vocab.Vocab) -> None:
        self.source_vocab_inv = vocab.reverse_vocab(source_vocabs[0])
        self.target_vocab_inv = vocab.reverse_vocab(target_vocab)
        self.models = models
        self.exclude_list = {source_vocabs[0][C.BOS_SYMBOL], target_vocab[C.EOS_SYMBOL], C.PAD_ID}

    def score(self,
              score_iter,
              output_handler: OutputHandler):

        total_time = 0.
        sentence_no = 0
        for i, batch in enumerate(score_iter):

            batch_tic = time.time()

            # Run the model and get the outputs
            scores = [model.run(batch)[0] for model in self.models]

            batch_time = time.time() - batch_tic
            total_time += batch_time

            for i, (source, ctx_source, target) in enumerate(zip(batch.data[0], batch.data[1], batch.data[2])):

                # The "zeros" padding method will have filled remainder batches with zeros, so we can skip them here
                if source[0][0] == C.PAD_ID:
                    break

                sentence_no += 1
                score = [s[i] for s in scores]

                # Transform arguments in preparation for printing
                source_ids = [int(x) for x in source.asnumpy().tolist()]
                source_tokens = list(data_io.ids2tokens(source_ids, self.source_vocab_inv, self.exclude_list))
                ctx_source_ids = [int(x) for x in ctx_source.asnumpy().tolist()]
                ctx_source_tokens = list(data_io.ids2tokens(ctx_source_ids, self.source_vocab_inv, self.exclude_list))

                target_ids = [int(x) for x in target.asnumpy().tolist()]
                target_string = C.TOKEN_SEPARATOR.join(
                    data_io.ids2tokens(target_ids, self.target_vocab_inv, self.exclude_list))

                score = [s.asscalar() for s in score]

                if len(scores) > 1:
                    t_output = TranslatorOutput(sentence_no, target_string, None, None, None, score)
                else:
                    t_output = TranslatorOutput(sentence_no, target_string, None, None, score[0])

                # Output handling routines require us to make use of inference classes.
                output_handler.handle(TranslatorInput(sentence_no, source_tokens, ctx_tokens=ctx_source_tokens),
                                      t_output,
                                      batch_time)
