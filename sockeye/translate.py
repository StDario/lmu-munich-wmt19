# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
Translation CLI.
"""
import argparse
import sys
import time
from contextlib import ExitStack
from math import ceil
from typing import Generator, Optional, List

from sockeye.lexicon import TopKLexicon
from sockeye.log import setup_main_logger
from sockeye.output_handler import get_output_handler, OutputHandler
from sockeye.utils import determine_context, log_basic_info, check_condition, grouper
from . import arguments
from . import constants as C
from . import data_io
from . import inference

logger = setup_main_logger(__name__, file_logging=False)


def main():
    params = arguments.ConfigArgumentParser(description='Translate CLI')
    arguments.add_translate_cli_args(params)
    args = params.parse_args()
    run_translate(args)


def run_translate(args: argparse.Namespace):

    if args.output is not None:
        global logger
        logger = setup_main_logger(__name__,
                                   console=not args.quiet,
                                   file_logging=True,
                                   path="%s.%s" % (args.output, C.LOG_NAME))

    if args.checkpoints is not None:
        check_condition(len(args.checkpoints) == len(args.models), "must provide checkpoints for each model")

    if args.skip_topk:
        check_condition(args.beam_size == 1, "--skip-topk has no effect if beam size is larger than 1")
        check_condition(len(args.models) == 1, "--skip-topk has no effect for decoding with more than 1 model")

    log_basic_info(args)

    output_handler = get_output_handler(args.output_type,
                                        args.output,
                                        args.sure_align_threshold)

    with ExitStack() as exit_stack:
        check_condition(len(args.device_ids) == 1, "translate only supports single device for now")
        context = determine_context(device_ids=args.device_ids,
                                    use_cpu=args.use_cpu,
                                    disable_device_locking=args.disable_device_locking,
                                    lock_dir=args.lock_dir,
                                    exit_stack=exit_stack)[0]
        logger.info("Translate Device: %s", context)

        if args.override_dtype == C.DTYPE_FP16:
            logger.warning('Experimental feature \'--override-dtype float16\' has been used. '
                           'This feature may be removed or change its behaviour in future. '
                           'DO NOT USE IT IN PRODUCTION!')

        models, source_vocabs, target_vocab = inference.load_models(
            context=context,
            max_input_len=args.max_input_len,
            max_ctx_input_len=args.max_ctx_input_len + 1,
            max_doc_input_len=args.max_doc_input_len + 1,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            model_folders=args.models,
            checkpoints=args.checkpoints,
            softmax_temperature=args.softmax_temperature,
            max_output_length_num_stds=args.max_output_length_num_stds,
            decoder_return_logit_inputs=args.restrict_lexicon is not None,
            cache_output_layer_w_b=args.restrict_lexicon is not None,
            override_dtype=args.override_dtype)
        restrict_lexicon = None  # type: Optional[TopKLexicon]
        if args.restrict_lexicon:
            restrict_lexicon = TopKLexicon(source_vocabs[0], target_vocab)
            restrict_lexicon.load(args.restrict_lexicon, k=args.restrict_lexicon_topk)
        store_beam = args.output_type == C.OUTPUT_HANDLER_BEAM_STORE
        translator = inference.Translator(context=context,
                                          ensemble_mode=args.ensemble_mode,
                                          bucket_source_width=args.bucket_width,
                                          length_penalty=inference.LengthPenalty(args.length_penalty_alpha,
                                                                                 args.length_penalty_beta),
                                          beam_prune=args.beam_prune,
                                          beam_search_stop=args.beam_search_stop,
                                          models=models,
                                          source_vocabs=source_vocabs,
                                          target_vocab=target_vocab,
                                          restrict_lexicon=restrict_lexicon,
                                          avoid_list=args.avoid_list,
                                          store_beam=store_beam,
                                          strip_unknown_words=args.strip_unknown_words,
                                          skip_topk=args.skip_topk,
                                          use_previous_translation=args.use_previous_translation,
                                          ctx_step_size=args.ctx_step_size,
                                          doc_step_size=args.doc_step_size)

        if models[0].config.config_encoder.use_doc_pool:
            translator.use_doc_pool = models[0].config.config_encoder.use_doc_pool
            translator.pool_window = models[0].config.config_encoder.doc_pool_window
            translator.pool_stride = models[0].config.config_encoder.doc_pool_stride

        read_and_translate(translator=translator,
                           output_handler=output_handler,
                           chunk_size=args.chunk_size,
                           input_file=args.input,
                           ctx_input_file=args.ctx_input,
                           doc_input_file=args.doc_input,
                           input_factors=args.input_factors,
                           input_is_json=args.json_input)


def make_inputs(input_file: Optional[str],
                ctx_input_file: Optional[str],
                doc_input_file: Optional[str],
                translator: inference.Translator,
                input_is_json: bool,
                input_factors: Optional[List[str]] = None) -> Generator[inference.TranslatorInput, None, None]:
    """
    Generates TranslatorInput instances from input. If input is None, reads from stdin. If num_input_factors > 1,
    the function will look for factors attached to each token, separated by '|'.
    If source is not None, reads from the source file. If num_source_factors > 1, num_source_factors source factor
    filenames are required.

    :param input_file: The source file (possibly None).
    :param translator: Translator that will translate each line of input.
    :param input_is_json: Whether the input is in json format.
    :param input_factors: Source factor files.
    :return: TranslatorInput objects.
    """
    if input_file is None:
        check_condition(input_factors is None, "Translating from STDIN, not expecting any factor files.")
        for sentence_id, line in enumerate(sys.stdin, 1):
            if input_is_json:
                yield inference.make_input_from_json_string(sentence_id=sentence_id, json_string=line)
            else:
                yield inference.make_input_from_factored_string(sentence_id=sentence_id,
                                                                factored_string=line,
                                                                translator=translator)
    else:
        input_factors = [] if input_factors is None else input_factors
        inputs = [input_file] + input_factors
        if not input_is_json:
            check_condition(translator.num_source_factors == len(inputs),
                            "Model(s) require %d factors, but %d given (through --input and --input-factors)." % (
                                translator.num_source_factors, len(inputs)))
        with ExitStack() as exit_stack:
            # streams = [exit_stack.enter_context(data_io.smart_open(i)) for i in inputs]
            # for sentence_id, inputs in enumerate(zip(*streams), 1):
            #     if input_is_json:
            #         yield inference.make_input_from_json_string(sentence_id=sentence_id, json_string=inputs[0])
            #     else:
            #         yield inference.make_input_from_multiple_strings(sentence_id=sentence_id, strings=list(inputs))

            for sentence_id, (input, ctx_input, doc_input) in enumerate(zip(open(input_file), open(ctx_input_file), open(doc_input_file))):

                yield inference.make_input_from_factored_string_ctx_doc(sentence_id=sentence_id,
                                                                        factored_string=input,
                                                                        ctx_factored_string=ctx_input,
                                                                        doc_factored_string=doc_input,
                                                                        translator=translator)


def read_and_translate(translator: inference.Translator,
                       output_handler: OutputHandler,
                       chunk_size: Optional[int],
                       input_file: Optional[str] = None,
                       ctx_input_file: Optional[str] = None,
                       doc_input_file: Optional[str] = None,
                       input_factors: Optional[List[str]] = None,
                       input_is_json: bool = False) -> None:
    """
    Reads from either a file or stdin and translates each line, calling the output_handler with the result.

    :param output_handler: Handler that will write output to a stream.
    :param translator: Translator that will translate each line of input.
    :param chunk_size: The size of the portion to read at a time from the input.
    :param input_file: Optional path to file which will be translated line-by-line if included, if none use stdin.
    :param input_factors: Optional list of paths to files that contain source factors.
    :param input_is_json: Whether the input is in json format.
    """
    batch_size = translator.batch_size
    if chunk_size is None:
        if translator.batch_size == 1:
            # No batching, therefore there is not need to read segments in chunks.
            chunk_size = C.CHUNK_SIZE_NO_BATCHING
        else:
            # Get a constant number of batches per call to Translator.translate.
            chunk_size = C.CHUNK_SIZE_PER_BATCH_SEGMENT * translator.batch_size
    else:
        if chunk_size < translator.batch_size:
            logger.warning("You specified a chunk size (%d) smaller than the batch size (%d). This will lead to "
                           "a reduction in translation speed. Consider choosing a larger chunk size." % (chunk_size,
                                                                                                         batch_size))

    logger.info("Translating...")

    total_time, total_lines = 0.0, 0
    for chunk in grouper(make_inputs(input_file, ctx_input_file, doc_input_file, translator, input_is_json, input_factors), size=chunk_size):
        chunk_time = translate(output_handler, chunk, translator)
        total_lines += len(chunk)
        total_time += chunk_time

    if total_lines != 0:
        logger.info("Processed %d lines in %d batches. Total time: %.4f, sec/sent: %.4f, sent/sec: %.4f",
                    total_lines, ceil(total_lines / batch_size), total_time,
                    total_time / total_lines, total_lines / total_time)
    else:
        logger.info("Processed 0 lines.")


def translate(output_handler: OutputHandler,
              trans_inputs: List[inference.TranslatorInput],
              translator: inference.Translator) -> float:
    """
    Translates each line from source_data, calling output handler after translating a batch.

    :param output_handler: A handler that will be called once with the output of each translation.
    :param trans_inputs: A enumerable list of translator inputs.
    :param translator: The translator that will be used for each line of input.
    :return: Total time taken.
    """
    tic = time.time()
    trans_outputs = translator.translate(trans_inputs)
    total_time = time.time() - tic
    batch_time = total_time / len(trans_inputs)
    for trans_input, trans_output in zip(trans_inputs, trans_outputs):
        output_handler.handle(trans_input, trans_output, batch_time)
    return total_time


if __name__ == '__main__':
    main()
