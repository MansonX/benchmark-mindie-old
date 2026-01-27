import os
import sys
import csv
import json

from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from ais_bench.benchmark.models.base import BaseModel
from ais_bench.benchmark.models.performance import PerformanceModel
from ais_bench.benchmark.models.base_api import APITemplateParser
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.logging import get_logger
from ais_bench.benchmark.utils.prompt import PromptList


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}


@MODELS.register_module()
class MindieLLMModel(PerformanceModel):
    """
    Model wrapper around MindIE-LLM models.
    """

    def __init__(self,
                 environ_kwargs: Optional[Dict] = None,
                 **kwargs):
        super().__init__(path=kwargs.get('weight_dir'),
                         max_seq_len=kwargs.get('output_length'),
                         tokenizer_only=False,
                         meta_template=None)
        for key, value in environ_kwargs.items():
            os.environ[key] = value

        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = kwargs.get('world_size')
        self.block_size = kwargs.get('block_size')

        self.model_name = kwargs.get('model_name')
        self.data_type = kwargs.get('data_type')  # fp16 / bf16
        self.weight_dir = kwargs.get('weight_dir')
        self.max_position_embedding = kwargs.get('max_position_embedding')
        self.is_chat_model = kwargs.get('is_chat_model')
        self.prefill_batch_size = kwargs.get('prefill_batch_size')
        self.kw_args = kwargs.get('kw_args')
        self.dp = kwargs.get('dp')
        self.tp = kwargs.get('tp')
        self.sp = kwargs.get('sp')
        self.moe_tp = kwargs.get('moe_tp')
        self.pp = kwargs.get('pp')
        self.microbatch_size = kwargs.get('microbatch_size')
        self.moe_ep = kwargs.get('moe_ep')
        self.trust_remote_code = kwargs.get('trust_remote_code')
        self.ignore_eos = kwargs.get('ignore_eos')
        self.input_length = kwargs.get('input_length')
        self.output_length = kwargs.get('output_length')
        self.decode_batch_size = kwargs.get('decode_batch_size')
        self.input_token_len = kwargs.get('input_token_len', None)
        self.logger = get_logger()
        self.pa_runner = None
        self.rank_table_file = kwargs.get('rank_table_file')
        if self.rank_table_file:
            os.environ['RANK_TABLE_FILE'] = self.rank_table_file
            try:
                os.environ['WORLD_SIZE'] = str(self.world_size)
            except Exception as e:
                raise TypeError("world_size invalid") from e

        self.batch_latencies = []

        os.environ["ATB_LLM_BENCHMARK_ENABLE"] = "1"
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.pa_runner_perf_file_path = os.path.join(cur_dir, "benchmark.csv")
        os.environ["ATB_LLM_BENCHMARK_FILEPATH"] = self.pa_runner_perf_file_path

        self.get_model_or_runner(self.input_length, self.output_length)
        self.check_pa_runner()
        self.warm_up()

    def set_performance(self):
        self.do_performance = True
        self.ignore_eos = True # out len equal to max_out_len
        self.detail_perf_datas = []

    def check_pa_runner(self):
        if self.pa_runner == None:
            raise RuntimeError("Model loading failed")

    def warm_up(self):
        self.pa_runner.warm_up()

    def merge_perf_datas(self):
        ms = " ms"
        unit_token = " token/s"
        total_req = len(self.detail_perf_datas)
        e2el = sum(self.batch_latencies)
        if total_req <= 0 or e2el <= 0:
            self.logger.warning("No performance data to merge, please check")
            return {}
        common_metric_units_map = {
            "Benchmark Duration": ms,
            "Total Requests": None,
            "Request Throughput": " req/s",
            "Total Input Tokens": None,
            "Prefill Token Throughput": "",
            "Input Token Throughput": unit_token,
            "Total Output Tokens": None,
            "Output Token Throughput": unit_token,
            "Total Token Throughput": unit_token,
        }
        perf_key = "total"
        merge_res = {
            "Benchmark Duration": {perf_key: e2el * 1000},
            "Total Requests": {perf_key: total_req},
            "Request Throughput": {perf_key: total_req / e2el},
            "Total Input Tokens": {
                perf_key: sum(data["seq_len_in"] for data in self.detail_perf_datas)
            },
            "Prefill Token Throughput": {
                perf_key: sum(data["seq_len_in"] for data in self.detail_perf_datas)
                / sum(data["first_token_time"] for data in self.detail_perf_datas)
            },
            "Input Token Throughput": {
                perf_key: sum(data["seq_len_in"] for data in self.detail_perf_datas) / e2el
            },
            "Total Output Tokens": {
                perf_key: sum(data["seq_len_out"] for data in self.detail_perf_datas)
            },
            "Output Token Throughput": {
                perf_key: sum(data["seq_len_out"] for data in self.detail_perf_datas) / e2el
            },
            "Total Token Throughput": {
                perf_key: (
                    sum(
                        data["seq_len_in"] + data["seq_len_out"]
                        for data in self.detail_perf_datas
                    )
                    / e2el
                )
            },
        }
        for key,value in merge_res.items():
            value[perf_key] = str(round(value[perf_key], 4))
            if common_metric_units_map[key]:
                value[perf_key] += common_metric_units_map[key]
        return merge_res

    def handle_perf_result(self, output_filepath, output_filename):
        e2e_latency = sum(self.batch_latencies)
        if self.pa_runner_perf_file_path is not None and self.input_token_len is not None and self.rank == 0: # get pa runner special performance data
            if not os.path.exists(output_filepath):
                os.makedirs(output_filepath, mode=0o750)
            json_path = os.path.join(output_filepath, f"pa_runner_special_perf_data_{output_filename}.json")
            with open(json_path, "w") as file:
                json.dump(self.detail_perf_datas, file, ensure_ascii=False, indent=4)

            self.logger.info(f"PARUNNER special performance datas saved in {json_path}")
            return self.merge_perf_datas()
        return {"Benchmark Duration":{"total":str(round(e2e_latency, 4)) + ' ms'}}

    def get_model_or_runner(self, input_length, output_length, warmup_bs=0):

        try:
            ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
            if ATB_SPEED_HOME_PATH not in sys.path:
                sys.path.insert(0, os.path.join(ATB_SPEED_HOME_PATH, "../.."))
                sys.path.insert(0, ATB_SPEED_HOME_PATH)
            from atb_llm.utils.env import ENV
            from examples.run_pa import PARunner
        except Exception:
            raise RuntimeError("Failed to import necessary packages")

        rank = "rank"
        world_size = "world_size"
        local_rank = "local_rank"
        model_path = "model_path"
        max_position_embeddings = "max_position_embeddings"
        max_input_length = "max_input_length"
        max_output_length = "max_output_length"
        trust_remote_code = "trust_remote_code"


        prefill_batch_size = self.decode_batch_size if self.prefill_batch_size == 0 else self.prefill_batch_size

        input_dict = {
            rank: self.rank,
            local_rank: self.local_rank,
            world_size: self.world_size,
            'max_prefill_tokens': -1,
            'block_size': self.block_size,
            model_path: self.weight_dir,
            max_position_embeddings: (self.max_position_embedding
                                        if self.max_position_embedding != -1
                                        else input_length + output_length),
            'max_prefill_batch_size': prefill_batch_size,
            'max_batch_size': warmup_bs if warmup_bs != 0 else self.decode_batch_size,
            max_input_length: input_length,
            max_output_length: output_length,
            'kw_args': self.kw_args,
            'dp': self.dp,
            'tp': self.tp,
            'sp': self.sp,
            'moe_tp': self.moe_tp,
            'pp': self.pp,
            'microbatch_size': self.microbatch_size,
            'moe_ep': self.moe_ep,
            trust_remote_code: self.trust_remote_code
        }
        if self.model_name == "qwen2_72b" or self.model_name == "qwen2_7b":
            input_dict[max_position_embeddings] = None
        self.pa_runner = PARunner(**input_dict)
        model_dtype = self.pa_runner.model.dtype
        self.tokenizer = self.pa_runner.model.tokenizer
        user_dtype = DTYPE_MAP.get(self.data_type, None)
        if user_dtype != model_dtype:
            self.logger.error(
                "Inconsistent dtype: Input dtype: %s, model weight dtype: %s. please check",
                user_dtype, model_dtype)
            raise RuntimeError(
                f"Inconsistent dtype: Input dtype: {user_dtype}, " +
                f"model weight dtype: {model_dtype}. please check")

        self.logger.info('%d pa_runner: %s', self.rank, self.pa_runner)


    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.do_performance and self.input_token_len is not None: # enable token_input
            inputs = self._trans_to_input_ids(inputs)
            inputs = [self._padding_input_ids(input_ids) for input_ids in inputs]

        generate_texts, _, e2e_latency_per_bs = self.pa_runner.infer(inputs,
                                                    len(inputs),
                                                    max_out_len,
                                                    self.ignore_eos,
                                                    self.is_chat_model)

        if hasattr(self, "do_performance") and self.do_performance:
            self.batch_latencies.append(e2e_latency_per_bs)
            if self.pa_runner_perf_file_path is not None and self.input_token_len is not None and self.rank == 0: # get pa runner special performance data
                with open(self.pa_runner_perf_file_path, mode='r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    next(csv_reader)
                    second_row = next(csv_reader)
                    first_token_time = float(second_row[4]) / 1000
                    non_first_token_time = float(second_row[5]) / 1000
                try:
                    non_first_token_throughput = len(inputs) / non_first_token_time
                except ZeroDivisionError:
                    non_first_token_throughput = 0
                e2e_throughput = len(inputs) * max_out_len / e2e_latency_per_bs

                self.logger.info(
                    "seq_len_in: %d, seq_len_out: %d, total_time(s): %f,"
                    "first_token_time(ms): %f,"
                    "non_first_token_time(ms): %f,"
                    "non_first_token_throughput(1/s): %f,"
                    "e2e_time(s): %f, e2e_throughput(tokens/s): %f",
                    self.input_token_len, max_out_len, e2e_latency_per_bs,
                    first_token_time * 1000,
                    non_first_token_time * 1000,
                    non_first_token_throughput,
                    e2e_latency_per_bs, e2e_throughput
                )

                self.detail_perf_datas.append(
                    dict(
                        batch_size = len(inputs),
                        seq_len_in = self.input_token_len,
                        seq_len_out = max_out_len,
                        total_time = e2e_latency_per_bs,
                        first_token_time = first_token_time * 1000,
                        non_first_token_time = non_first_token_time * 1000,
                        e2e_time = e2e_latency_per_bs,
                        e2e_throughput = e2e_throughput
                    )
                )
            return None
        else:
            return generate_texts

    def _trans_to_input_ids(self, inputs: List[str]):
        input_ids_list = []
        for input in inputs:
            input_ids_list.append(self.tokenizer.encode(input, add_special_tokens=False))
        return input_ids_list

    def _padding_input_ids(self, input_ids: list):
        if len(input_ids) == 0:
            raise RuntimeError("Input for model infer is empty, please check")
        while (len(input_ids) < self.input_token_len):
            input_ids = input_ids * 2
        return input_ids[:self.input_token_len]

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

