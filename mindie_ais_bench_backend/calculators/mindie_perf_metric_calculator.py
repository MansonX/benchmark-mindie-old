import csv
import collections
import math
import numpy as np
import copy
from ais_bench.benchmark.utils import get_logger
from ais_bench.benchmark.registry import PERF_METRIC_CALCULATORS
from ais_bench.benchmark.calculators.base_perf_metric_calculator import is_legal_percentage_str, DEFAULT_STATS, MAX_STATS_LEN


@PERF_METRIC_CALCULATORS.register_module()
class MindIEPerfMetricCalculator():
    def __init__(self, stats_list: list = DEFAULT_STATS):
        self.logger = get_logger()
        self._get_legal_stats_list(stats_list)

    def _init_datas(self, perf_details: dict):
        if sum(perf_details["requests"]["is_success"]) == 0:
            self.logger.error("All requests failed, can't calculate performance results. Please check the ERROR log from every responses!")
            raise ValueError("All requests failed!")
        self.stage_dict = {
            "total": self._get_requests_id(perf_details)
        }
        self.result = {}
        self.max_concurrency = perf_details["task"]["max_concurrency"]
        self.data_count = {}
        self.decode_latencies = {}
        self.success_count = {}
        self.empty_count = {}
        self.infer_time = {}
        self.metrics = {}
        self.common_metrics = {}
        self.ttft_sum = {}
        self.chars_sum = {}

        for stage_name, _ in self.stage_dict.items():
            self._process_result(perf_details.get("requests"), stage_name)

    def _get_requests_id(self, perf_details):
        return list(range(len(perf_details["requests"]["id"])))

    def _get_legal_stats_list(self, stats_list):
        if len(stats_list) > MAX_STATS_LEN:
            self.logger.warning(f"Len of stats list is over {MAX_STATS_LEN}! Only reserve the first {MAX_STATS_LEN} stat!")
            stats_list = stats_list[:MAX_STATS_LEN]
        self.stats_list = []
        for stat in stats_list:
            if stat not in ["Average", "Min", "Max", "Median"] and not is_legal_percentage_str(stat):
                self.logger.warning(f"Unknown stat: {stat}, won't take effect!")
                continue
            self.stats_list.append(stat)
        if len(self.stats_list) == 0:
            self.logger.warning("Can't find valid stat set, use \"Avarage\" stat.")
            self.stats_list.append("Average")

    def _process_result(self, full_result, stage_name):
        id_list = self.stage_dict.get(stage_name)
        result = {k: [v[i] for i in id_list] for k, v in full_result.items() if v is not None}
        self.data_count[stage_name] = len(full_result["is_success"])
        self.decode_latencies[stage_name] = result["decode_token_latencies"]
        self.success_count[stage_name] = sum(full_result["is_success"])
        self.empty_count[stage_name] = sum(full_result["is_empty"])
        self.infer_time[stage_name] = max(result["end_time"]) - min(result["start_time"])
        self.ttft_sum[stage_name] = sum(result["prefill_latency"])
        self.chars_sum[stage_name] = sum(result["generate_characters_len"])
        per_request_avg_decode_time = []
        # Compute the average decode latency per request
        if not math.isclose(sum(result["prefill_latency"]), 0):
            for i, value in enumerate(result["seq_latency"]):
                if value and result["generate_tokens_len"][i] > 1:  # Skip empty lists
                    tpot = (value - result["prefill_latency"][i]) / (result["generate_tokens_len"][i] - 1)
                    per_request_avg_decode_time.append(tpot)
            result["average_decode_latencies"] = per_request_avg_decode_time[:]
        else:
            result["average_decode_latencies"] = result["prefill_latency"]
        self.logger.info("Converting perf results of stage ...")
        self.result[stage_name] = self.convert_result(copy.deepcopy(result))
        self.logger.info("Finish Converting!")

    def get_common_res(self):
        return {k: v for k, v in self.common_metrics.items() if v is not None}

    def save_performance(self, out_path: str):
        """
        Save performance metrics to a CSV file.

        :param out_path: Path to the output CSV file.
        """
        if not self.metrics:
            raise ValueError("Metrics data is empty, cannot save to file.")

        try:
            # Extract headers from the first available entry
            first_entry = next(iter(self.metrics.values()), None)
            if first_entry is None:
                raise ValueError("Metrics data structure is invalid.")

            headers = list(first_entry[list(self.stage_dict.keys())[0]].keys())

            with open(out_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                # Write header: First column is the object name, followed by metric keys
                writer.writerow(["Performance Parameters"] + ["Stage"] + headers)

                # Write each object's data
                for obj_name, values in self.metrics.items():
                    for stage_name, _ in self.stage_dict.items():
                        row = [obj_name] + [stage_name] + [values[stage_name].get(key, "") for key in headers]
                        writer.writerow(row)

        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write to file '{out_path}': {e}")

    def convert_result(self, result: dict):
        remove_keys = [
            "id",
            "start_time",
            "end_time",
            "input_data",
            "input_token_id",
            "is_success",
            "is_empty",
            "request_id",
            "output",
            "output_token_id",
            "prefill_throughput",
        ]
        for key in remove_keys:
            result.pop(key, None)
        mapping = {
            "seq_latency": "E2EL",
            "prefill_latency": "TTFT",
            "average_decode_latencies": "TPOT",
            "decode_token_latencies": "ITL",
            "last_decode_latency": "LastITL",
            "decode_max_token_latency": "MaxITL",
            "input_tokens_len": "InputTokens",
            "generate_tokens_len": "OutputTokens",
            "generate_tokens_speed": "OutputTokenThroughput",
            "generate_characters_len": "GeneratedCharacters",
            "prefill_batch_size": "PrefillBatchsize",
            "decode_batch_size": "DecoderBatchsize",
            "queue_wait_time": "QueueWaitTime",
        }

        ans = {mapping_value: [] for mapping_value in mapping.values()}

        # Use a dictionary comprehension to populate the values
        for mapping_key, mapping_value in mapping.items():
            for value in result[mapping_key]:
                if isinstance(value, list):
                    ans[mapping_value].extend(value)
                else:
                    ans[mapping_value].append(value)
        if sum(ans["PrefillBatchsize"]) == 0:
            ans.pop("PrefillBatchsize")

        for key in ["DecoderBatchsize", "QueueWaitTime"]:
            res = ans.get(key)
            if not res or not res[-1]:
                ans.pop(key)

        for key in ["ITL"]:
            if isinstance(ans[key][0], np.ndarray) and not ans[key][0].any():
                ans.pop(key)

        for key in ["TTFT", "TPOT", "LastITL", "MaxITL"]:
            if math.isclose(sum(ans[key]), 0):
                ans.pop(key)

        return ans

    def __calc_metrics(self):
        """Calculate various statistical metrics for performance analysis."""
        # Iterate over all collected metrics
        for stage_name, _ in self.stage_dict.items():
            for metric, value in self.result[stage_name].items():
                stats = {k: 0 for k in self.stats_list}
                if value:
                    # Special handling for batch size metrics
                    if metric in {"PrefillBatchsize", "DecoderBatchsize"}:
                        value = self.__statistic_prefill_or_decode_batch_size(value)

                    # Compute statistical values
                    if isinstance(value[0], np.ndarray):
                        arr = np.concatenate(value)
                    else:
                        arr = np.array(value)
                    for stat in self.stats_list:
                        if stat == "Average":
                            stats[stat] = round(arr.mean(), 4)
                        elif stat == "Min":
                            stats[stat] = round(float(arr.min()), 4)
                        elif stat == "Max":
                            stats[stat] = round(float(arr.max()), 4)
                        elif stat == "Median":
                            stats[stat] = round(np.percentile(arr, 50), 4)
                        elif is_legal_percentage_str(stat):
                            stats[stat] = round(np.percentile(arr, int(stat[1:])), 4)

                # Store the computed metrics
                if self.metrics.get(metric) is None:
                    self.metrics[metric] = {stage_name: stats}
                else:
                    self.metrics[metric][stage_name] = stats

            # Assign fixed count value for all metrics
            for key in self.metrics:
                self.metrics[key][stage_name]["N"] = self.success_count[stage_name]
                # TPOT and ITL is the average decode latency per request, count the number of requests that have decode latency
                if key == "TPOT" or key == "ITL":
                    self.metrics[key][stage_name]["N"] = sum(
                        [
                            1
                            for decode_list in self.decode_latencies[stage_name]
                            if np.array(decode_list).any()
                        ]
                    )

    def __statistic_prefill_or_decode_batch_size(self, batch_sizes: list):
        """
        Process batch sizes by merging consecutive occurrences of the same number based on the first occurrence.

        Example:
            Input:  [2, 2, 5, 5, 3, 3, 5, 5, 5, 3]
            Output: [2, 5, 3]

        The method follows these rules:
        - When encountering a number X for the first time, store it and track its expected count (X-1).
        - Each subsequent occurrence of X decreases its count.
        - If count reaches zero, the number is removed from tracking.
        - If any unprocessed numbers remain, a warning is logged.

        Args:
            batch_sizes (list): List of batch sizes.

        Returns:
            list: Processed list of batch sizes.
        """
        if not batch_sizes:
            return []

        if isinstance(batch_sizes[0], list): # 2 dims list to 1 dim list
            batch_sizes = [item for sublist in copy.deepcopy(batch_sizes) for item in sublist]

        statistics = []
        count_dict = {}

        for batch_size in batch_sizes:
            if batch_size in count_dict:
                # Reduce remaining expected occurrences
                count_dict[batch_size] -= 1
                if count_dict[batch_size] == 0:
                    del count_dict[batch_size]  # Remove once count reaches zero
            else:
                # Register new batch size and track expected occurrences
                count_dict[batch_size] = batch_size - 1
                statistics.append(batch_size)

        if count_dict:
            self.logger.warning("Batch size is not fully compressed: %s", count_dict)

        return statistics

    def calculate(self):
        self.logger.info("Start calculating metrics ...")
        self.__calc_metrics()
        self.logger.info("Start calculating common metrics ...")
        self.__calc_common_metrics()
        self.logger.info("Start calculating add units ...")
        self.add_units()
        self.logger.info("Finish calculating perf data!")

    def __calc_common_metrics(self):
        common_metric_names = ["Benchmark Duration", "Total Requests", "Failed Requests", "Success Requests",
            "Concurrency", "Max Concurrency", "Request Throughput", "Total Input Tokens",
            "Prefill Token Throughput", "Total generated tokens", "Input Token Throughput",
            "Output Token Throughput", "Total Token Throughput", "lpct", "GenerateSpeedPerClient",
            "CharacterPerToken"]
        for name in common_metric_names:
            if self.common_metrics.get(name) is None:
                self.common_metrics[name] = {}

        for stage_name, _ in self.stage_dict.items():
            self.common_metrics["Benchmark Duration"][stage_name] = round(self.infer_time[stage_name] * 1000, 4)
            self.common_metrics["Total Requests"][stage_name] = self.data_count[stage_name]
            self.common_metrics["Failed Requests"][stage_name] = self.data_count[stage_name] - self.success_count[stage_name]
            self.common_metrics["Success Requests"][stage_name] = self.success_count[stage_name]
            self.common_metrics["Concurrency"][stage_name] = round(
                sum(self.result[stage_name]["E2EL"]) / self.infer_time[stage_name] / 1000, 4
            )
            self.common_metrics["Max Concurrency"][stage_name] = self.max_concurrency

            try:
                self.common_metrics["Request Throughput"][stage_name] = round(
                    self.success_count[stage_name] / self.infer_time[stage_name], 4
                )
            except ZeroDivisionError:
                self.common_metrics["Request Throughput"][stage_name] = 0

            self.common_metrics["Total Input Tokens"][stage_name] = sum(self.result[stage_name]["InputTokens"])
            if self.common_metrics["Total Input Tokens"][stage_name] != 0 and self.result[stage_name].get("TTFT") is not None:
                self.common_metrics["Prefill Token Throughput"][stage_name] = round(
                    1000
                    * self.common_metrics["Total Input Tokens"][stage_name]
                    / sum(self.result[stage_name]["TTFT"]),
                    4,
                )
            else:
                self.common_metrics.pop("Prefill Token Throughput", None)

            self.common_metrics["Total generated tokens"][stage_name] = sum(self.result[stage_name]["OutputTokens"])
            self.common_metrics["lpct"][stage_name] = self.ttft_sum[stage_name] / self.common_metrics["Total Input Tokens"][stage_name]
            self.common_metrics["CharacterPerToken"][stage_name] = self.chars_sum[stage_name] / \
                self.common_metrics["Total generated tokens"][stage_name]
            if self.infer_time[stage_name] > 0:
                self.common_metrics["Input Token Throughput"][stage_name] = round(
                    self.common_metrics["Total Input Tokens"][stage_name] / self.infer_time[stage_name], 4
                )
                self.common_metrics["Output Token Throughput"][stage_name] = round(
                    sum(self.result[stage_name]["OutputTokens"]) / self.infer_time[stage_name], 4
                )
                self.common_metrics["Total Token Throughput"][stage_name] = round(
                    (
                        self.common_metrics["Total Input Tokens"][stage_name]
                        + sum(self.result[stage_name]["OutputTokens"])
                    )
                    / self.infer_time[stage_name],
                    4,
                )

    def add_units(self):
        ms = " ms"
        unit_token = " token/s"
        metrics_units_map = {
            "E2EL": ms,
            "TTFT": ms,
            "TPOT": ms,
            "ITL": ms,
            "LastITL": ms,
            "MaxITL": ms,
            "InputTokens": None,
            "OutputTokens": None,
            "PrefillTokenThroughput": unit_token,
            "OutputTokenThroughput": unit_token,
            "GeneratedCharacters": None,
            "PrefillBatchsize": None,
            "DecoderBatchsize": None,
            "QueueWaitTime": " μs",
        }


        for metric, values in self.metrics.items():
            for stage_name, _ in self.stage_dict.items():
                if metric not in metrics_units_map or metrics_units_map.get(metric) is None:
                    continue
                for key, val in values[stage_name].items():
                    if key == "N":
                        continue
                    values[stage_name][key] = str(val) + metrics_units_map.get(metric)
        common_metric_units_map = {
            "Benchmark Duration": ms,
            "Total Requests": None,
            "Failed Requests": None,
            "Success Requests": None,
            "Concurrency": None,
            "Max Concurrency": None,
            "Request Throughput": " req/s",
            "Total Input Tokens": None,
            "Prefill Token Throughput": unit_token,
            "Input Token Throughput": unit_token,
            "Total Output Tokens": None,
            "Output Token Throughput": unit_token,
            "Total Token Throughput": unit_token,
            "lpct": ms,
            "CharacterPerToken": None,
        }

        for metric, value in self.common_metrics.items():
            for stage_name, _ in self.stage_dict.items():
                if (
                    metric not in common_metric_units_map
                    or common_metric_units_map.get(metric) is None
                ):
                    continue
                self.common_metrics[metric][stage_name] = str(value[stage_name]) + common_metric_units_map.get(
                    metric
                )