import os
import re
import glob
from collections import defaultdict
import json
import csv
from const import Const


def process_data(cfg):
    performance_path = cfg[Const.PERFORMANCES]
    output_path = cfg[Const.OUTPUT_PATH]
    if Const.MODELS not in cfg.keys() or len(cfg[Const.MODELS]) < 1:
        model_name = Const.UNKNOWN
    else:
        model_name = cfg[Const.MODELS][0].get(Const.MODEL_NAME, Const.UNKNOWN)

    grouped_data = get_grouped_data(performance_path, model_name)
    if len(grouped_data) > 0:
        save_data(grouped_data, output_path, cfg)


def get_grouped_data(performance_path, model_name):
    grouped_data = defaultdict(list)
    for task_name in os.listdir(performance_path):
        task_input_dir = os.path.join(performance_path, task_name)
        if not os.path.isdir(task_input_dir):
            continue
        try:
            batch_size = extract_batch_size(task_name)
        except ValueError as e:
            print(f"Skip {task_name}: {e}")
            continue

        json_path = find_performance_json(task_input_dir)

        if not json_path or not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skip")
            continue
        try:
            result = process_task_json(json_path, batch_size, model_name)
            grouped_data[batch_size].append(result)
            print(f"Process success: {task_name}")
        except Exception as e:
            print(f"Process failed: {task_name}, error: {e}")
    return grouped_data


def find_performance_json(task_dir):
    pattern = os.path.join(task_dir, f"{Const.PERF_JSON_PREFIX}*.json")
    matched_files = glob.glob(pattern)
    return matched_files[0] if matched_files else None


def process_task_json(json_path, batch_size, model_name):

    with open(json_path, "r") as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"Empty data in {json_path}")

    sums = {
        Const.TOTAL_TIME: 0.0,
        Const.FIRST_TOKEN_TIME: 0.0,
        Const.NON_FIRST_TOKEN_TIME: 0.0,
        Const.NON_FIRST_TOKEN_THROUGHPUT: 0.0,
        Const.E2E_THROUGHPUT: 0.0
    }
    count = 0

    for item in data:
        cur_bs = item.get(Const.BATCH_SIZE, 1)
        if cur_bs >= batch_size or (len(data) == 1 and cur_bs < batch_size):
            count += 1
            for key in sums:
                if key == Const.NON_FIRST_TOKEN_THROUGHPUT:
                    non_first_token_time = item.get(Const.NON_FIRST_TOKEN_TIME, 0.0)
                    sums[key] += cur_bs / (non_first_token_time / 1000) if non_first_token_time > 0 else 0
                else:
                    sums[key] += item.get(key, 0.0)

    for key in sums:
        sums[key] = sums[key] / count if count > 0 else 0

    result = {
        Const.HEADER_MODEL: model_name,
        Const.HEADER_BATCH_SIZE: data[0].get(Const.BATCH_SIZE, Const.UNKNOWN),
        Const.HEADER_IN_SEQ: data[0].get(Const.SEQ_LEN_IN, Const.UNKNOWN),
        Const.HEADER_OUT_SEQ: data[0].get(Const.SEQ_LEN_OUT, Const.UNKNOWN),
        Const.HEADER_TIME: sums[Const.TOTAL_TIME],
        Const.HEADER_FIRST_TOKEN_TIME: sums[Const.FIRST_TOKEN_TIME],
        Const.HEADER_NON_FIRST_TOKEN_TIME: sums[Const.NON_FIRST_TOKEN_TIME],
        Const.HEADER_NON_FIRST_TOKEN_THROUGHPUT: sums[Const.NON_FIRST_TOKEN_THROUGHPUT],
        Const.HEADER_THROUGHPUT: sums[Const.E2E_THROUGHPUT]
    }
    return result


def save_data(grouped_data, output_path, cfg):
    for batch_size, group_data in grouped_data.items():
        compute_non_first_token_throughput_average(group_data)
        compute_e2e_throughput_average(group_data)
        save_group_to_csv(batch_size, group_data, output_path, cfg)


def save_group_to_csv(batch_size, group_data, output_dir, cfg):
    tp = cfg[Const.MODELS][0].get(Const.WORLD_SIZE, 1)

    output_path = os.path.join(output_dir, f"performance_pa_batch{batch_size}_tp{tp}_result.csv")

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=Const.CSV_HEADER)
        writer.writeheader()
        writer.writerows(group_data)
    print(f"Save performance result to {output_path}")


def extract_batch_size(folder_name):
    match = re.search(r"bs_(\d+)", folder_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Invalid batch size in {folder_name}")


def compute_non_first_token_throughput_average(group_data):
    non_first_token_throughput = [data.get(Const.HEADER_NON_FIRST_TOKEN_THROUGHPUT, 0) for data in group_data]
    if len(non_first_token_throughput) > 0:
        non_first_token_throughput_average = sum(non_first_token_throughput) / len(non_first_token_throughput)
    else:
        non_first_token_throughput_average = 0
    group_data[-1][Const.HEADER_NON_FIRST_TOKEN_THROUGHPUT_AVG] = non_first_token_throughput_average
    return group_data


def compute_e2e_throughput_average(group_data):
    e2e_throughput = [data.get(Const.HEADER_THROUGHPUT, 0) for data in group_data]
    if len(e2e_throughput) > 0:
        e2e_throughput_average = sum(e2e_throughput) / len(e2e_throughput)
    else:
        e2e_throughput_average = 0
    group_data[-1][Const.HEADER_E2E_THROUGHPUT_AVG] = e2e_throughput_average
    return group_data
