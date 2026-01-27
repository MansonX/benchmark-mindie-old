import os
import uuid
from mmengine.config import Config

from utils import process_case_batch
from const import Const


def get_config_from_arg(args):

    if not os.path.exists(args.config):
        raise ValueError(f"Error: Config file {args.config} does not exist.")

    cfg = Config.fromfile(args.config)

    if not cfg.models:
        raise ValueError("Error: At least one model configuration is required in the default settings.")

    cfg.models = generate_models_config(args, cfg.models)
    if args.dataset_path:
        if Const.DATASETS not in cfg.keys() or len(cfg.datasets) < 1:
            print("Warning: dataset_path is specified but no datasets are defined in the config file.")
        else:
            for dataset in cfg.datasets:
                dataset[Const.PATH] = args.dataset_path

    if cfg.models[0].get(Const.BATCH_SIZE, None):
        for dataset in cfg.datasets:
            try:
                del dataset[Const.INFER_CFG][Const.INFERENCER][Const.BATCH_SIZE]
            except (KeyError, TypeError):
                pass

    for model_cfg in cfg.models:
        model_cfg[Const.PATH] = model_cfg.get(Const.WEIGHT_PATH)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    unique_filename = f"temp_modified_config_{uuid.uuid4()}.py"
    temp_config_path = os.path.join(script_dir, unique_filename)

    cfg.dump(temp_config_path)

    return temp_config_path, cfg


def generate_models_config(args, model_list):
    case_pair = args.case_pair
    batch_size_list = args.batch_size

    modified_models = handle_case_pair_and_batch(model_list, case_pair, batch_size_list)

    return modified_models


def handle_case_pair_and_batch(model_list, case_pair, batch_size_list):

    if not case_pair and not batch_size_list:
        return handle_no_modifications(model_list)
    elif case_pair and not batch_size_list:
        return handle_case_pair_only(model_list, case_pair)
    elif not case_pair and batch_size_list:
        return handle_batch_size_only(model_list, batch_size_list)
    else:
        return handle_both_case_and_batch(model_list, case_pair, batch_size_list)


def handle_no_modifications(model_list):
    """Return original models when no modifications are needed"""
    return [model.copy() for model in model_list]


def handle_case_pair_only(model_list, case_pair):
    """Handle case where only case_pair is specified"""
    modified_models = []
    abbr_num = 0

    for model in model_list:
        for case in case_pair:
            new_model = model.copy()
            new_model.update({
                Const.INPUT_LENGTH: case[0],
                Const.OUTPUT_LENGTH: case[1],
                Const.INPUT_TOKEN_LEN: case[0],
                Const.MAX_OUT_LEN: case[1],
                Const.ABBR: (f"{new_model.get(Const.ABBR, Const.TASK)}"
                           f"{Const.REPLACEMENT_CHARACTER}{abbr_num}")
            })
            modified_models.append(new_model)
            abbr_num += 1
    return modified_models


def handle_batch_size_only(model_list, batch_size_list):
    """Handle case where only batch_size is specified"""
    modified_models = []
    abbr_num = 0

    if not isinstance(batch_size_list, list):
        raise ValueError("Error: batch_size_list must be a list.")
    if len(batch_size_list) < 1:
        raise ValueError("Error: batch_size_list must be a non-empty list.")

    for model in model_list:
        for bs in batch_size_list:
            new_model = model.copy()
            new_model.update({
                Const.BATCH_SIZE: bs,
                Const.DECODE_BATCH_SIZE: bs,
                Const.ABBR: (f"{new_model.get(Const.ABBR, Const.TASK)}"
                            f"{Const.REPLACEMENT_CHARACTER}{abbr_num}"
                            f"{Const.BS_SEGMENT}{bs}")
            })
            modified_models.append(new_model)
            abbr_num += 1
    return modified_models


def handle_both_case_and_batch(model_list, case_pair, batch_size_list):
    """Handle case where both case_pair and batch_size are specified"""
    modified_models = []
    abbr_num = 0

    if isinstance(batch_size_list[0], list):
        case_pair, batch_size_list = process_case_batch(case_pair, batch_size_list)

    combinations = []
    if not isinstance(batch_size_list[0], list):
        combinations = [(case, bs) for case in case_pair for bs in batch_size_list]
    else:
        combinations = [(case, bs) for case, bs_list in zip(case_pair, batch_size_list)
                      for bs in bs_list]

    for model in model_list:
        for case, bs in combinations:
            new_model = model.copy()
            new_model.update({
                Const.BATCH_SIZE: bs,
                Const.DECODE_BATCH_SIZE: bs,
                Const.INPUT_LENGTH: case[0],
                Const.OUTPUT_LENGTH: case[1],
                Const.INPUT_TOKEN_LEN: case[0],
                Const.MAX_OUT_LEN: case[1],
                Const.ABBR: (f"{new_model.get(Const.ABBR, Const.TASK)}"
                           f"{Const.REPLACEMENT_CHARACTER}{abbr_num}"
                           f"{Const.BS_SEGMENT}{bs}")
            })
            modified_models.append(new_model)
            abbr_num += 1
    return modified_models
