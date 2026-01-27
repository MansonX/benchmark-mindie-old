import os
import argparse
from mmengine.config import Config

from const import Const
from utils import parse_lst_args, exec_command
from config import get_config_from_arg
from post_process import process_data


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, Const.MINDIE_LLM_EXAMPLES, Const.DEFAULT_CONFIG_FILE)
    if not os.path.exists(default_config_path):
        default_config_path = Const.DEFAULT_PATH
        print(f"Warning: default config file not found at {default_config_path}. Please specify a config file.")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=default_config_path, help='Aisbench config file path')

    parser.add_argument(
        '--case_pair',
        type=parse_lst_args,
        help='Specify input/output sequence length pairs in the format [[in_len1,out_len1],[in_len2,out_len2], ...]. '
            'Example: [[2048,2048],[1024,1024],[512,512],[256,256]]. '
            'Default: [[2048,2048],[1024,1024],[512,512],[256,256]]',
        default=[[2048,2048],[1024,1024],[512,512],[256,256]]
    )
    parser.add_argument(
        '--batch_size',
        type=parse_lst_args,
        help='Batch size specification. Can be: '
            '1) A single value (e.g., 1 or [1]), '
            '2) A range (e.g., 1,4 or [1,4]), or '
            '3) Multiple ranges (e.g., [[1,4],[2,8]]). '
            'Default: [16]',
        default=[16]
    )
    parser.add_argument('--dataset_path', help='The path to the dataset. Default: None.', default=None)
    parser.add_argument('--output_path', help='The path to save the result CSV file. Default: current directory.',
                        default=Const.DEFAULT_PATH)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    modified_config, cfg = get_config_from_arg(args)

    exec_command(modified_config)

    run_cfg = cfg.models[0].get(Const.RUN_CFG, {})
    node_rank = run_cfg.get(Const.NODE_RANK, 0)

    if not os.path.exists(cfg.work_dir) or not os.listdir(
                    cfg.work_dir) and node_rank == 0:
        raise ValueError(f"No performance data found in {cfg.work_dir}. Please check if the run was successful.")
    else:
        dirs = os.listdir(cfg.work_dir)
        dir_time_str = sorted(dirs)[-1]
        performance_path = os.path.join(cfg.work_dir, dir_time_str, Const.PERFORMANCES)
        if not os.path.exists(performance_path):
            raise ValueError(f"No performance data found in {performance_path}. Please check if the run was successful.")
        cfg[Const.PERFORMANCES] = performance_path
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, mode=0o750)
        cfg[Const.OUTPUT_PATH] = args.output_path

        process_data(cfg)

    if os.path.exists(modified_config):
        os.remove(modified_config)


if __name__ == "__main__":
    main()
