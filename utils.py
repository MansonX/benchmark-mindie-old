import ast
import sys
import subprocess

def process_case_batch(case_pair, input_list):
    if len(case_pair) != len(input_list):
        raise ValueError("inconsistent case_pair and batch_size input, length should be the same")
    sorted_batch = [sorted(bs, reverse=True) for bs in input_list]
    combined = list(zip(case_pair, sorted_batch))
    combined.sort(key=lambda x: sum(x[0]), reverse=True)
    sorted_case_pair = [pair for pair, _ in combined]
    sorted_input_list = [bs for _, bs in combined]
    return sorted_case_pair, sorted_input_list


def parse_lst_args(input_lst):

    try:
        input_lst = int(input_lst)
        return [input_lst]
    except ValueError:
        pass
    try:
        input_list = [int(bs) for bs in input_lst.split(',')]
        input_lst = input_list
        return input_lst
    except ValueError:
        pass

    try:
        input_lst = ast.literal_eval(input_lst)
        if isinstance(input_lst, list):
            if not input_lst:
                raise ValueError("Input is empty")
            else:
                return input_lst
        raise ValueError("Wrong input_lst format")
    except (ValueError, SyntaxError) as e:
        raise ValueError("Wrong input_lst format") from e


def exec_command(modified_config):
    command = ["ais_bench", modified_config, "--mode", "perf", "--debug"]

    try:
        print(f"Running command: {' '.join(command)}")
        # 使用 Popen 启动进程，实时捕获 stdout/stderr
        with subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,  # 行缓冲模式
            text=True,   # 输出为字符串而非字节
            encoding="utf-8",
            errors="replace"  # 避免编码错误中断
        ) as proc:
            # 实时读取标准输出
            for line in proc.stdout:
                print(line, end='', flush=True)  # 实时打印
            # 实时读取错误输出
            for line in proc.stderr:
                print(line, end='', flush=True, file=sys.stderr)

        # 等待进程结束并检查返回码
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, command)

    except subprocess.CalledProcessError as e:
        print(f"Command failed with code {e.returncode}", file=sys.stderr)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)


