class Const:
    """
    Class for const
    """
    MINDIE_LLM_EXAMPLES = 'mindie_llm_examples'
    DEFAULT_CONFIG_FILE = 'infer_mindie_llm_general.py'
    PATH = 'path'
    DATASETS = 'datasets'
    BATCH_SIZE = 'batch_size'
    DECODE_BATCH_SIZE = 'decode_batch_size'
    INPUT_TOKEN_LEN = 'input_token_len'
    INPUT_LENGTH = "input_length"
    OUTPUT_LENGTH = "output_length"
    MAX_OUT_LEN = 'max_out_len'
    INFER_CFG = 'infer_cfg'
    INFERENCER = 'inferencer'
    RUN_CFG = 'run_cfg'
    NODE_RANK = 'node_rank'
    ABBR = 'abbr'
    TASK = 'task'
    REPLACEMENT_CHARACTER = '_'
    BS_SEGMENT = '_bs_'
    DEFAULT_PATH = './'
    PERFORMANCES = 'performances'
    OUTPUT_PATH = 'output_path'
    WEIGHT_PATH = 'weight_dir'
    MODELS = 'models'
    WORLD_SIZE = 'world_size'
    UNKNOWN = 'unknown'
    MODEL_NAME = 'model_name'
    PERF_JSON_PREFIX = "pa_runner_special_perf_data_"

    SEQ_LEN_IN = 'seq_len_in'
    SEQ_LEN_OUT ='seq_len_out'
    TOTAL_TIME = 'total_time'
    FIRST_TOKEN_TIME = 'first_token_time'
    NON_FIRST_TOKEN_TIME = 'non_first_token_time'
    NON_FIRST_TOKEN_THROUGHPUT = 'non_first_token_throughput'
    E2E_THROUGHPUT = 'e2e_throughput'

    HEADER_MODEL = 'Model'
    HEADER_BATCH_SIZE = 'Batchsize'
    HEADER_IN_SEQ = 'In_seq'
    HEADER_OUT_SEQ = 'Out_seq'
    HEADER_TIME = 'Total time(s)'
    HEADER_FIRST_TOKEN_TIME = 'First token time(ms)'
    HEADER_NON_FIRST_TOKEN_TIME = 'Non-first token time(ms)'
    HEADER_NON_FIRST_TOKEN_THROUGHPUT = 'Non-first token Throughput(Tokens/s)'
    HEADER_THROUGHPUT = 'Throughput(Tokens/s)'
    HEADER_NON_FIRST_TOKEN_THROUGHPUT_AVG = 'Non-first token Throughput Average(Tokens/s)'
    HEADER_E2E_THROUGHPUT_AVG = 'E2E Throughput Average(Tokens/s)'

    CSV_HEADER = [
        HEADER_MODEL,
        HEADER_BATCH_SIZE,
        HEADER_IN_SEQ,
        HEADER_OUT_SEQ,
        HEADER_TIME,
        HEADER_FIRST_TOKEN_TIME,
        HEADER_NON_FIRST_TOKEN_TIME,
        HEADER_NON_FIRST_TOKEN_THROUGHPUT,
        HEADER_THROUGHPUT,
        HEADER_NON_FIRST_TOKEN_THROUGHPUT_AVG,
        HEADER_E2E_THROUGHPUT_AVG
    ]