from mmengine.config import read_base
from mindie_ais_bench_backend.models import MindieLLMModel

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen import synthetic_datasets

datasets = [ # all_dataset_configs.py中导入了其他数据集配置，可以将gsm8k_0_shot_cot_str替换为其他一个或多个数据集
    *gsm8k_0_shot_cot_str,
]


models = [
    dict(
        ## 下列参数用于控制AISBench benchmark工具实现功能
        type=MindieLLMModel,
        attr="local", # local or service
        abbr='mindie-llm-api',
        max_out_len = 1024,  # 推理接口调用时设定的最大输出长度，建议不超过MindIE-LLM推理后端参数output_length
        run_cfg = dict(     # 多卡/多机多卡 参数，使用torchrun拉起任务
            num_gpus=2,     # 当前机器下使用的卡数
            num_procs=2,    # 当前机器下使用的进程数，与卡数应该相同
            nnodes=1,       # 使用的机器个数
            node_rank=0,    # 当前机器的id
            master_addr="localhost",   # 主机器的IP地址
            ),
        input_token_len = 16,        # 性能测评模式下期望用于模型推理的长度，建议不超过MindIE-LLM推理后端参数input_length

        ## 下列参数是用于拉起MindIE-LLM推理后端的参数，用于透传给MindIE-LLM后端，具体功能和含义由用户保证
        world_size = 2,  # 本次推理使用的卡总数
        block_size = 128,  # 初始化推理对象所需参数，预先计算内存所需的参数
        model_name = "qwen",  # 模型名称
        data_type = "bf16",  # 模型配置数据类型
        weight_dir = "/data/Qwen2.5-7B-Instruct",  # 模型权重路径
        max_position_embedding = -1,  # 初始化推理对象所需参数，-1表示使用input_length + output_length
        is_chat_model = False,  # 是否使用chat模板
        batch_size = 32,  # batch数，需要与decoddecode_batch_size相同
        decode_batch_size = 32,  # decode阶段的batchsize，需要与数据集测评任务中设定的batch_size相同
        prefill_batch_size = 0,  # prefill阶段的batchsize
        kw_args = "",
        trust_remote_code = False,  # 是否信任远端代码
        ignore_eos = False,  # 是否忽略推理终止符；设置了enable_detail_perf情况下,ignore_eos强制开启
        input_length = 4096,  # 初始化推理对象参数，input长度
        output_length = 1024,  # 初始化推理对象参数，output长度

        dp = -1,  # dp tp sp moe_tp pp microbatch_size moe_ep 模型并行策略参数
        tp = -1,
        sp = -1,
        moe_tp = -1,
        moe_ep = -1,
        pp = -1,
        microbatch_size = -1,

        rank_table_file = "",  # 多机模式下，rank_table路径

        environ_kwargs = dict(  # mindie-llm推理后端所需的环境变量配置, 具体模型有对应所需的环境变量
            ATB_LAYER_INTERNAL_TENSOR_REUSE = "1",
            ATB_OPERATION_EXECUTE_ASYNC = "1",
            ATB_CONVERT_NCHW_TO_ND = "1",
            TASK_QUEUE_ENABLE = "1",
            ATB_WORKSPACE_MEM_ALLOC_GLOBAL = "1",
            ATB_CONTEXT_WORKSPACE_SIZE = "0",
            ATB_LAUNCH_KERNEL_WITH_TILING = "1",
            ATB_LLM_ENABLE_AUTO_TRANSPOSE = "0",
            PYTORCH_NPU_ALLOC_CONF = "expandable_segments:True",
            LCCL_DETERMINISTIC = "1",
            HCCL_DETERMINISTIC = "true",
            ATB_MATMUL_SHUFFLE_K_ENABLE = "0",
            # ENABLE_GREEDY_SEARCH_OPT = "0",   # BoolQ数据数据集精度测评环境变量
        ),
    )
]


work_dir = 'outputs/mindie-llm-model/' # 工作路径