from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.utils.tokenizer import BenchmarkTokenizer
from mindie_ais_bench_backend.clients import MindieStreamTokenClient

from ais_bench.benchmark.models.base_api import handle_synthetic_input
from ais_bench.benchmark.models.performance_api import PerformanceAPIModel
from ais_bench.benchmark.utils.build import build_client_from_cfg

PromptType = Union[PromptList, str, dict]


@MODELS.register_module()
class MindieStreamTokenAPI(PerformanceAPIModel):
    """Model wrapper around OpenAI's models.

    Args:
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        request_rate (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        retry (int): Number of retires if the API call fails. Defaults to 2.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        host_ip (str): The  host ip of custom service, default "localhost".
        host_port (int): The host port of custom service, default "8080".
        enable_ssl (bool, optional): .
    """

    is_api: bool = True

    def __init__(self,
                 path,
                 max_seq_len: int = 4096,
                 request_rate: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 verbose: bool = False,
                 host_ip: str = "localhost",
                 host_port: int = 8080,
                 enable_ssl: bool = False,
                 custom_client = dict(type=MindieStreamTokenClient), # BaseClient
                 generation_kwargs: Optional[Dict] = None,
                 trust_remote_code: bool = False,
                 ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         request_rate=request_rate,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         generation_kwargs=generation_kwargs,
                         verbose=verbose,
                         trust_remote_code=trust_remote_code
                         )
        if self.tokenizer is None:
            self.tokenizer = self.tokenizer = BenchmarkTokenizer(self.path, trust_remote_code=self.trust_remote_code)

        self.host_ip = host_ip
        self.host_port = host_port
        self.enable_ssl = enable_ssl
        self.base_url = self._get_base_url()
        self.endpoint_url = os.path.join(self.base_url, "infer_token")
        self.init_client(custom_client)

    def init_client(self, custom_client):
        if not isinstance(custom_client, dict):
            self.logger.warning(f"Value of custom_client: {custom_client} is not a dict! Use Default")
            custom_client = dict(type=MindieStreamTokenClient)
        custom_client['url'] = self.endpoint_url
        custom_client['retry'] = self.retry
        custom_client['tokenizer'] = self.tokenizer
        self.client = build_client_from_cfg(custom_client)

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in AISBench'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        batch_size = kwargs.get("batch_size", len(inputs))
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(
                tqdm(executor.map(self._generate, inputs,
                                  [max_out_len] * len(inputs)),
                     total=len(inputs),
                     desc='Inferencing'))
        return results

    @handle_synthetic_input
    def _generate(self, input: PromptType, max_out_len: int) -> str:
        """Generate result given a input.

        Args:
            input (PromptType): A string or PromptDict.
                The PromptDict should be organized in AISBench'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        if isinstance(input, dict):
            data_id = input.get('data_id')
            input = input.get('prompt')
        else:
            data_id = -1

        if max_out_len <= 0:
            return ''
        cache_data = self.prepare_input_data(input, data_id)
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update({"max_new_tokens": max_out_len})

        response = self.client.request(cache_data, generation_kwargs)
        self.set_result(cache_data)
        return ''.join(response)

    def _get_base_url(self):
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}"
        return f"http://{self.host_ip}:{self.host_port}"