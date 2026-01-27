import os
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from mmengine.config import ConfigDict

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList, is_mm_prompt

from ais_bench.benchmark.models.base_api import handle_synthetic_input
from ais_bench.benchmark.models.performance_api import PerformanceAPIModel
from ais_bench.benchmark.clients import OpenAIChatStreamClient, OpenAIChatTextClient
from ais_bench.benchmark.models import VLLMCustomAPIChat, VLLMCustomAPIChatStream

PromptType = Union[PromptList, str, dict]


@MODELS.register_module()
class VLLMCustomAPIChatLora(VLLMCustomAPIChat):
    def __init__(self,
                 path: str = "",
                 model: str = "",
                 max_seq_len: int = 4096,
                 request_rate: int = 1,
                 traffic_cfg: Optional[ConfigDict] = None,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 verbose: bool = False,
                 host_ip: str = "localhost",
                 host_port: int = 8080,
                 enable_ssl: bool = False,
                 custom_client = dict(type=OpenAIChatTextClient),
                 generation_kwargs: Optional[Dict] = None,
                 trust_remote_code: bool = False):
        super().__init__(path=path,
                         model=model,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         request_rate=request_rate,
                         traffic_cfg=traffic_cfg,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         verbose=verbose,
                         host_ip=host_ip,
                         host_port=host_port,
                         enable_ssl=enable_ssl,
                         custom_client=custom_client,
                         generation_kwargs=generation_kwargs,
                         trust_remote_code=trust_remote_code)
        lora_data_map_file = generation_kwargs.get("lora_data_map_file")
        self.lora_data_map = None
        if lora_data_map_file and isinstance(lora_data_map_file, str):
            file_path = os.path.abspath(lora_data_map_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as file:
                        self.lora_data_map = json.load(file)
                except Exception as e:
                    self.logger.warning(f"Failed to load lora data map file {file_path}, lora map will be empty")
                    self.lora_data_map = None

    @handle_synthetic_input
    def _generate(self, input: PromptType, max_out_len: int) -> str:
        """Generate results given a list of inputs.

        Args:
            inputs (PromptType): A string or PromptDict.
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

        if isinstance(input, str) or self.is_multi_modal:
            messages = [{'role': 'user', 'content': input}]
        elif is_mm_prompt(input):
            self.is_multi_modal = True
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update({"max_tokens": max_out_len})
        generation_kwargs.update({"model": self.model})
        if self.lora_data_map:
            lora_model_name = self.lora_data_map.get(f"{data_id}")
            if lora_model_name:
                generation_kwargs.update({"adapter_id": [lora_model_name]})

        cache_data = self.prepare_input_data(messages, data_id)

        response = self.client.request(cache_data, generation_kwargs)
        self.set_result(cache_data)

        return ''.join(response)


@MODELS.register_module()
class VLLMCustomAPIChatStreamLora(VLLMCustomAPIChatStream):
    def __init__(self,
                 path,
                 model: str = "",
                 max_seq_len: int = 4096,
                 request_rate: int = 1,
                 traffic_cfg: Optional[ConfigDict] = None,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 verbose: bool = False,
                 host_ip: str = "localhost",
                 host_port: int = 8080,
                 enable_ssl: bool = False,
                 custom_client = dict(type=OpenAIChatStreamClient),
                 generation_kwargs: Optional[Dict] = None,
                 trust_remote_code: bool = False):
        super().__init__(path=path,
                         model=model,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         request_rate=request_rate,
                         traffic_cfg=traffic_cfg,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         host_ip=host_ip,
                         host_port=host_port,
                         enable_ssl=enable_ssl,
                         custom_client=custom_client,
                         generation_kwargs=generation_kwargs,
                         verbose=verbose,
                         trust_remote_code=trust_remote_code)
        lora_data_map_file = generation_kwargs.get("lora_data_map_file")
        self.lora_data_map = None
        if lora_data_map_file and isinstance(lora_data_map_file, str):
            file_path = os.path.abspath(lora_data_map_file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as file:
                        self.lora_data_map = json.load(file)
                except Exception as e:
                    self.logger.warning(f"Failed to load lora data map file {file_path}, lora map will be empty")
                    self.lora_data_map = None

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
        if isinstance(input, str) or self.is_multi_modal:
            messages = [{'role': 'user', 'content': input}]
        elif is_mm_prompt(input):
            self.is_multi_modal = True
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update({"max_tokens": max_out_len})
        generation_kwargs.update({"model": self.model})
        if self.lora_data_map:
            lora_model_name = self.lora_data_map.get(f"{data_id}")
            if lora_model_name:
                generation_kwargs.update({"adapter_id": [lora_model_name]})

        cache_data = self.prepare_input_data(messages, data_id)

        response = self.client.request(cache_data, generation_kwargs)
        self.set_result(cache_data)

        return ''.join(response)