from abc import ABC
import json
import random

from ais_bench.benchmark.clients.base_client import BaseStreamClient
from ais_bench.benchmark.utils import MiddleData
from ais_bench.benchmark.registry import CLIENTS
from ais_bench.benchmark.clients.openai_chat_stream_client import OpenAIChatStreamClient as OpenAIChatStreamClientOrg


@CLIENTS.register_module()
class OpenAIChatStreamClient(OpenAIChatStreamClientOrg, ABC):
    def construct_request_body(
        self,
        inputs: list,
        parameters: dict = None,
    ) -> dict:
        data = super().construct_request_body(inputs, parameters)
        if isinstance(data.get("adapter_id"), list) and len(data.get("adapter_id")) > 0:
            data["model"] = random.choice(data["adapter_id"])
            del data["adapter_id"]
        return data
