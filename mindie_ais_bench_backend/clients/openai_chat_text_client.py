from abc import ABC
import random

from ais_bench.benchmark.clients.base_client import BaseClient
from ais_bench.benchmark.utils import MiddleData
from ais_bench.benchmark.registry import CLIENTS


@CLIENTS.register_module()
class OpenAIChatTextClient(BaseClient, ABC):
    def construct_request_body(
        self,
        inputs: dict,
        parameters: dict = None,
    ) -> dict:
        data = dict(
            messages = inputs,
            stream = False,
        )
        data = data | parameters
        if isinstance(data.get("adapter_id"), list) and len(data.get("adapter_id")) > 0:
            data["model"] = random.choice(data["adapter_id"])
            del data["adapter_id"]
        return data

    def update_middle_data(self, res: dict, inputs: MiddleData):
        try:
            generated_text = res['choices'][0]['message']['content']
        except Exception as e:
            raise RuntimeError(f"Process response failed and the reason is {e}")
        if generated_text:
            inputs.output = generated_text
            inputs.num_generated_chars = len(generated_text)
            inputs.prefill_latency = res.get("prefill_time", 0)
            inputs.decode_cost = res.get("decode_time_arr", [])
        try:
            inputs.num_generated_tokens = res["usage"]["completion_tokens"]
        except Exception:
            pass

        return generated_text
