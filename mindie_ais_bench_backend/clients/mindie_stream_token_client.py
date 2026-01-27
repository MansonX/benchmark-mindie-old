from abc import ABC

from ais_bench.benchmark.clients.base_client import BaseStreamClient
from ais_bench.benchmark.utils import MiddleData
from ais_bench.benchmark.utils import get_logger
from ais_bench.benchmark.registry import CLIENTS
logger = get_logger()


@CLIENTS.register_module()
class MindieStreamTokenClient(BaseStreamClient, ABC):
    def __init__(self, url, retry, tokenizer):
        super().__init__(url, retry)
        self.tokenizer = tokenizer

    def construct_request_body(
        self,
        inputs: str,
        parameters: dict = None,
    ) -> dict:
        if parameters.get("details") != True:
            logger.warning("Value of request parameter \"details\" will be changed to True")
        parameters["details"] = True
        return dict(
            input_id=self.tokenizer.encode(inputs),
            stream=True,
            parameters=parameters
        )

    def process_stream_line(self, json_content: dict) -> dict:
        response = {}
        generated_text = json_content["token"].get("text", None)
        if generated_text:
            response.update({"generated_text": generated_text})
        if json_content.get("details"):
            response.update({"generated_tokens": json_content["details"]["generated_tokens"]})
        return response

    def update_middle_data(self, res: dict, inputs: MiddleData):
        generated_text = res.get("generated_text", "")
        if generated_text:
            inputs.output += generated_text
            inputs.num_generated_chars = len(generated_text)
        prefill_time = res.get("prefill_time")
        if prefill_time:
            inputs.prefill_latency = prefill_time
        decode_time = res.get("decode_time")
        if decode_time:
            inputs.decode_cost.append(decode_time)
        if res.get("generated_tokens"):
            inputs.num_generated_tokens = res.get("generated_tokens")
