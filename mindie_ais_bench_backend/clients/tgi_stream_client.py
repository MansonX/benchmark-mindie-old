from abc import ABC
import random

from ais_bench.benchmark.clients.base_client import BaseStreamClient
from ais_bench.benchmark.utils import MiddleData
from ais_bench.benchmark.clients.tgi_stream_client import TGIStreamClient as TGIStreamClientOrg
from ais_bench.benchmark.utils import get_logger
from ais_bench.benchmark.registry import CLIENTS
logger = get_logger()


@CLIENTS.register_module()
class TGIStreamClient(TGIStreamClientOrg, ABC):
    def construct_request_body(
        self,
        inputs: str,
        parameters: dict = None,
    ) -> dict:
        if parameters.get("details") != True:
            logger.warning("Value of request parameter \"details\" will be changed to True")
        parameters["details"] = True
        if isinstance(parameters.get("adapter_id"), list) and len(parameters.get("adapter_id")) > 0:
            parameters["adapter_id"] = random.choice(parameters["adapter_id"])
        return dict(inputs=inputs, parameters=parameters)

    def process_stream_line(self, json_content: dict) -> dict:
        response = super().process_stream_line(json_content)
        if json_content.get("details"):
            response.update({"generated_tokens": json_content["details"]["generated_tokens"]})
        return response

    def update_middle_data(self, res: dict, inputs: MiddleData):
        super().update_middle_data(res, inputs)
        if res.get("generated_tokens"):
            inputs.num_generated_tokens = res.get("generated_tokens")
