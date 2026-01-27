from abc import ABC

import uuid
from ais_bench.benchmark.clients.base_client import BaseClient
from ais_bench.benchmark.utils import MiddleData
from ais_bench.benchmark.utils import get_logger
from ais_bench.benchmark.registry import CLIENTS
logger = get_logger()


@CLIENTS.register_module()
class TritonTextClient(BaseClient, ABC):
    def construct_request_body(
        self,
        inputs: str,
        parameters: dict = None,
    ) -> dict:
        if parameters.get("details") != True:
            logger.warning("Value of request parameter \"details\" will be changed to True")
        parameters.update({"details": True})
        if parameters.get("perf_stat") != True:
            logger.warning("Value of request parameter \"perf_stat\" will be changed to True")
        parameters.update({"perf_stat": True})
        return dict(id=str(uuid.uuid4()), text_input=inputs, parameters=parameters)

    def update_middle_data(self, res: dict, inputs: MiddleData):
        try:
            generated_text = res["text_output"]
        except Exception as e:
            raise RuntimeError(f"Process response failed and the reason is {e}")
        if generated_text:
            inputs.output = generated_text
            inputs.num_generated_chars = len(generated_text)
            inputs.prefill_latency = res["details"]["perf_stat"][0][1]
            inputs.decode_cost = [res["details"]["perf_stat"][i + 1][1] for i in range(len(res["details"]["perf_stat"]) - 1)]
            try:
                inputs.num_generated_tokens = res["details"]["generated_tokens"]
            except Exception:
                pass
        return generated_text