# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path

# To suppress BF16 compile related issue in the CI runs with turing/V100
import torch._dynamo
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

import pytorch_lightning as pl
import torch

from zeus.optimizer.pipeline_frequency import PipelineFrequencyOptimizer

torch._dynamo.config.suppress_errors = True
mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_llama_3_2_3b_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # Override some settings for comparison
    with open_dict(cfg):     
        # Ensure deterministic behavior
        cfg.model.seed = 42
        cfg.trainer.deterministic = True

    trainer = MegatronTrainerBuilder(cfg).create_trainer()

    exp_manager(trainer, cfg.exp_manager)

    # opt = PipelineFrequencyOptimizer(
    #     server_addr="127.0.0.1", 
    #     server_port=7787
    # )

    model = MegatronGPTModel(cfg.model, trainer, None)

    trainer.fit(model)


if __name__ == '__main__':
    main()
