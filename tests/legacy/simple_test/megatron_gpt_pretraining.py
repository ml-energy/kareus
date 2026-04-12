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
from nvtx_iteration_callback import NVTXIterationCallback
import torch

torch._dynamo.config.suppress_errors = True
mp.set_start_method("spawn", force=True)


@hydra_runner(config_path="conf", config_name="megatron_gpt_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    # Override some settings for comparison
    with open_dict(cfg):     
        # Ensure deterministic behavior
        cfg.model.seed = 42
        cfg.trainer.deterministic = True

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    
    nvtx_cb = NVTXIterationCallback()
    trainer.callbacks.append(nvtx_cb)
    
    # Add PyTorch Profiler for execution graph tracing
    if cfg.get('enable_pytorch_profiler', False):
        from pytorch_lightning.callbacks import Callback
        
        class PyTorchProfilerCallback(Callback):
            def __init__(self, start_step=5, end_step=7, trace_dir="./pytorch_traces"):
                super().__init__()
                self.start_step = start_step
                self.end_step = end_step
                self.trace_dir = Path(trace_dir)
                self.trace_dir.mkdir(parents=True, exist_ok=True)
                self.profiler = None
                
            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                if trainer.global_step == self.start_step:
                    logging.info(f"====== Starting PyTorch profiler at step {trainer.global_step} ======")
                    self.profiler = torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
                        on_trace_ready=self.trace_handler,
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True,
                        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
                    )
                    self.profiler.start()
                    
            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if self.profiler is not None:
                    if trainer.global_step < self.end_step:
                        self.profiler.step()
                    else:
                        logging.info(f"====== Stopping PyTorch profiler at step {trainer.global_step} ======")
                        self.profiler.stop()
                        self.profiler = None
                        
            def trace_handler(self, prof):
                # Save Chrome trace for execution graph visualization
                trace_file = self.trace_dir / f"trace_step_{prof.step_num}.json"
                prof.export_chrome_trace(str(trace_file))
                logging.info(f"PyTorch trace saved: {trace_file}")
                
                # Save table view for detailed analysis
                table_file = self.trace_dir / f"table_step_{prof.step_num}.txt"
                with open(table_file, 'w') as f:
                    f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
                logging.info(f"PyTorch table saved: {table_file}")
        
        pytorch_profiler_cb = PyTorchProfilerCallback(
            start_step=5,
            end_step=7,
            trace_dir="./pytorch_traces"
        )
        trainer.callbacks.append(pytorch_profiler_cb)

    exp_manager(trainer, cfg.exp_manager)

    # Continual training
    if cfg.model.get("restore_from_path") is not None:
        # Option 1: Restore only the model weights from a .nemo file
        logging.info(f"Continual training: loading weights from {cfg.model.restore_from_path}")
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel

        model_cfg = MegatronGPTSFTModel.merge_cfg_with(cfg.model.restore_from_path, cfg)
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.model.restore_from_path,
            override_config_path=model_cfg,
            trainer=trainer,
            save_restore_connector=NLPSaveRestoreConnector(),
        )
    elif cfg.model.get("restore_from_ckpt") is not None:
        # Option 2: Restore both model weights and optimizer states from a PTL checkpoint
        logging.info(f"Continual training: loading weights and optimizer states from {cfg.model.restore_from_ckpt}")
        trainer.ckpt_path = Path(cfg.model.restore_from_ckpt)
        model = MegatronGPTModel(cfg.model, trainer)

    # Start new pretraining or resume from a checkpoint if it exists
    else:
        model = MegatronGPTModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()
