# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
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
import os
from typing import Any, Dict, List, Tuple, Callable, Optional, Union

import numpy as np
import torch
import torch.distributed
from diffusers import FluxPipeline
from diffusers.utils import is_torch_xla_available
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps, calculate_shift

from cfuser.config import EngineConfig, InputConfig
from cfuser.core.distributed import (
    get_runtime_state,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from .base_pipeline import cFuserPipelineBaseWrapper
from .register import cFuserPipelineWrapperRegister

from copy import deepcopy

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from cfuser.logger import init_logger

logger = init_logger(__name__)


@cFuserPipelineWrapperRegister.register(FluxPipeline)
class cFuserFluxPipeline(cFuserPipelineBaseWrapper):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        **kwargs,
    ):
        pipeline = FluxPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
        # tf_model = FluxTransformer2DModel(
        #     patch_size=1,
        #     in_channels=64,
        #     num_layers=1,
        #     num_single_layers=1,
        #     attention_head_dim=128,
        #     num_attention_heads=24,
        #     joint_attention_dim=4096,
        #     pooled_projection_dim=768,
        #     guidance_embeds=True,
        #     axes_dims_rope=(16, 56, 56),
        # ).to(torch.bfloat16)
        # pipeline.transformer = tf_model
        return cls(pipeline, engine_config)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def init_runtime_inputs(self, latents, latent_image_ids, index_req: int = 0):
        latents = latents[:, get_runtime_state(index_req).latent_idx[0] : get_runtime_state(index_req).latent_idx[1], :]
        latent_image_ids = latent_image_ids[
            get_runtime_state(index_req).latent_idx[0] : get_runtime_state(index_req).latent_idx[1], :
        ]
        return latents, latent_image_ids

    def prepare_run(
        self,
        input_config: InputConfig,
        steps: int = 3,
    ):
        # NOTE(@lry89757) this is only for torch.compile, for xdit they use this as a warmup for their pipeline.
        prompt = [""] * input_config.batch_size if input_config.batch_size > 1 else ""
        self.__call__(
            height=input_config.height,
            width=input_config.width,
            prompt=prompt,
            num_inference_steps=steps,
            output_type="latent",
            max_sequence_length=input_config.max_sequence_length,
            generator=torch.Generator(device="cuda").manual_seed(42),
        )

    def prologue_latents(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        index_req: int = 0,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        #! ---------------------------------------- ADDED BELOW ----------------------------------------
        # * set runtime state input parameters
        get_runtime_state(index_req).set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
        )
        #! ---------------------------------------- ADDED ABOVE ----------------------------------------

        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]  # 2D image seqlen downsampled by vae_scale_factor and flattened
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        latents, latent_image_ids = self.init_runtime_inputs(latents, latent_image_ids, index_req)

        return (
            latents,
            prompt_embeds,
            pooled_prompt_embeds,
            latent_image_ids,
            num_inference_steps,
            timesteps,
            guidance,
            text_ids,
            num_warmup_steps,
        )

    def denoise_latents(
        self,
        latents,
        prompt_embeds,
        pooled_prompt_embeds,
        latent_image_ids,
        num_inference_steps,
        timesteps,
        guidance,
        text_ids,
        num_warmup_steps,
        latents_2=None,
        prompt_embeds_2=None,
        pooled_prompt_embeds_2=None,
        latent_image_ids_2=None,
        timesteps_2=None,
        guidance_2=None,
        text_ids_2=None,
        inline_inference=False,
        async_op=False,
        no_stream=False,
        pack_qkv=True,
    ):
        if timesteps_2 is None:
            timesteps_2 = timesteps

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, (t, t_2) in enumerate(zip(timesteps, timesteps_2)):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                timestep_2 = timestep
                if latents_2 is not None:
                    timestep_2 = t_2.expand(latents_2.shape[0]).to(latents_2.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    hidden_states_1=latents_2,
                    timestep_1=timestep_2 / 1000,
                    guidance_1=guidance_2,
                    pooled_projections_1=pooled_prompt_embeds_2,
                    encoder_hidden_states_1=prompt_embeds_2,
                    txt_ids_1=text_ids_2,
                    img_ids_1=latent_image_ids_2,
                    joint_attention_kwargs_1=self.joint_attention_kwargs,
                    return_dict=False,
                    inline_inference=inline_inference,
                    async_op=async_op,
                    no_stream=no_stream,
                    pack_qkv=pack_qkv,
                )

                if latents_2 is not None:
                    noise_pred, noise_pred_2 = noise_pred
                else:
                    noise_pred = noise_pred[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents_2 is not None:
                    self.scheduler._step_index -= 1  # NOTE(@runyu): this is a hack to make the step index consistent with the original scheduler, but actually we should have two schedulers, or self.scheduler.step_index_2
                    latents_2 = self.scheduler.step(noise_pred_2, t_2, latents_2, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

            if get_sequence_parallel_world_size() > 1:
                sp_degree = get_sequence_parallel_world_size()
                sp_latents_list = [torch.empty_like(latents) for _ in range(sp_degree)]
                torch.distributed.all_gather(sp_latents_list, latents, group=get_sp_group())
                latents = torch.cat(sp_latents_list, dim=-2)
                if latents_2 is not None:
                    sp_latents_list_2 = [torch.empty_like(latents_2) for _ in range(sp_degree)]
                    torch.distributed.all_gather(sp_latents_list_2, latents_2, group=get_sp_group())
                    latents_2 = torch.cat(sp_latents_list_2, dim=-2)

        return latents, latents_2

    def denoise_latents_infer_req_batch(
        self,
        latents_list,
        prompt_embeds_list,
        pooled_prompt_embeds_list,
        latent_image_ids_list,
        num_inference_steps_list,
        timesteps_list,
        guidance_list,
        text_ids_list,
        num_warmup_steps_list,
    ):

        assert len(set(num_inference_steps_list)) == 1, "All num_inference_steps must be the same currently"
        num_inference_steps = num_inference_steps_list[0]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t_tuple in enumerate(zip(*timesteps_list)):
                if self.interrupt:
                    continue

                timestep_list = []
                for t, latent in zip(t_tuple, latents_list):
                    timestep = t.expand(latent.shape[0]).to(latent.dtype)
                    timestep_list.append(timestep)

                from cfuser.model_executor.models.transformers.tf_flux_infer_req_batch import (
                    transformer_flux_infer_req_batch_forward,
                )

                noise_pred_list = transformer_flux_infer_req_batch_forward(
                    self.transformer,
                    hidden_states_list=deepcopy(latents_list),
                    timestep_list=[timestep / 1000 for timestep in timestep_list],
                    guidance_list=guidance_list,
                    pooled_projections_list=pooled_prompt_embeds_list,
                    encoder_hidden_states_list=prompt_embeds_list,
                    txt_ids_list=text_ids_list,
                    img_ids_list=latent_image_ids_list,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )
                # print(f"noise_pred_list: {noise_pred_list[0].shape}")

                # compute the previous noisy sample x_t -> x_t-1
                for i, (noise_pred, latent) in enumerate(zip(noise_pred_list, latents_list)):

                    if i > 0:
                        self.scheduler._step_index -= 1  # NOTE(@runyu): this is a hack to make the step index consistent with the original scheduler, but actually we should have many schedulers
                    # logger.info(f"noise_pred: {noise_pred.shape}")
                    # logger.info(f"timestep: {t_tuple[i].shape}")
                    # logger.info(f"latent: {latent.shape}")
                    latents_list[i] = self.scheduler.step(noise_pred, t_tuple[i], latent, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps_list[0]) - 1 or (
                    (i + 1) > num_warmup_steps_list[0] and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

            if get_sequence_parallel_world_size() > 1:
                for i, latents in enumerate(latents_list):
                    sp_degree = get_sequence_parallel_world_size(index_req=i)
                    sp_latents_list = [torch.empty_like(latents) for _ in range(sp_degree)]
                    torch.distributed.all_gather(sp_latents_list, latents, group=get_sp_group(index_req=i))
                    latents_list[i] = torch.cat(sp_latents_list, dim=-2)

        return latents_list

    def decode_latents(self, latents, height, width, output_type):
        if latents is None:
            return None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            if output_type == "pil_latent":
                image = self.image_processor.postprocess(image, output_type="pil")
            else:
                image = self.image_processor.postprocess(image, output_type=output_type)

        return image

    def epilogue_latents(
        self,
        latents,
        height,
        width,
        output_type="pil",
        latents_2=None,
        height_2=None,
        width_2=None,
        output_type_2="pil",
    ):

        return self.decode_latents(latents, height, width, output_type), self.decode_latents(
            latents_2, height_2, width_2, output_type_2
        )

    @torch.no_grad()
    def inference_requests_batch(
        self,
        input_configs: Union[InputConfig, List[InputConfig]],
        generators: Union[torch.Generator, List[torch.Generator]],
    ):
        latents_list = []
        prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        latent_image_ids_list = []
        num_inference_steps_list = []
        timesteps_list = []
        guidance_list = []
        text_ids_list = []
        num_warmup_steps_list = []
        for index_req, (input_config, generator) in enumerate(zip(input_configs, generators)):
            (
                latents,
                prompt_embeds,
                pooled_prompt_embeds,
                latent_image_ids,
                num_inference_steps,
                timesteps,
                guidance,
                text_ids,
                num_warmup_steps,
            ) = self.prologue_latents(
                height=input_config.height,
                width=input_config.width,
                prompt=input_config.prompt,
                num_inference_steps=input_config.num_inference_steps,
                output_type=input_config.output_type,
                max_sequence_length=input_config.max_sequence_length,
                guidance_scale=0.0,
                generator=generator,
                index_req=index_req,
            )

            latents_list.append(latents)
            prompt_embeds_list.append(prompt_embeds)
            pooled_prompt_embeds_list.append(pooled_prompt_embeds)
            latent_image_ids_list.append(latent_image_ids)
            num_inference_steps_list.append(num_inference_steps)
            timesteps_list.append(timesteps)
            guidance_list.append(guidance)
            text_ids_list.append(text_ids)
            num_warmup_steps_list.append(num_warmup_steps)

        latents_list = self.denoise_latents_infer_req_batch(
            latents_list=latents_list,
            prompt_embeds_list=prompt_embeds_list,
            pooled_prompt_embeds_list=pooled_prompt_embeds_list,
            latent_image_ids_list=latent_image_ids_list,
            num_inference_steps_list=num_inference_steps_list,
            timesteps_list=timesteps_list,
            guidance_list=guidance_list,
            text_ids_list=text_ids_list,
            num_warmup_steps_list=num_warmup_steps_list,
        )

        image_list = [
            self.decode_latents(latents, input_config.height, input_config.width, input_config.output_type)
            for latents, input_config in zip(latents_list, input_configs)
        ]

        return image_list

    @torch.no_grad()
    def inference(
        self,
        input_config: InputConfig,
        input_config_2: InputConfig = None,
        # TODO(@lry89757): redesign the interface of the inference function
        inline_inference: bool = False,
        async_op: bool = False,  # actually, if input_config_2 is not None, async_op is always True
        no_stream: bool = False,
        return_dict: bool = True,
        pack_qkv: bool = True,
    ):

        logger.warning("inference is deprecated, use inference_requests_batch instead, will be removed in the future")

        (
            latents,
            prompt_embeds,
            pooled_prompt_embeds,
            latent_image_ids,
            num_inference_steps,
            timesteps,
            guidance,
            text_ids,
            num_warmup_steps,
        ) = self.prologue_latents(
            height=input_config.height,
            width=input_config.width,
            prompt=input_config.prompt,
            num_inference_steps=input_config.num_inference_steps,
            output_type=input_config.output_type,
            max_sequence_length=input_config.max_sequence_length,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
        )

        if input_config_2 is not None:
            (
                latents_2,
                prompt_embeds_2,
                pooled_prompt_embeds_2,
                latent_image_ids_2,
                num_inference_steps_2,
                timesteps_2,
                guidance_2,
                text_ids_2,
                num_warmup_steps_2,
            ) = self.prologue_latents(
                height=input_config_2.height,
                width=input_config_2.width,
                prompt=input_config_2.prompt,
                num_inference_steps=input_config_2.num_inference_steps,
                output_type=input_config_2.output_type,
                max_sequence_length=input_config_2.max_sequence_length,
                guidance_scale=0.0,
                generator=torch.Generator(device="cuda").manual_seed(input_config_2.seed),
                index_req=1,
            )

            assert (
                num_warmup_steps == num_warmup_steps_2
            ), "current overlap only supports the same number of warmup steps"
            assert (
                len(timesteps) == len(timesteps_2) == num_inference_steps
            ), "current overlap only supports the same number of inference steps"
            assert (
                num_inference_steps == num_inference_steps_2
            ), "current overlap only supports the same number of inference steps"

        else:
            latents_2 = None
            prompt_embeds_2 = None
            pooled_prompt_embeds_2 = None
            latent_image_ids_2 = None
            timesteps_2 = None
            guidance_2 = None
            text_ids_2 = None
            num_warmup_steps_2 = None

        latents, latents_2 = self.denoise_latents(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            latent_image_ids=latent_image_ids,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance=guidance,
            text_ids=text_ids,
            num_warmup_steps=num_warmup_steps,
            latents_2=latents_2,
            prompt_embeds_2=prompt_embeds_2,
            pooled_prompt_embeds_2=pooled_prompt_embeds_2,
            latent_image_ids_2=latent_image_ids_2,
            timesteps_2=timesteps_2,
            guidance_2=guidance_2,
            text_ids_2=text_ids_2,
            # overlap=overlap,
            inline_inference=inline_inference,
            no_stream=no_stream,
            async_op=async_op,
            pack_qkv=pack_qkv,
        )

        image, image_2 = self.epilogue_latents(
            latents=latents,
            height=input_config.height,
            width=input_config.width,
            output_type=input_config.output_type,
            latents_2=latents_2,
            height_2=input_config_2.height if input_config_2 is not None else None,
            width_2=input_config_2.width if input_config_2 is not None else None,
            output_type_2=(input_config_2.output_type if input_config_2 is not None else None),
        )
        if input_config.output_type == "pil_latent":
            return (latents, image) if image_2 is None else ((latents, image), (latents_2, image_2))
        elif return_dict:
            return (
                (FluxPipelineOutput(images=image),)
                if image_2 is None
                else (
                    FluxPipelineOutput(images=image),
                    FluxPipelineOutput(images=image_2),
                )
            )
        else:
            return (image, image_2) if image_2 is not None else (image,)

    @torch.no_grad()
    @cFuserPipelineBaseWrapper.check_model_parallel_state(sequence_parallel_available=True)
    @cFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        #! ---------------------------------------- ADDED BELOW ----------------------------------------
        # * set runtime state input parameters
        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
        )
        #! ---------------------------------------- ADDED ABOVE ----------------------------------------

        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        latents, latent_image_ids = self.init_runtime_inputs(latents, latent_image_ids)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                # logger.info(f"noise_pred: {noise_pred.shape}")
                # logger.info(f"t: {t.shape}")
                # logger.info(f"latents: {latents.shape}")
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

            if get_sequence_parallel_world_size() > 1:
                sp_degree = get_sequence_parallel_world_size()
                sp_latents_list = [torch.empty_like(latents) for _ in range(sp_degree)]
                torch.distributed.all_gather(sp_latents_list, latents, group=get_sp_group())
                latents = torch.cat(sp_latents_list, dim=-2)

        image = self.decode_latents(latents, height, width, output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if output_type == "pil_latent":
            return (latents, image)
        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
