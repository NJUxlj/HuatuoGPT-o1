# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import gc
import math
import os
# textwrap 是 Python 标准库中的一个模块，用于文本包装和填充
# 主要用于格式化文本，比如自动换行、缩进等文本处理功能
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple, Union

import random,re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils import is_peft_available
from transformers.utils.deprecation import deprecate_kwarg

from trl.core import masked_mean, masked_whiten
from trl.models import create_reference_model
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from trl.trainer.ppo_config import PPOConfig
from trl.trainer.utils import generate_model_card, peft_module_casting_to_bf16


if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb


INVALID_LOGPROB = 1.0

# for o1
accumulate_rewards = []

# Using our get_reward
def get_reward_o1(
    model, response_ids, tokenizer, reward_tokenizer, pad_token_id, sub_answer,max_length = 4000

) -> Tuple[torch.Tensor]:
    '''
    :response_ids:  K 个 模型预测的答案 【以 token_id的形式】

    :sub_answer:  K 个 参考答案

    :max_length: 最大长度

    :return:  K 个 奖励值
    '''

    tmp = """<Model Response>
{}
</Model Response>

<Reference Answer>
{}
</Reference Answer>

Your task is to evaluate the model response by comparing it to the reference answer. If the model response is correct and aligns with the reference answer, output "True" . If it is incorrect or fails to select the correct option (if options are provided), output "False" . {}"""

    output_pattern = r"## Final Response\n\n(.*)"
    processed_batch = []
    output_matches = []
    for i in range(len(sub_answer)):  # 遍历每一对 模型预测 和 参考答案
        response = tokenizer.decode(response_ids[i], skip_special_tokens=True)

        count_en = response.count('## Final Response\n\n')
        count_thinking_en = response.count('## Thinking')

        if '## Final Response\n\n' in response and count_en == 1 and count_thinking_en == 1:
            output_match = re.search(output_pattern, response, re.S)
        else:
            output_match = None

        output_matches.append(output_match)

        if output_match is None:
            response = 'I do not know the answer.'
        else:
            response = output_match.group(1).strip()

        format_response = tmp.format(response, sub_answer[i], reward_tokenizer.eos_token)
        processed_batch.append(format_response)
    
    input_batch = reward_tokenizer(processed_batch, return_tensors="pt", add_special_tokens=False, max_length=max_length, padding=True,truncation=True).to(model.device)
    
    with torch.no_grad():
        logits = model(**input_batch,return_dict=True).logits
        probabilities = F.softmax(logits, dim=-1) 
        rewards = probabilities[:, 1] * 10

        rewards_list = []
        for i in range(len(sub_answer)):   # 遍历每个参考答案
            if output_matches[i] is None:
                rewards_list.append(0.0)
            else:
                p = probabilities[i, 1].item()  
                if p > 0.4:
                    rewards_list.append(1.0)
                else:
                    rewards_list.append(0.1)
        rewards = torch.tensor(rewards_list, device=probabilities.device, dtype=probabilities.dtype)

    # Update global reward statistics
    global accumulate_rewards
    accumulate_rewards.append(rewards.sum().item() / len(processed_batch))

    # Debugging rewards
    if random.random() < 0.1:
        for ii in range(len(processed_batch)):
            print('[reward_input]',processed_batch[ii],flush=True)
            print('[reward]',rewards[ii].item(),'\n',flush=True)
        print('-----------[avg_rewards]----------',sum(accumulate_rewards[-50:])/len(accumulate_rewards[-50:]),'\n',flush=True)
    return rewards


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(
            **kwargs,
        )
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits


class PPOTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    @deprecate_kwarg("tokenizer", new_name="processing_class", version="0.15.0", raise_if_both_names=True)
    def __init__(
        self,
        config: PPOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        reward_processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        policy: nn.Module,
        ref_policy: Optional[nn.Module],
        reward_model: nn.Module,
        train_dataset: Dataset,
        value_model: Optional[nn.Module] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
        peft_config: Optional["PeftConfig"] = None,
    ) -> None:
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must make a copy of it, or `None` if you use peft."
            )

        self.args = config
        args = config
        self.processing_class = processing_class
        self.reward_processing_class = reward_processing_class
        self.policy = policy

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        # peft support
        if not is_peft_available() and peft_config is not None:
            raise ImportError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_confg, we merge and unload it first
            if isinstance(self.policy, PeftModel):
                self.policy = self.policy.merge_and_unload()

            # get peft model with the given config
            self.policy = get_peft_model(self.policy, peft_config)
            if args.bf16 and getattr(self.policy, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(self.policy)

        self.is_peft_model = is_peft_available() and isinstance(self.policy, PeftModel)
        self.model_adapter_name = args.model_adapter_name
        self.ref_adapter_name = args.ref_adapter_name

        if ref_policy:
            self.ref_policy = ref_policy
        elif self.is_peft_model:
            self.ref_policy = None
        else:
            self.ref_policy = create_reference_model(self.policy)

        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_model = value_model
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = args.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [self.policy, self.ref_policy, self.value_model, self.reward_model]:
            if module is not None:
                disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = processing_class.eos_token_id
        self.model = PolicyAndValueWrapper(self.policy, self.value_model)
        self.model.config = self.policy.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )

            if self.ref_policy is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_policy = prepare_deepspeed(
                    self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            print("not using deepspeed!!!!!!!!",flush=True)
            if self.ref_policy is None:
                if not self.is_peft_model:
                    raise ValueError("No reference model and model is not a Peft model.")
            else:
                self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        # 如果模型是PEFT模型且没有指定ref_adapter_name,则禁用adapter
        # 否则使用nullcontext()(即不执行任何操作)
        with self.accelerator.unwrap_model(
            self.model.policy
        ).disable_adapter() if self.is_peft_model and not self.ref_adapter_name else nullcontext():
            # 如果指定了ref_adapter_name,则切换到该adapter
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.ref_adapter_name)
            yield
            # 在上下文结束后,如果之前切换了adapter,则切换回默认adapter
            if self.ref_adapter_name:
                self.model.policy.set_adapter(self.model_adapter_name or "default")

    # fix the save_model bug
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        Trainer.save_model(self, output_dir, _internal_call)

        self.model = backup_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.is_deepspeed_enabled:
            # 在DeepSpeed模式下,模型权重名称会带有'policy.'前缀
            # 这是因为在DeepSpeed中,模型被包装成了policy属性
            # 保存时需要去掉这个前缀,以保持权重名称的一致性
            state_dict = {name.removeprefix('policy.'): param for name, param in state_dict.items()
                          if name.startswith('policy.')}

        super()._save(output_dir, state_dict)

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        # 定义统计数据的形状,包含三个维度:
        # 1. PPO训练轮数(num_ppo_epochs)
        # 2. 每轮中的mini-batch数量(num_mini_batches) 
        # 3. 梯度累积步数(gradient_accumulation_steps)
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        
        # 初始化各种统计指标的张量:
        # approxkl_stats: 近似KL散度统计,用于监控新旧策略的差异
        # pg_clipfrac_stats: 策略梯度裁剪比例统计,记录被裁剪的更新比例
        # pg_loss_stats: 策略梯度损失统计
        # vf_loss_stats: 价值函数损失统计
        # vf_clipfrac_stats: 价值函数裁剪比例统计
        # entropy_stats: 策略熵统计,用于监控探索程度
        # ratio_stats: 新旧策略概率比统计
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)  # 获取一个 batch
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                allanswer = data["answer"] 
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []  # pi(a|s)
                ref_logprobs = []  # pi_ref(a|s)
                scores = []  
                sequence_lengths = []  # 每个样本的实际长度
                values = []
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,  
                        processing_class.pad_token_id,
                        generation_config,
                    )

                # 按批次大小遍历所有查询
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    # 获取当前批次的查询、答案和模型响应
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    sub_answer = allanswer[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    # 提取模型生成的响应部分(去掉输入上下文)
                    response = query_response[:, context_length:]   # shape = (batch_size, seq_len)
                    # 获取当前批次的logits
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    # 计算log概率分布
                    all_logprob = F.log_softmax(logits, dim=-1)
                    # 提取实际生成token的log概率
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    # 清理内存
                    del logits, all_logprob
                    torch.cuda.empty_cache()

                    # 计算参考策略的log概率
                    if ref_policy is None:
                        # 如果没有参考策略,使用null上下文
                        with self.null_ref_context():
                            ref_output = forward(model.policy, query_response, processing_class.pad_token_id)
                    else:
                        # 使用参考策略计算
                        ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    # 提取参考策略的logits并应用温度缩放
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    # 计算参考策略的log概率分布
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    # 提取实际生成token的参考log概率
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    # 清理内存
                    del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    # 计算每个序列的实际长度(不包括padding部分)
                    # first_true_indices返回每个序列中第一个pad_token_id的位置,减1得到实际长度
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    
                    # 从模型中提取value_model部分用于计算价值
                    # accelerator.unwrap_model用于获取原始模型(去掉分布式训练包装)
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    
                    # 使用value_model计算每个token的价值
                    # context_length用于区分输入和生成部分
                    full_value, _, _ = get_reward(
                        unwrapped_value_model, query_response, processing_class.pad_token_id, context_length
                    )
                    # 使用 context_length - 1 : -1 的原因:
                    # 1. context_length - 1: 从输入序列的最后一个token开始,因为value_model需要基于前一个token预测当前token的价值
                    # 2. -1: 去掉最后一个token,因为最后一个token没有下一个token可以预测
                    # 这样确保value的预测与生成序列的每个token一一对应
                    value = full_value[:, context_length - 1 : -1].squeeze(-1) # shape = (batch_size, seq_len)
                    score = get_reward_o1(
                        reward_model, postprocessed_response, processing_class, self.reward_processing_class, processing_class.pad_token_id, sub_answer
                    )  # shape = (batch_size, 1)

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)  # 奖励
                    values.append(value)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                # 检查每个序列是否包含结束标记(EOS token)
                # torch.any()在最后一个维度上检查是否存在EOS token
                # 返回一个布尔张量,表示每个序列是否包含EOS token
                contain_eos_token = torch.any(postprocessed_responses == self.processing_class.eos_token_id, dim=-1) # shape = (batch_size,)
                
                # 如果设置了缺失EOS token的惩罚值
                if self.args.missing_eos_penalty is not None:
                    # 对不包含EOS token的序列进行惩罚
                    # ~contain_eos_token 选择不包含EOS token的序列
                    # 从这些序列的分数中减去惩罚值
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                
                # 创建一个索引矩阵，用于标记每个序列中的位置
                # responses.shape[1]是序列长度，responses.shape[0]是批次大小
                # 例如，如果序列长度为5，批次大小为2，则response_idxs为:
                # [[0,1,2,3,4],
                #  [0,1,2,3,4]]
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                
                # 创建padding mask，用于标记超出实际序列长度的位置
                # sequence_lengths.unsqueeze(1)将序列长度扩展为列向量
                # 例如，如果sequence_lengths为[3,4]，则padding_mask为:
                # [[False,False,False,True,True],
                #  [False,False,False,False,True]]
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                
                # 将padding位置的logprobs和ref_logprobs设置为INVALID_LOGPROB
                # 这样可以确保这些位置不会参与后续的计算
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
                
                # 序列长度加1，用于计算value的padding mask
                # 因为value是预测下一个token的价值，所以需要多一个位置
                sequence_lengths_p1 = sequence_lengths + 1
                
                # 创建value的padding mask
                # 与logprobs的mask类似，但多了一个位置
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                
                # 将padding位置的value设置为0
                # 这样可以确保这些位置不会影响优势函数的计算
                values = torch.masked_fill(values, padding_mask_p1, 0)


                '''
                让我详细解释一下为什么需要 `response_idxs > (sequence_lengths_p1.unsqueeze(1))`：

                在PPO训练中，value网络预测的是每个位置的下一个token的价值。这意味着对于长度为n的序列，我们需要n+1个value值：
                - 前n个value值对应序列中每个位置的下一个token的价值
                - 最后一个value值对应序列结束后的价值（通常为0）

                举个例子：
                假设我们有一个序列长度为3的样本，那么：
                - `sequence_lengths = 3`
                - `sequence_lengths_p1 = 4` (因为需要多一个位置)
                - `response_idxs` 可能是 `[[0,1,2,3,4], [0,1,2,3,4]]` (假设最大长度为5)
                - `sequence_lengths_p1.unsqueeze(1)` 变成 `[[4], [4]]`

                当执行 `response_idxs > (sequence_lengths_p1.unsqueeze(1))` 时：
                - 对于第一个样本，会得到 `[False,False,False,False,True]`
                - 这表示前4个位置是有效的value值，第5个位置是padding

                这样做的原因是：
                1. 我们需要确保value预测覆盖到序列结束后的位置
                2. 同时要屏蔽掉超出实际需要的padding位置
                3. 这对于后续计算优势函数(advantage)和回报(return)是必要的
                '''

                # 4. 计算奖励值
                # 计算新旧策略的KL散度: log(π_new/π_old) = log(π_new) - log(π_old)
                kl = logprobs - ref_logprobs  
                
                # 将KL散度转换为非分数奖励,负号表示我们希望减少KL散度
                # kl_coef是控制KL惩罚强度的系数
                non_score_reward = -args.kl_coef * kl
                
                # 复制非分数奖励作为基础奖励
                rewards = non_score_reward.clone()
                
                # 获取每个序列的起始索引(0到batch_size-1)
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                
                # 计算每个序列的实际结束位置
                # 如果sequence_lengths_p1小于rewards的序列长度,使用sequence_lengths_p1
                # 否则使用sequence_lengths
                actual_end = torch.where(sequence_lengths_p1 < rewards.size(1), sequence_lengths_p1, sequence_lengths)
                
                # 在序列的起始和结束位置加上额外的分数奖励
                # 这通常用于鼓励模型生成更合理的序列
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                returns = advantages + values
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                # 将一整个 batch中的 index 打乱
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]
                            mb_return = returns[micro_batch_inds]
                            mb_values = values[micro_batch_inds]

                            # forward函数返回两个值:
                            # 1. output: 包含模型输出的完整信息
                            # 2. vpred_temp: 价值网络预测的原始价值,包含整个序列(包括context)的价值预测 【这里使用 policy model 来生成价值，之前的价值 values 是使用 ref model 生成的】
                            output, vpred_temp = forward(model, mb_query_responses, processing_class.pad_token_id)
                            
                            # 获取response部分的logits并应用temperature缩放
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            
                            # 计算新的策略分布
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)   # shape = (batch_size, seq_len)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            
                            # 由于 value model 是基于前n-1个token来预测第n个token的价值，
                            #    所以我们并不需要提取最后一个 token 对应的价值分数（因为这个分数（如果有的话）对应的是第 n+1 个 token的价值（n 是序列长度））
                                    # 因此我们需要使用 vpred_temp[:, context_length - 1 : -1]
                            # 从vpred_temp中提取response部分的价值预测
                            # vpred_temp包含整个序列的价值,我们只需要response部分
                            vpred = vpred_temp[:, context_length - 1 : -1].squeeze(-1)
                            vpred = torch.masked_fill(vpred, padding_mask_p1[micro_batch_inds], 0)
                            
                            # 对价值预测进行裁剪,防止与旧价值差异过大
                            vpredclipped = torch.clamp(
                                vpred,
                                mb_values - args.cliprange_value,
                                mb_values + args.cliprange_value,
                            )
                            vf_losses1 = torch.square(vpred - mb_return)  # shape = (batch_size, seq_len)
                            vf_losses2 = torch.square(vpredclipped - mb_return)
                            vf_loss_max = torch.max(vf_losses1, vf_losses2)
                            vf_loss = 0.5 * masked_mean(vf_loss_max, ~padding_mask_p1[micro_batch_inds])
                            vf_clipfrac = masked_mean(
                                (vf_losses2 > vf_losses1).float(), ~padding_mask_p1[micro_batch_inds]
                            )
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)  # shape = (batch_size, seq_len)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
                            loss = pg_loss + args.vf_coef * vf_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                # 计算策略梯度裁剪比例 - 统计有多少比例的样本被裁剪了
                                pg_clipfrac = masked_mean(
                                    (pg_losses2 > pg_losses).float(), ~padding_mask[micro_batch_inds]
                                )
                                
                                # 计算策略分布的概率分布
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)  # shape = (batch_size, seq_len, vocab_size)
                                
                                # 计算策略的熵 - 用于衡量策略的随机性/不确定性
                                # 两种等价的熵计算公式:
                                # 1. 标准形式: H = -sum(p * log(p))
                                # 2. 数值稳定形式: H = log(sum(exp(logits))) - sum(p * logits)
                                # 证明:
                                # p = softmax(logits) = exp(logits) / sum(exp(logits))
                                # log(p) = logits - log(sum(exp(logits)))
                                # H = -sum(p * log(p))
                                #   = -sum(p * (logits - log(sum(exp(logits)))))
                                #   = -sum(p * logits) + log(sum(exp(logits))) * sum(p)
                                #   = log(sum(exp(logits))) - sum(p * logits)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                
                                # 计算近似的KL散度 - 用于衡量新旧策略的差异
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                
                                # 记录各种统计指标
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                vf_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = vf_loss
                                vf_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    vf_clipfrac
                                )
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = ratio.mean()
                        gradient_accumulation_idx += 1   # micro batch 计数
                    minibatch_idx += 1   # mini batch 计数
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, vpred_temp, logits, new_all_logprobs, new_logprobs, vpred, vpredclipped,
                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,
                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["loss/value_avg"] = self.accelerator.gather(vf_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores, metrics, non_score_reward
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

        # 在训练结束时调用回调处理器的on_train_end方法
        # 这个方法会处理训练结束时的各种清理和收尾工作
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        
        # 如果控制标志指示需要保存模型
        if self.control.should_save:
            # 保存模型检查点,不包含trial和metrics信息
            self._save_checkpoint(model, trial=None, metrics=None)
            # 调用回调处理器的on_save方法处理保存后的操作
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )
                    table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `List[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            trainer_name="PPO",
            trainer_citation=citation,
            paper_title="Fine-Tuning Language Models from Human Preferences",
            paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
