# HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs
<div align="center">
<h3>
  HuatuoGPT-o1
</h3>
</div>

<p align="center">
📃 <a href="https://arxiv.org/pdf/2412.18925" target="_blank">Paper</a> ｜🤗 <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-7B" target="_blank">HuatuoGPT-o1-7B</a> ｜🤗 <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-8B" target="_blank">HuatuoGPT-o1-8B</a> ｜ 🤗 <a href="https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-70B" target="_blank">HuatuoGPT-o1-70B</a>  | 📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT" target="_blank">Data</a>
</p>


## ⚡ Introduction
Hello! Welcome to the repository for [HuatuoGPT-o1](https://arxiv.org/pdf/2412.18925)!

<div align=center>
<img src="assets/pic1.jpg"  width = "90%" alt="HuatuoGPT-o1" align=center/>
</div>


**HuatuoGPT-o1** is a medical LLM designed for advanced medical reasoning. It can identify mistakes, explore alternative strategies, and refine its answers.  By leveraging verifiable medical problems and a specialized medical verifier, it advances reasoning through:

- Using the verifier to guide the search for a complex reasoning trajectory for fine-tuning LLMs.
- Applying reinforcement learning (PPO) with verifier-based rewards to enhance complex reasoning further.

We open-sourced our models, data, and code here.

## 👨‍⚕️ Model
- **Model Access**

|                      | Backbone     | Supported Languages | Link                                                                  |
| -------------------- | ------------ | ----- | --------------------------------------------------------------------- |
| **HuatuoGPT-o1-8B**  | LLaMA-3.1-8B  | English    | [HF Link](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-8B) |
| **HuatuoGPT-o1-70B** | LLaMA-3.1-70B | English    | [HF Link](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-70B) |
| **HuatuoGPT-o1-7B**  | Qwen2.5-7B   | English & Chinese | [HF Link](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-7B) |
| **HuatuoGPT-o1-72B** | Qwen2.5-72B  | English & Chinese | [HF Link](https://huggingface.co/FreedomIntelligence/HuatuoGPT-o1-72B) |

- **Deploy**

HuatuoGPT-o1 can be used just like `Llama-3.1-8B-Instruct`. You can deploy it with tools like [vllm](https://github.com/vllm-project/vllm) or [Sglang](https://github.com/sgl-project/sglang),  or perform direct inference:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("FreedomIntelligence/HuatuoGPT-o1-8B",torch_dtype="auto",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("FreedomIntelligence/HuatuoGPT-o1-8B")

input_text = "How to stop a cough?"
messages = [{"role": "user", "content": input_text}]

inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True
), return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

HuatuoGPT-o1 adopts a *thinks-before-it-answers* approach, with outputs formatted as:

```
## Thinking
[Reasoning process]

## Final Response
[Output]
```

## 📚 Data
- **Data Access**

| Data                  | Description                                                                                   | Link                                                                                           |
| -------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Medical Verifiable Problems | Open-ended medical problems sourced from challenging medical exams,  paired with ground-truth answers. | [Link](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem)  |
| SFT Data in Stage 1        | Fine-tuning data generated using GPT-4o, including complex chains of thought (**Complex CoT**) and output (**Response**). | [Link](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)       |

- **Data Construction**

We provide scripts to construct verifiable problems and searching reasoning paths.

**1. Constructing Verifiable Problems from Multi-choice Questions.** 
```bash
python construct_verifiable_medical_problems.py --data_path  data/demo_data.json --filter_data --model_name gpt-4o --api_key [your api key]
```
**2. Searching Complex Reasoning Paths for SFT**

```bash
python search_for_complex_reasoning_path.py --data_path  data/demo_data.json --efficient_search True  --max_search_attempts 1 --max_search_depth 2 --model_name gpt-4o --api_key [your api key]
```


## Dataset Download
```

```


## 🚀 Training

- **Stage 1: Supervised Fine-Tuning (SFT)**

Fine-tune the model on an 8-GPU setup:
```bash
accelerate launch --config_file ./configs/deepspeed_zero3.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard SFT_stage1.py \
    --model_path [meta-llama/Llama-3.1-8B-Instruct] \
    --data_path [FreedomIntelligence/medical-o1-reasoning-SFT] 
```

- **Stage 2: Reinforcement Learning (RL)**

We provide a simple PPO script using the [trl](https://github.com/huggingface/trl) library. Below is an example for training an 8B model with PPO on an 8-GPU A100 machine. Ensure you first download our [medical verifier](https://huggingface.co/FreedomIntelligence/medical_o1_verifier_3B) as the reward model.

```bash
accelerate launch \
	--num_processes 8 \
	--num_machines 1 \
	--machine_rank 0 \
    --config_file ./configs/deepspeed_zero3.yaml \
	--deepspeed_multinode_launcher standard RL_stage2.py \
    --model_name_or_path [FreedomIntelligence/HuatuoGPT-o1-8B] \
    --reward_model_path [FreedomIntelligence/medical_o1_verifier_3B] \
    --value_model_path [meta-llama/Llama-3.2-3B-Instruct] \
    --dataset_name  [FreedomIntelligence/medical-o1-verifiable-problem]\
    --response_length 1300 \
    --temperature 0.5 \
    --local_rollout_forward_batch_size 8 \
    --num_ppo_epochs 3 \
    --num_mini_batches 1 \
    --total_episodes 20000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --bf16 True \
    --output_dir ./ckpts \
    --save_strategy steps \
    --save_step 20 \
    --save_total_limit 1 \
    --eval_strategy steps \
    --eval_steps 20 \
    --kl_coef 0.03 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.05 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ppo_medical_o1_8B \
    --num_sample_generations -1 \   # -1表示不限制采样数量，会使用数据集中的所有样本
    --report_to wandb
```

-  适用于Qwen2.5-1.5B-Instruct的脚本
- num_processes 的值， 必须和  local_rollout_forward_batch_size 一致
- num_mini_batches 等同于 per_device_train_batch_size

```
accelerate launch \
	--num_processes 4 \
	--num_machines 1 \
	--machine_rank 0 \
    --config_file ./configs/deepspeed_zero3.yaml \
	--deepspeed_multinode_launcher standard RL_stage2.py \
    --model_name_or_path /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
    --reward_model_path [FreedomIntelligence/medical_o1_verifier_3B] \
    --value_model_path /root/autodl-tmp/models/Qwen2.5-1.5B-Instruct \
    --dataset_name  /root/autodl-tmp/HuatuoGPT-o1/data/medical-o1-verifiable-problem \
    --response_length 1300 \
    --temperature 0.5 \local_rollout_forward_batch_sizelocal_rollout_forward_batch_size
    --local_rollout_forward_batch_size 4 \
    --num_ppo_epochs 3 \
    --num_mini_batches 1 \
    --total_episodes 20000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --bf16 True \
    --output_dir ./ckpts \
    --save_strategy steps \
    --save_step 20 \
    --save_total_limit 1 \
    --eval_strategy steps \
    --eval_steps 20 \
    --kl_coef 0.03 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.05 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ppo_medical_o1_8B \
    --num_sample_generations -1 \
    --report_to wandb
```


'''
## 生成控制参数详解

### `--response_length 1300`
**PPO算法角度**：这个参数直接影响**轨迹（trajectory）的长度**。在PPO中，每个episode包含一系列的状态-动作对，对于语言模型来说，每个token生成都是一个动作步骤。

- **轨迹收集**：PPO需要收集完整的轨迹来计算优势函数（advantage function）和价值估计
- **奖励计算**：更长的序列意味着更多的中间奖励和最终奖励信号
- **计算复杂度**：1300个token意味着每个episode有1300个决策步骤，显著增加计算开销
- **医疗应用考虑**：医疗诊断需要详细的推理过程，较长的响应长度有助于捕获完整的医疗思维链

### `--temperature 0.5`
**PPO算法角度**：Temperature参数控制**策略分布的熵（entropy）**，这在PPO中具有重要意义：

- **探索-利用平衡**：Temperature影响动作选择的随机性，0.5提供适中的探索水平
- **策略梯度计算**：较低的temperature使概率分布更尖锐，影响梯度的方差
- **KL散度影响**：Temperature直接影响新旧策略之间的KL散度计算
- **数学表达**：softmax概率计算为 `P(a|s) = exp(logits/T) / Σ exp(logits_i/T)`
- **训练稳定性**：0.5在保持一定随机性的同时避免过度探索导致的训练不稳定

### `--local_rollout_forward_batch_size 8`
**PPO算法角度**：这个参数控制**经验收集阶段**的批处理效率：

- **轨迹并行收集**：PPO需要收集大量的on-policy经验，批处理提高效率
- **内存管理**：8个并行rollout在内存使用和计算效率之间取得平衡
- **方差减少**：并行收集的轨迹提供更多样化的经验，减少梯度估计方差
- **硬件优化**：充分利用GPU的并行计算能力

## PPO训练参数详解

### `--num_ppo_epochs 3`
**PPO算法核心机制**：这是PPO相比传统策略梯度方法的**关键优势**之一。

- **多轮优化**：PPO允许对同一批经验进行多次更新，提高样本效率
- **理论基础**：通过剪切机制和KL散度约束，PPO可以安全地进行多轮更新而不会偏离太远
- **过拟合风险**：过多的epochs可能导致对特定批次数据的过拟合，3轮是经验上的平衡点
- **计算效率**：相比单次更新，多轮训练显著提高了样本利用率

### `--num_mini_batches 1`
**PPO算法角度**：控制**梯度更新的细粒度**：

- **内存效率**：将大批次分解为小批次，避免内存溢出
- **梯度方差**：较小的mini-batch可能增加梯度噪声，但在大模型训练中有助于泛化
- **优化动态**：1个mini-batch意味着每个epoch只有一次梯度更新
- **与other参数协同**：结合`gradient_accumulation_steps=16`，实际有效批次大小仍然足够大

### `--total_episodes 20000`
**PPO算法角度**：定义**总体训练规模**：

- **收敛性**：PPO通常需要大量的经验来收敛到最优策略
- **样本复杂度**：20000个episodes在大语言模型的PPO训练中是合理的规模
- **医疗领域特殊性**：医疗任务的复杂性可能需要更多的训练样本
- **计算资源考虑**：需要平衡训练效果和计算成本

### `--kl_coef 0.03`
**PPO算法核心**：这是PPO算法中**最关键的超参数**之一：

**理论基础**：
- **信任域约束**：KL散度系数实现了类似TRPO的信任域机制
- **目标函数**：完整的PPO目标函数为：`L = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)] - β*KL(π_old, π_new)`
- **其中β就是kl_coef参数**

**实际作用**：
- **策略稳定性**：0.03的系数确保新策略不会偏离旧策略太远
- **训练稳定性**：防止策略更新过大导致的性能崩溃
- **自适应调整**：许多实现会根据实际KL散度动态调整这个系数
- **经验值选择**：0.03是经过大量实验验证的合理取值

**数学细节**：
- **KL散度计算**：`KL(π_old||π_new) = Σ π_old(a|s) * log(π_old(a|s)/π_new(a|s))`
- **惩罚机制**：当KL散度过大时，损失函数会给予惩罚，迫使模型保持策略稳定性
- **平衡考虑**：过小的系数可能导致训练不稳定，过大的系数可能阻碍学习进展

## 参数间的协同作用

这些参数在PPO训练中协同工作：

1. **轨迹收集阶段**：`response_length`和`temperature`控制经验生成
2. **批处理优化**：`local_rollout_forward_batch_size`和`num_mini_batches`管理内存和计算
3. **策略更新**：`num_ppo_epochs`和`kl_coef`确保稳定且高效的学习
4. **整体规模**：`total_episodes`定义训练的总体范围

这种配置特别适合医疗领域的大语言模型训练，因为医疗任务需要：
- **详细推理**（长响应长度）
- **可靠性**（适中的温度和KL约束）
- **充分训练**（足够的episodes和epochs）
'''

## 🧐 Evaluation
1. You first need to install [Sglang](https://github.com/sgl-project/sglang). After installation, deploy the model you want to test using Sglang with the following command:
```bash
log_num=0
model_name="FreedomIntelligence/HuatuoGPT-o1-8B" # Path to the model you are deploying
port=28${log_num}35
CUDA_VISIBLE_DEVICES=0  python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 1 --tp 1  > sglang${log_num}.log 2>&1 &
```
```Plain Text
1. `log_num=0`:
   -  定义一个变量 `log_num` 并将其赋值为 0。这个变量将在后续的命令中使用，用于生成文件名和端口号，方便管理多个 sglang 服务实例。

2. `model_name="FreedomIntelligence/HuatuoGPT-o1-8B"`:
   -  定义一个变量 `model_name`，并将其赋值为 `"FreedomIntelligence/HuatuoGPT-o1-8B"`。这通常代表托管在 Hugging Face Hub 上的一个模型 ID 或本地模型的文件路径。`FreedomIntelligence/HuatuoGPT-o1-8B`  很可能是一个特定版本的或微调过的 LLM 名称。 sglang 会加载并使用这个指定的模型。

3. `port=28${log_num}35`:
   -  定义一个变量 `port`，并将其赋值为一个由字符串和变量组合成的数值。`${log_num}` 会被替换为 `log_num` 变量的值(0)，所以最终 `port` 的值为 `28035`。 这个端口号将用于 sglang 服务的监听，客户端可以通过这个端口与服务进行通信。使用变量可以让多个服务实例运行在不同的端口上，避免冲突。

4. `CUDA_VISIBLE_DEVICES=0`:
   -  这是一个环境变量设置。`CUDA_VISIBLE_DEVICES=0`  告诉 CUDA 只使用 GPU 设备 0。 如果系统有多个 GPU，这个设置可以控制 sglang 服务使用哪个 GPU。这对于多 GPU 服务器非常重要，可以进行资源分配和隔离。

5. `python -m sglang.launch_server`:
   -  调用 Python 解释器来执行 `sglang.launch_server` 模块。 `-m`  选项告诉 Python 将 `sglang.launch_server`  视为一个模块来运行。 `sglang.launch_server`  很可能是 sglang 库提供的用于启动服务的脚本。

6. `--model-path $model_name`:
   -  这是一个传递给 `sglang.launch_server`  的命令行参数。`--model-path`  指定了要加载的模型的路径或 ID。`${model_name}` 会被替换为之前定义的 `model_name`  变量的值。

7. `--port $port`:
   -  另一个传递给 `sglang.launch_server`  的命令行参数。`--port`  指定了服务监听的端口。`${port}` 会被替换为之前定义的 `port`  变量的值。

8. `--mem-fraction-static 0.8`:
   -  指定静态内存分配的比例。这告诉 sglang 服务预留 80% 的 GPU 内存，以供模型和相关操作使用。 静态分配可以提高性能，避免动态分配带来的开销。

9. `--dp 1 --tp 1`:
   - 这些参数控制数据并行（DP）和张量并行（TP）的设置。 `--dp 1`  和 `--tp 1`  都设置为 1 意味着禁用数据并行和张量并行。  当设置为1时，表示所有数据和张量都存储在一个设备上。 如果您有多个 GPU 并且想要加速推理过程，您可以增加这些值以启用并行处理。

10. `> sglang${log_num}.log 2>&1 &`:
    -  这一部分处理输出重定向和后台运行。
    -  `>`  符号将标准输出 (stdout) 重定向到名为 `sglang${log_num}.log`  的文件。由于 `log_num` 为 0，文件名将是 `sglang0.log`。
    -  `2>&1`  将标准错误 (stderr) 重定向到与标准输出相同的位置。这意味着错误信息也会被写入到 `sglang0.log`  文件中。
    -  `&`  符号将整个命令放在后台运行。 这意味着该命令将在后台启动，而不会阻塞当前的终端会话。

总而言之，这条命令启动了一个 sglang 服务，加载指定的 LLM (HuatuoGPT-o1-8B)，监听在端口 28035， 使用 GPU 0，设置静态内存分配，禁用并行处理，并将服务的输出和错误信息写入 `sglang0.log`  文件，并且在后台运行该服务。
```


2. Wait for the model to be deployed. After deployment, you can run the following code for evaluation. We use prompts that allow the model to respond freely. We find that the extracted results are consistently reliable and broadly cover the intended scope. You can also set the `--strict_prompt` option to use stricter prompts for more precise answer extraction.
```bash
python evaluation/eval.py --model_name $model_name  --eval_file evaluation/data/eval_data.json --port $port 
```
3. After completing the evaluation, run the following code to stop the Sglang service and release GPU memory.
```bash
bash evaluation/kill_sglang_server.sh
```
The evaluation code above can be used to test most models supported by Sglang.

## 🩺 HuatuoGPT Series 

Explore our HuatuoGPT series:
- [**HuatuoGPT**](https://github.com/FreedomIntelligence/HuatuoGPT): Taming Language Models to Be a Doctor
- [**HuatuoGPT-II**](https://github.com/FreedomIntelligence/HuatuoGPT-II): One-stage Training for Medical Adaptation of LLMs
- [**HuatuoGPT-Vision**](https://github.com/FreedomIntelligence/HuatuoGPT-Vision): Injecting Medical Visual Knowledge into Multimodal LLMs at Scale
- [**CoD (Chain-of-Diagnosis)**](https://github.com/FreedomIntelligence/Chain-of-Diagnosis): Towards an Interpretable Medical Agent using Chain of Diagnosis
- [**HuatuoGPT-o1**](https://github.com/FreedomIntelligence/HuatuoGPT-o1): Towards Medical Complex Reasoning with LLMs


## 📖 Citation
```
@misc{chen2024huatuogpto1medicalcomplexreasoning,
      title={HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs}, 
      author={Junying Chen and Zhenyang Cai and Ke Ji and Xidong Wang and Wanlong Liu and Rongsheng Wang and Jianye Hou and Benyou Wang},
      year={2024},
      eprint={2412.18925},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.18925}, 
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=FreedomIntelligence/HuatuoGPT-o1&type=Date)](https://star-history.com/#FreedomIntelligence/HuatuoGPT-o1&Date)
