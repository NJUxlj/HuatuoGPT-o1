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
    --num_sample_generations -1 \
    --report_to wandb
```

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
