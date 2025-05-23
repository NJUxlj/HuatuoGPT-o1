import os
import warnings
from dataclasses import dataclass
import wandb
import torch
from datasets import load_dataset,load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer,PreTrainedTokenizerBase
import json,random


from trl import (
    ModelConfig,
    ScriptArguments
)

from ppo_utils.ppo_config_medo1 import PPOConfig
from ppo_utils.ppo_trainer_medo1 import PPOTrainer


os.environ["WANDB_MODE"] = "offline"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'



from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser
)

class ppo_dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length = 1000,debug = 0):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
    
        newdata = []
        for da in self.data:
            if len(da['Open-ended Verifiable Question']) > 0 and len(da['Ground-True Answer']) > 0:
                newdata.append({'question':da['Open-ended Verifiable Question'],'answer':da['Ground-True Answer']})
        print(len(self.data),' -> ',len(newdata))
        self.data = newdata

        self.debug = debug     

    def __getitem__(self, index):
        return self.data[index]

    def get_prompt(self,da):
        message = [{"role": "user", "content": da['question']}]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        '''
        add_generation_prompt 参数是Hugging Face Transformers库中 apply_chat_template 方法的一个布尔参数，它的作用是：

        1. 功能 ：控制在生成的提示文本中是否添加一个特殊的生成提示标记
        2. 典型用途 ：当设置为True时，会在对话模板末尾添加一个标记（如 <|assistant|> ），表示接下来是模型需要生成回复的部分
        3. 效果 ：帮助模型识别生成回复的起始位置，提高生成质量
        4. 默认值 ：通常默认为False
        '''
        input_token = self.tokenizer(
            prompt,
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )  # 不要做任何的规范化处理，直接转成 token id

        da['input_ids'] = input_token["input_ids"]
        return da

    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]  # 在 da 中加入 input_ids 字段
        input_ids = [item["input_ids"] for item in data]
        question = [item["question"] for item in data]
        answer = [item["answer"] for item in data]

        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len,self.max_length)  # 防止数据集中的最大prompt长度超出模型上下文长度限制
        input_ids = [ [self.tokenizer.pad_token_id]*(max_len-len(item)) + item[:max_len] for item in input_ids]   # 先截断，然后再补齐到最大长度

        if self.debug > 0:   # 检查数据集中的倒数 “debug” 个样本
            print('[input_ids]',self.tokenizer.decode(input_ids[-1])) 
            print('[question]',question[-1])
            print('[answer]',answer[-1])
            self.debug -= 1
        return {
                "input_ids": torch.LongTensor(input_ids),  # List[List] -> LongTensor
                "question": question,
                "answer": answer
            }

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_into_dataclasses()  # 将命令行参数解析到对应的数据类实例中
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    '''
    - 第一行创建了一个HfArgumentParser解析器，用于解析三个数据类(ScriptArguments, PPOConfig, ModelConfig)
    - 第二行调用parse_args_into_dataclasses方法，将命令行参数解析到对应的数据类实例中
    - 第三行设置了梯度检查点的参数，use_reentrant=False表示使用非重入式的梯度检查点实现
    '''

    output_dir = training_args.output_dir
    run_name = training_args.run_name
    if run_name not in output_dir:  #   确保输出目录名包含任务名称
        output_dir = os.path.join(output_dir,run_name)
        training_args.output_dir = output_dir
    
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(    # 每个输出序列对应一个奖励分数
        training_args.reward_model_path, attn_implementation="flash_attention_2",num_labels=2
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(  # 用于输出在 token t 时刻， 对未来的回报分数的期望 （预估模型输出整个序列后的质量分数）
        training_args.value_model_path, trust_remote_code=model_config.trust_remote_code, attn_implementation="flash_attention_2",num_labels=1
    )
    # 参考模型：就是每一轮 epoch 开始时的基准模型
    ref_policy = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,attn_implementation="flash_attention_2")

    # 策略模型：ppo loss 需要直接更新的模型
    policy = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path,attn_implementation="flash_attention_2")

    reward_tokenizer = AutoTokenizer.from_pretrained(training_args.reward_model_path)

    if '<|eot_id|>' in tokenizer.vocab:
        assert '<|end_of_text|>' in tokenizer.vocab
        tokenizer.pad_token = '<|end_of_text|>'
        tokenizer.pad_token_id = tokenizer.encode('<|end_of_text|>',add_special_tokens=False)[0]
    assert tokenizer.pad_token_id != tokenizer.eos_token_id

    training_args.stop_token_id = tokenizer.eos_token_id

    eval_ratio = 0.1
    eval_max_num = 200
    with open(script_args.dataset_name) as f: # 加载数据集
        data = json.load(f)

    random.shuffle(data)
    eval_num = min(int(len(data) * eval_ratio),eval_max_num)
    train_dataset = ppo_dataset(data[eval_num:],tokenizer, debug = 1)  # 打印最后一条
    eval_dataset = ppo_dataset(data[:eval_num],tokenizer)

    trainer = PPOTrainer(
        config=training_args,
        processing_class=tokenizer,
        reward_processing_class = reward_tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator = train_dataset.collate_fn
    )
    trainer.train()