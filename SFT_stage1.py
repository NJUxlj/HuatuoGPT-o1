import os
import json
import torch
import logging
import argparse

from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import wandb
from accelerate import Accelerator
from transformers import set_seed, get_cosine_schedule_with_warmup
import shutil
import json
import traceback
from jinja2 import Template

from transformers import AutoModelForCausalLM, AutoTokenizer
os.umask(0)


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')


class Train_dataset(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        with open(config.data_path) as f:
            self.data = json.load(f)
        
        newdata = []
        for da in self.data:
            newdata.append(da)
        print('过滤掉',len(self.data),len(newdata))
        self.data = newdata

        self.max_seq_len = self.config.max_seq_len 
        self.debug = 0

        # 如果从Base LLMs训练，选择 llama3-instruct作为模版
        chat_template_llama3 = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        if not tokenizer.chat_template:
            tokenizer.chat_template = chat_template_llama3
            
        self.template = Template(tokenizer.chat_template)

    def __getitem__(self, index):
        return self.data[index]

    def get_response(self,da):
        temp = '## Thinking\n\n{}\n\n## Final Response\n\n{}'
        return temp.format(da['Complex_CoT'],da['Response'])


    def get_prompt(self,da):

        q = da['Question']
        a = self.get_response(da)
        assert q is not None and a is not None, f'q:{q} a:{a}'

        input =  self.template.render(messages=[{"role": "user", "content": q},{"role": "assistant", "content": a}],bos_token=self.tokenizer.bos_token,add_generation_prompt=False)
        input_ids = self.tokenizer.encode(input,add_special_tokens= False)

        query = self.template.render(messages=[{"role": "user", "content": q}],bos_token=self.tokenizer.bos_token,add_generation_prompt=True)
        query_ids = self.tokenizer.encode(query,add_special_tokens= False)

        labels = [-100]*len(query_ids) + input_ids[len(query_ids):]
        assert len(labels) == len(input_ids)
        # 截取最后max_seq_len长度的序列，因为:
        # 1. 模型有最大序列长度限制，需要截断过长序列
        # 2. 保留后面部分是因为这包含了response，即模型需要学习生成的部分
        # 3. labels对应input_ids，所以也要同样截断
        return {"input_ids": input_ids[-self.max_seq_len:], "labels": labels[-self.max_seq_len:]}

    def collate_fn(self, batch):
        data = [ self.get_prompt(da) for da in batch]
        input_ids = [item["input_ids"] for item in data]
        labels = [item["labels"] for item in data]
        max_len = max(len(x) for x in input_ids)
        max_len = min(max_len,self.max_seq_len)

        # 先截断，再补齐
        input_ids = [ item[:max_len] + [self.tokenizer.eos_token_id]*(max_len-len(item)) for item in input_ids]
        labels = [ item[:max_len] + [-100]*(max_len-len(item)) for item in labels]
        if self.debug < 3:
            print('input_ids',self.tokenizer.decode(input_ids[-1]))
            print('labels',self.tokenizer.decode([0 if x == -100 else x for x in labels[-1]]))
            self.debug += 1

        return {
                "input_ids": torch.LongTensor(input_ids),
                "labels": torch.LongTensor(labels),
            }
    
    def __len__(self):
        return len(self.data)

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0  
        self.right = torch.Tensor([0]).to(device=device)  # 各GPU上预测正确的token总数
        self.total = torch.Tensor([0]).to(device=device)  # 各GPU上有效token（非padding）的总数
        self.total_loss = torch.Tensor([0]).to(device=device) # 各GPU上的损失值总和
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        '''
        在一个 batch 上进行了各种指标的计算

        技术原理：
            输入输出错位：语言模型训练时，输入序列为[t0,t1,t2]，需要预测[t1,t2,t3]
            [:-1]：截取输入序列的前n-1个token（作为模型看到的上下文）
            [1:]：截取标签的后n-1个token（作为模型需要预测的目标）
            具体示例： 原始输入: [BOS, "Hello", "world", EOS]
            模型输入: [BOS, "Hello", "world"] (通过[:-1]获得)
            模型预测：["Hello", "world", EOS]（.argmax(dim=-1)在词汇表维度取概率最大的token ID）
            预测目标: ["Hello", "world", EOS] (通过[1:]获得)

        实现效果：
            确保模型在预测第i个token时，只能看到前i-1个token
            符合语言模型的自回归生成特性
            避免模型"偷看"当前要预测的token
            
            结合掩码机制：
                -100标签会通过.masked_fill()被忽略
                最终只计算有效token位置的损失和准确率
        '''
        self.n_step += 1
        with torch.no_grad():  # 禁用梯度计算
            # len(labels) == len(input_ids) == len(preds)
            # 序列偏移处理：通过[:-1]和[1:]实现输入-输出的错位对齐
            shift_preds = logits[..., :-1, :].argmax(dim=-1)  #shape = [batch_size, seq_len-1]  # 取除最后一个token外的所有
            shift_labels = labels[..., 1:]  # 忽略第一个token的标签(对应输入序列)
            '''
            含义：这行代码通过对 logits 进行处理获取预测的令牌。logits 是模型输出的原始分数，表示每个token的预测概率（通常是通过softmax函数获得）。这里通过[..., :-1, :]将 logits 的最后一个时间步的数据移除，因为我们想要预测的是下一个令牌。
                argmax(dim=-1)：在最后一个维度（每个token的预测概率）中找到具有最高概率的token索引。结果 shift_preds 的形状为 [batch_size, seq_len-1]。
            '''
            self.right += (shift_preds == shift_labels)   # 统计预测正确的token
            .masked_fill(shift_labels.eq(-100), 0)   # 忽略标签为-100的位置(padding/输入部分)
            .sum().item()

            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        # 全聚合
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)  #  聚合所有GPU的正确预测数
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)  # 聚合所有GPU的有效token总数  
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)  # 聚合所有GPU的损失总和

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss


def train(args):

    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=args.gradient_accumulation_steps) 

    if accelerator.is_main_process:
        wandb.init(project = args.experiment_name, config=args, dir=args.log_dir, mode="offline")
    
    accelerator.print(f'args:\n{args}')

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = args.train_bsz_per_gpu * dist.get_world_size() * accelerator.gradient_accumulation_steps

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    # open gradient checkpointing
    model.gradient_checkpointing_enable()  
    
    '''
    核心作用：通过牺牲计算时间换取显存节省
        默认情况下，前向传播需要保存所有中间激活值用于反向传播
        开启后只保存部分关键激活，其余在反向传播时重新计算
    '''

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },  # # 对普通参数应用衰减
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },  # 对偏置和LayerNorm参数禁用衰减
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = Train_dataset(args, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_bsz_per_gpu, shuffle=True, drop_last=True, collate_fn=train_dataset.collate_fn)

    num_training_steps = int(len(train_dataloader) * (args.n_epochs)) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_rates * num_training_steps), num_training_steps=num_training_steps)
    accelerator.print(f'gradient_accumulation_steps:{accelerator.gradient_accumulation_steps} data_path:{args.data_path} lr:{args.learning_rate} num_training_steps:{num_training_steps}')
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    start_epoch = 0
    start_step = 0
    global_step = 0

    metric = SFTMetric(device=torch.cuda.current_device())  # 确保指标计算发生在当前GPU设备上，通过 dist.all_reduce 实现跨GPU的指标聚合


    def save_checkpoint(epoch, step, global_step):
        save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
        if accelerator.is_main_process:
            checkpoint_files = os.listdir(args.output_dir)
            checkpoint_files = [file for file in checkpoint_files if file.startswith("checkpoint-")]
            num_checkpoints = len(checkpoint_files)
            if args.max_ckpts>0:
                if num_checkpoints >= args.max_ckpts:
                    checkpoint_files.sort(key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
                    oldest_checkpoint = checkpoint_files[0]
                    shutil.rmtree(os.path.join(args.output_dir, oldest_checkpoint))        
            os.makedirs(save_dir, exist_ok=True)
            output_dir = os.path.join(save_dir, 'tfmr')
            if accelerator.state.deepspeed_plugin.zero_stage!=3:
                model.save_pretrained(output_dir,state_dict=accelerator.get_state_dict(model))
            tokenizer.save_pretrained(output_dir)
            copy_files = []
            for item in os.listdir(args.model_path):
                if os.path.exists(os.path.join(output_dir,item)):
                    continue
                if item.startswith("pytorch_model") and item.endswith(".bin"):
                    continue
                if item.endswith(".index.json") or item.endswith(".safetensors"):
                    continue
                s = os.path.join(args.model_path, item)
                if os.path.isfile(s):
                    shutil.copy(s, os.path.join(output_dir,item))
                copy_files.append(item)
            print(f'huggingface model save in {output_dir}, copy file:{copy_files}')

        if accelerator.state.deepspeed_plugin.zero_stage==3:
            unwrap_model = accelerator.unwrap_model(model)
            unwrap_model.save_pretrained(os.path.join(save_dir, f'tfmr'),is_main_process=accelerator.is_main_process,save_function=accelerator.save,state_dict=accelerator.get_state_dict(model))
            
        accelerator.wait_for_everyone()
        accelerator.save({"epoch": epoch, "step": step, "global_step": global_step}, os.path.join(save_dir, "training_state.pt"))
        accelerator.print(f'checkpoint checkpoint-{epoch}-{global_step} is saved...')

    accelerator.print(accelerator.deepspeed_config)
    model.train()

    for epoch in range(start_epoch, args.n_epochs):
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if accelerator.is_main_process else enumerate(train_dataloader)
        for batch_cnt, batch in train_dataloader_iterator:
            if epoch==start_epoch and batch_cnt<start_step:
                continue

            if batch_cnt == 1 and epoch == 0:
                torch.cuda.empty_cache()

            input_ids=batch['input_ids']
            labels=batch['labels']

            output = model(input_ids=input_ids, labels=labels, return_dict=True,use_cache=False)
            loss = output.loss

            metric(output.logits, labels, loss)
            acc, train_loss = metric.get_metric()
            accelerator.backward(loss)
            if (global_step+1) % accelerator.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=epoch, current_step=batch_cnt, total_step=len(train_dataloader), skip=accelerator.optimizer_step_was_skipped, loss=round(train_loss, 3), acc=round(acc, 3), length=len(input_ids[0]), lr=lr_scheduler.get_last_lr()[0])

            if global_step % 3 == 0 and accelerator.is_main_process:
                wandb.log({
                    'skip': int(accelerator.optimizer_step_was_skipped),
                    'loss': train_loss,
                    'acc': acc,
                    'lr': lr_scheduler.get_last_lr()[0]
                }, step=global_step)

        accelerator.wait_for_everyone()
        save_checkpoint(epoch, batch_cnt, global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args of sft')
    # Experiment Args
    parser.add_argument('--experiment_name', type=str,default='sft_stage1')

    # Model Args
    parser.add_argument('--model_path', required=True, type=str)

    # Data Args
    parser.add_argument('--data_path', required=True, type=str)

    # Training Args
    parser.add_argument('--output_dir', default='./ckpts', type=str)
    parser.add_argument('--max_ckpts', default=2, type=int)
    parser.add_argument('--log_dir', default='./train_logs', type=str)
    parser.add_argument('--max_seq_len', default=8192, type=int)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int)
    parser.add_argument('--train_bsz_per_gpu', default=2, type=int)
    parser.add_argument('--weight_decay', default=0.1, type=float)
    parser.add_argument('--learning_rate', default=5e-6, type=float)
    parser.add_argument('--warmup_rates', default=0.05, type=float)
    parser.add_argument('--n_epochs', default=3, type=int)

    # Other Args
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir,args.experiment_name)
    args.output_dir = os.path.join(args.output_dir,args.experiment_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    train(args)           
