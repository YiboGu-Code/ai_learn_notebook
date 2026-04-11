from datasets import Dataset      # Hugging Face 数据集工具。把你的数据变成模型能训练的格式
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
# DataCollatorForSeq2Seq：批量数据对齐（训练专用）；TrainingArguments：训练参数配置（学习率、epoch、batch…）；
# Trainer：训练器：一键开始训练；GenerationConfig：生成配置：温度、top_k…
from peft import LoraConfig, TaskType, get_peft_model


def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # pad_token_id：结束符；attention_mask 全是 1，表示全部有效
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    # 这是指令微调的关键：问题部分 → 全部设为 -100；PyTorch 会自动忽略 -100，不计算损失；只让模型学习回答部分
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct', device_map="auto",torch_dtype=torch.bfloat16)
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法
    tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct', use_fast=False, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 将JSON文件转换为Hugging Face 原生数据集格式
    df = pd.read_json('huanhuan.json')
    ds = Dataset.from_pandas(df)   # 将 JSON 转为 Hugging Face 原生数据集格式，支持懒加载与高效 map。
    # 对每一条数据执行 process_func 函数；删掉原来的文字列（instruction、input、output）
    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters() # 打印总训练参数

    # per_device_train_batch_size=4：每块显卡一次训练 4 条数据
    # gradient_accumulation_steps=4：梯度累积 4 步再更新一次（用小显存模拟大 batch）
    # logging_steps=10：每训练 10 步，打印一次日志
    # num_train_epochs=3：把所有数据完整训练 3 轮
    # save_steps=100：每训练 100 步，保存一次模型
    # learning_rate=1e-4：学习率
    # save_on_each_node=True：多卡训练用的，单卡不用管
    # gradient_checkpointing=True：开启梯度检查点；显存降低 60%~80%
    args = TrainingArguments(
        output_dir="./output/llama3_1_instruct_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=3,
        save_steps=100, # 为了快速演示，这里设置10，建议你设置成100
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_id,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    trainer.train() # 开始训练 
    # 在训练参数中设置了自动保存策略此处并不需要手动保存。

















