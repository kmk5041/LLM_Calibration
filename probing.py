import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
import wandb
from typing import List, Dict


class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = (
            "You are a helpful assistant.\n"
            f"{sample['content']}\n"
            f"Predicted answer: {sample['predicted_answer']}\n"
            "Is the answer correct? Respond with yes or no."
        )
        
        label = sample['label']
        return {
            'prompt': prompt,
            "label": torch.tensor(label, dtype=torch.long),
        }


class MyCollator:
    def __init__(self, tokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        prompts = [item['prompt'] for item in batch]
        labels = torch.stack([item["label"] for item in batch], dim=0)

        # Tokenize the prompts
        encoding = self.tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        encoding['labels'] = labels

        return encoding



class LlamaWithClassifier(nn.Module):
    def __init__(self, model_name, output_dim=2):
        super().__init__()
        # Load and freeze Llama backbone
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,     # ← FP16으로 로드
            low_cpu_mem_usage=True,        # ← CPU 메모리 사용 최적화
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        hidden_size = self.backbone.config.hidden_size
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim, bias=True)  # Output dim is 2 for binary classification
            )

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        # Forward through Llama
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,  
            )
        # Take CLS-like representation: first token of last hidden state
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # Get the last token's hidden state
        logits = self.classifier(last_hidden).half()  # Convert to FP16
        return logits

class BinaryClssifierTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        logits= model(**inputs)
        loss=F.cross_entropy(logits, labels)
        if return_outputs:
            return loss, logits
        return loss

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
       

    # a) 데이터 로드
    data_path = "dataset_train.json"
    test_data_path = "dataset_test.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if local_rank==0:
        print(f"Train size: {len(data)}")
        print(f"Test size: {len(test_data)}")
    

    # b) 데이터셋과 토크나이저 준비
    MODEL_NAME = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collator=  MyCollator(tokenizer, max_length=512)

    train_dataset = QADataset(data, tokenizer)
    test_dataset = QADataset(test_data, tokenizer)


    model= LlamaWithClassifier(MODEL_NAME, output_dim=2)

    num_train_epochs = 25
     
    training_args = TrainingArguments(
        output_dir=f"probing/checkpoints/{num_train_epochs}_epochs",
        learning_rate=5e-5,        # 0.00001: 1e-4보다 10배 낮춰 안정성 확보
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,          # 전체 스텝의 10% 동안 워밍업
        weight_decay=0.01,         # 기본 L2 페널티
        per_device_train_batch_size=16,
        num_train_epochs=num_train_epochs,
        logging_dir="logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=num_train_epochs,
        eval_strategy="epoch",
        fp16=True if torch.cuda.is_available() else False,
        local_rank=local_rank,
        report_to="wandb" if local_rank == 0 else None,  # Only log to wandb from main process
        run_name=f"probing-{num_train_epochs}epochs",
        ddp_find_unused_parameters=False,  # DDP 환경에서 unused parameters 경고 방지
        remove_unused_columns=False,  # Hugging Face Trainer가 내부적으로 DDP 래퍼를 붙습니다.

    )

    trainer = BinaryClssifierTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()
