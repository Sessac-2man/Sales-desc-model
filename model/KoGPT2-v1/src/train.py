from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch


import torch
import torch.nn.functional as F
import numpy as np

class Train:
    def __init__(self):
        base_model = "skt/kogpt2-base-v2"
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(base_model,
                                                                 bos_token='<s>',
                                                                 eos_token='</s>',
                                                                 pad_token='<pad>',
                                                                 unk_token='<unk>',
                                                                 mask_token='<mask>'
                                                                 )
        self.model = GPT2LMHeadModel.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)


    def model_train(self, epochs,
                    train_batch,
                    vali_batch,
                    train_dataset,
                    eval_dataset,
                    output_dir,
                    learning_rate):

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch,
            per_device_eval_batch_size=vali_batch,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            eval_strategy="steps",
            learning_rate=learning_rate,
            eval_steps=500,
            logging_dir='./logs',
            fp16=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )


        print(f"setting batch_size train: {training_args.per_device_train_batch_size}")
        print(f"setting batch_size eval: {training_args.per_device_eval_batch_size}")
        print(f"setting num_train epochs : {training_args.num_train_epochs}")
        print(f'device settings : {self.device}')
        trainer.train()


        trainer.save_model("output_dir")
        self.tokenizer.save_pretrained("output_dir")
