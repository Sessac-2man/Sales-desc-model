from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

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

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        auc = roc_auc_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auroc': auc
        }

    def model_train(self, epochs,
                    batch_size,
                    train_dataset,
                    eval_dataset,
                    output_dir,
                    learning_rate):

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            eval_strategy="steps",
            learning_rate=learning_rate,
            eval_steps=500,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        trainer.save_model("output_dir")
        self.tokenizer.save_pretrained("output_dir")
