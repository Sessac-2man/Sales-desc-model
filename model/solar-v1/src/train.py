import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import SFTTrainer


class Train:
    def __init__(self):
        base_model = "beomi/OPEN-SOLAR-KO-10.7B"  # 🔹 원본 모델 사용

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # ✅ 올바른 설정
            bnb_4bit_compute_dtype=torch.float16,  # ✅ 4-bit 연산 타입 설정
            bnb_4bit_use_double_quant=True,  # ✅ 더블 양자화 적용 (VRAM 절약)
        )

        # 🔹 GPU 상태 확인
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"✅ 현재 사용 중인 디바이스: {self.device}")
        if self.device == "cuda":
            print(f"✅ 사용 가능한 GPU 개수: {torch.cuda.device_count()}")
            print(f"✅ 현재 사용 중인 GPU: {torch.cuda.get_device_name(0)}")

        # 🔹 4-bit 양자화된 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=bnb_config  # ✅ 올바른 설정
        )

        # 🔹 k-bit Training을 위한 준비
        self.model = prepare_model_for_kbit_training(self.model)

        # 🔹 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 🔹 LoRA 설정
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # ✅ LoRA 적용할 레이어
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 🔹 LoRA 적용
        self.model = get_peft_model(self.model, config)

    def model_train(self, epochs,
                    train_batch,
                    vali_batch,
                    train_dataset,
                    eval_dataset,
                    output_dir,
                    learning_rate):

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=train_batch,
            per_device_eval_batch_size=vali_batch,
            save_steps=500,  # 🔹 더 짧은 간격으로 체크포인트 저장
            save_total_limit=3,
            gradient_accumulation_steps=8,  # ✅ VRAM 최적화 (8GB 환경 고려)
            logging_steps=100,
            eval_strategy="steps",
            learning_rate=learning_rate,
            eval_steps=500,
            logging_dir='./logs',
            fp16=True,  # ✅ `fp16=True`로 변경 (RTX 4070은 bf16 미지원)
            report_to="none",
            gradient_checkpointing=True,  # ✅ VRAM 절약 (필수)
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        print(f"🔹 Batch Size (Train): {training_args.per_device_train_batch_size}")
        print(f"🔹 Batch Size (Eval): {training_args.per_device_eval_batch_size}")
        print(f"🔹 Num Train Epochs: {training_args.num_train_epochs}")

        trainer.train()

        # 🔹 LoRA Adapter 저장 (bnb 모델은 LoRA만 저장 가능)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print(f"✅ Fine-tuned LoRA 모델 저장 완료: {output_dir}")
