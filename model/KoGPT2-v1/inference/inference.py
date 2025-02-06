import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class KoGPT2Generator:
    def __init__(self, model_name="../output_dir", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def generate_text(self, prompt, max_length=140, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.2,
                      do_sample=True, num_candidates=5):
        """
        KoGPT2를 사용하여 여러 개의 텍스트 후보를 생성하는 함수.

        Returns:
        - 생성된 텍스트 리스트 (list)
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # 🔹 여러 개의 후보 생성 (num_return_sequences 사용)
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            do_sample=do_sample,
            num_return_sequences=num_candidates,  # 🔹 여러 개의 응답 생성
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        # 🔹 후보 텍스트 정리
        candidates = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)

            # 🔹 "."으로 문장 분리 후, 공백 제거
            sentences = generated_text.split(".")
            cleaned_text = [s.strip() for s in sentences if s.strip()]

            # 🔹 마지막 문장이 너무 짧으면 삭제
            if cleaned_text and len(cleaned_text[-1]) < 5:
                cleaned_text.pop()

            # 🔹 "."을 다시 붙여서 문장 연결
            final_text = ". ".join(cleaned_text)

            # 🔹 마지막 문장이 "." 없이 끝나면 추가
            if not final_text.endswith("."):
                final_text += "."

            candidates.append(final_text)

        return candidates  # 🔹 여러 개의 응답 반환

# ✅ 사용 예시
if __name__ == "__main__":
    generator = KoGPT2Generator()

    prompt = "상품명: 바닐라아이스크림\n설명:"
    results = generator.generate_text(prompt, num_candidates=5)

    print("🔥 생성된 텍스트 후보들:")
    for idx, text in enumerate(results, 1):
        print(f"[{idx}] {text}")
