from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
UTILS_DIR = BASE_DIR / "utils"
sys.path.insert(0, str(UTILS_DIR))

import config as cfg
from deep_translator import GoogleTranslator
from device import get_device #type: ignore
from encode import encode_to_device #type: ignore
import torch
from train import set_train #type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer

class JagalGpt:

    def __init__(self):
        self.device = get_device()
        self.dialogue = ""
        print(f"able device: {self.device}")
    
    def torch_device(self):
        pass

    def load_model(self):
        # gpt2 : 117M(파라미터 약 1.17억 개), 500MB
        # gpt2-medium : 345M(파라미터 약 3.45억 개), 1.5GB
        # gpt2-large : 774M(파라미터 약 7.74억 개), 3GB
        # gpt2-xl : 1.5 billion(파라미터 약 15억 개), 6GB
        # gpt-3 : 175 billion, 약 350GB
        # gpt-4 : 파라미터 수와 모델 크기에 대한 공식적인 정보가 공개되지 않았음

        try:
            model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, cache_dir=cfg.CACHE_DIR)
            tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, cache_dir=cfg.CACHE_DIR)

            model.to(self.device) # default: CPU 연산
            model = set_train(model, False) # 모델을 학습모드에서 추론모드로 전환

            return model, tokenizer, None
        except Exception as e:
            return None, None, e
    
    def generate_response(self, model, tokenizer):
        input_ids = encode_to_device(tokenizer, self.dialogue, self.device)

        embedding_output = model.transformer.wte(input_ids)
        print(f"Embedding output: {embedding_output}")
        print(f"Embedding shape: {embedding_output.shape}")

        # 토큰 제한을 둬서 최신 대화만 유지 필요
        max_length = input_ids.shape[1] + cfg.TOKEN_LENGTH  # 입력 길이 + 생성할 길이
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(self.device)
        pad_token_id = tokenizer.eos_token_id

        print(f"Token length: {input_ids.shape[1]}")

        with torch.no_grad(): # 그래디언트 계산을 비활성화하여 추론 중 메모리를 절약하고 속도를 높임
            outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1, attention_mask=attention_mask, pad_token_id=pad_token_id, do_sample=True)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    def reset_history(self) -> None:
        self.dialogue = ""
    
    def append_history(self, text) -> None:
        self.dialogue += f"{text}\n"

    def main(self):
        model, tokenizer, e = self.load_model()

        if model is None or tokenizer is None:
            print(f"Error loading model or tokenizer: {e}")
            exit()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit"]:
                break
            elif user_input.lower() == "reset":
                self.reset_history()
                print("Dialogue history has been reset.")
                continue

            self.append_history(user_input)

            try:
                response = self.generate_response(model, tokenizer)

                ai_response = response[len(self.dialogue):]
                print(f"AI: {ai_response}")
                print(f"AI: {GoogleTranslator(source='auto', target='ko').translate(ai_response)}")

                self.append_history(ai_response)

            except Exception as e:
                print(f"Error occurred while running model: {e}")

if __name__ == "__main__":
    llm = JagalGpt()
    llm.main()
