import faulthandler
faulthandler.enable()

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
UTILS_DIR = BASE_DIR / "utils"
sys.path.insert(0, str(UTILS_DIR))

import argparse
import config as cfg
import certifi
from datasets import load_dataset
from device import get_device #type: ignore
from encode import encode_to_device #type: ignore
from pretrained import load_model #type: ignore
import logging
import multiprocessing as mp
import os
from peft import LoraConfig, get_peft_model
import requests
import ssl
import torch
import traceback
from translator import any_to_english, english_to_korean #type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

class JagalGpt:

    def __init__(self, mode: str, dataset: str):
        self.mode = mode
        self.dataset = dataset
        self.device = get_device()
        self.dialogue = ""
        print(f"========== Able device: {self.device}")

        os.environ['SSL_CERT_FILE'] = certifi.where()
        print(f"========== SSL cert path: {ssl.get_default_verify_paths()}")

        print(f"========== Insecure HTTPS request : {requests.get('https://huggingface.co', verify=False)}")
    
    def generate_response(self, model, tokenizer) -> str:
        input_ids = encode_to_device(tokenizer, self.dialogue, self.device)

        if "gpt" in cfg.MODEL_NAME:
            embedding_output = model.transformer.wte(input_ids)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(self.device)

        elif "gemma" in cfg.MODEL_NAME:
            embedding_output = model.get_input_embeddings()(input_ids)
            attention_mask = torch.ones_like(input_ids).to(self.device)
        
        elif "llama" in cfg.MODEL_NAME:
            pass

        elif "phi" in cfg.MODEL_NAME:
            pass

        elif "mistralai" in cfg.MODEL_NAME:
            pass

        elif "bloom" in cfg.MODEL_NAME:
            pass

        elif "falcon" in cfg.MODEL_NAME:
            pass

        else:
            pass

        # 토큰 제한을 둬서 최신 대화만 유지 필요
        max_length = input_ids.shape[1] + cfg.TOKEN_LENGTH  # 입력 길이 + 생성할 길이

        print(f"========== Embedding output: {embedding_output}")
        print(f"========== Embedding shape: {embedding_output.shape}")
        print(f"========== Token length: {input_ids.shape[1]}")

        with torch.no_grad(): # 그래디언트 계산을 비활성화하여 추론 중 메모리를 절약하고 속도를 높임
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,

                # Length
                max_new_tokens=cfg.TOKEN_LENGTH,
                min_new_tokens=1,
                max_time=None,

                # Sampling
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                typical_p=0.95,
                epsilon_cutoff=0.0,
                eta_cutoff=0.0,

                # Quality
                repetition_penalty=1.1,
                encoder_repetition_penalty=1.0,
                no_repeat_ngram_size=3,
                renormalize_logits=True,
                remove_invalid_values=True,

                # Beam
                num_beams=1,
                num_beam_groups=1,
                diversity_penalty=0.0,
                early_stopping=False,
                length_penalty=1.0,

                # Token
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                forced_bos_token_id=None,
                forced_eos_token_id=None,

                # Limitation
                bad_words_ids=None,
                force_words_ids=None,
                constraints=None,
                suppress_tokens=None,
                begin_suppress_tokens=None,

                # Performance
                use_cache=True,

                # Output
                num_return_sequences=1,
                return_dict_in_generate=False,
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False,

                # Others
                synced_gpus=False,
                exponential_decay_length_penalty=None,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    def reset_history(self) -> None:
        self.dialogue = ""
    
    def append_history(self, text) -> None:
        self.dialogue += f"{text}\n"

    def infer(self) -> None:
        model: AutoModelForCausalLM | None
        tokenizer: AutoTokenizer | None
        e: Exception | None

        model, tokenizer, e = load_model(cfg.MODEL_NAME, cfg.CACHE_DIR, self.device, self.mode)

        if model is None or tokenizer is None:
            print(f"Error loading model or tokenizer: {e}")
            exit()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"========== Total parameters: {total_params}")

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
                print(f"AI: {ai_response}", end="\n\n")
                print(f"AI: {english_to_korean(ai_response)}", end="\n\n")

                self.append_history(ai_response)

            except Exception as e:
                print(f"Error occurred while running model: {e}")
                traceback.print_exc()

    def preprocess(example: dict, tokenizer: PreTrainedTokenizerBase) -> dict:
        user = example["messages"][0]["content"]
        assistant = example["messages"][1]["content"]

        text = f"User: {user}\nAssistant: {assistant}"

        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512
        )

        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    def train(self) -> None:
        model: AutoModelForCausalLM | None
        tokenizer: PreTrainedTokenizerBase | None
        e: Exception | None

        model, tokenizer, e = load_model(cfg.MODEL_NAME, cfg.CACHE_DIR, self.device, self.mode)

        if model is None or tokenizer is None:
            print(f"Error loading model or tokenizer: {e}")
            exit()

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"]
        )

        model = get_peft_model(model, lora_config)

        dataset = load_dataset("json", data_files=cfg.DATASET_DEFAULT)
        train_dataset = dataset["train"]

        train_dataset = train_dataset.map(
            lambda x: self.preprocess(x, tokenizer),
            remove_columns=train_dataset.column_names
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        training_args = TrainingArguments(
            output_dir="./result",
            per_device_train_batch_size=2,
            num_train_epochs=3
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        trainer.train()

if __name__ == "__main__":
    # gpt2 : 117M(파라미터 약 1.17억 개), 500MB
    # gpt2-medium : 345M(파라미터 약 3.45억 개), 1.5GB
    # gpt2-large : 774M(파라미터 약 7.74억 개), 3GB
    # gpt2-xl : 1.5 billion(파라미터 약 15억 개), 6GB
    # gpt-3 : 175 billion, 약 350GB
    # gpt-4 : 파라미터 수와 모델 크기에 대한 공식적인 정보가 공개되지 않았음

    mp.set_start_method("spawn", force=True)

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "infer"])
    parser.add_argument("--dataset", type=str, default=cfg.DATASET_DEFAULT)
    args = parser.parse_args()

    print(f"========== params(--mode): [{args.mode}], params(--dataset): [{args.dataset}]")

    llm = JagalGpt(args.mode, args.dataset)
    
    if not args.mode or "infer" == args.mode: # Inference mode (파라미터 없을시 추론 모드로 진입)
        llm.infer()
    
    elif "train" == args.mode: # Training mode (학습셋 파일 명시 필수)
        llm.train()

    else:
        pass
