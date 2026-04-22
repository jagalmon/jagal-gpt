import faulthandler
faulthandler.enable()

from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
UTILS_DIR = BASE_DIR / "utils"
sys.path.insert(0, str(UTILS_DIR))

import argparse as ap
import config as cfg
import certifi as ctf
from classifier import load_classifier, detect_lang #type: ignore
from datasets import load_dataset
from device import get_device #type: ignore
from dotenv import load_dotenv
from encode import encode_to_device #type: ignore
import getpass as gp
from harness import init_harness, proc_harness #type: ignore
import json
from pretrained import load_model #type: ignore
import logging
import multiprocessing as mp
import numpy as np
import os
from peft import LoraConfig, get_peft_model
import platform as pf
from pydantic import BaseModel, Field, PrivateAttr
import pytest
import requests as rqs
import ssl
import torch
import traceback as tb
from training import serialize_messages #type: ignore
from translator import any_to_english, english_to_korean #type: ignore
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from unittest.mock import patch
import urllib.request

class JagalGpt(BaseModel):
    mode: str | None
    dataset: str
    username: str
    device: str = Field(default_factory=get_device)
    dialogue: str = ""
    _classifier = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)

        os.environ['SSL_CERT_FILE'] = ctf.where()
        print(f"========== SSL cert path: {ssl.get_default_verify_paths()}")

        print(f"========== Insecure HTTPS request: {rqs.get('https://huggingface.co', verify=False)}")

    def model_post_init(self, __context):
        print(f"========== Able device: {self.device}")

        classifier_path = os.path.join(cfg.CACHE_DIR, cfg.CLASSIFIER_PATH)
        print(f"========== Classifier path: {classifier_path}")

        if not os.path.exists(classifier_path):
            print(f"========== Classifier not exists")
            urllib.request.urlretrieve(
                cfg.CLASSIFIER_URL,
                classifier_path
            )
        else:
            print(f"========== Classifier already exists")

        self._classifier = load_classifier()

        init_harness()

    def generate_response(self, model, tokenizer) -> str:
        input_ids = encode_to_device(tokenizer, self.dialogue, self.device)

        if "gpt" in cfg.MODEL_NAME:
            embedding_output = model.transformer.wte(input_ids)
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(self.device)

        elif "gemma" in cfg.MODEL_NAME:
            embedding_output = model.get_input_embeddings()(input_ids)
            attention_mask = torch.ones_like(input_ids).to(self.device)
        
        elif "llama" in cfg.MODEL_NAME:
            embedding_output = model.get_input_embeddings()(input_ids)
            attention_mask = torch.ones_like(input_ids).to(self.device)

        elif "phi" in cfg.MODEL_NAME:
            embedding_output = model.get_input_embeddings()(input_ids)
            attention_mask = torch.ones_like(input_ids).to(self.device)

        elif "mistralai" in cfg.MODEL_NAME:
            embedding_output = model.get_input_embeddings()(input_ids)
            attention_mask = torch.ones_like(input_ids).to(self.device)

        elif "bloom" in cfg.MODEL_NAME:
            embedding_output = model.transformer.word_embeddings(input_ids)
            attention_mask = torch.ones_like(input_ids).to(self.device)

        elif "falcon" in cfg.MODEL_NAME:
            embedding_output = model.transformer.word_embeddings(input_ids)
            attention_mask = torch.ones_like(input_ids).to(self.device)

        else:
            pass

        # 토큰 제한을 둬서 최신 대화만 유지 필요
        max_length = input_ids.shape[1] + cfg.TOKEN_LENGTH_INFER  # 입력 길이 + 생성할 길이

        print(f"========== Embedding output: {embedding_output}")
        print(f"========== Embedding shape: {embedding_output.shape}")
        print(f"========== Token length: {input_ids.shape[1]}")

        #with torch.no_grad(): # gradient 계산 비활성화
        with torch.inference_mode(): # gradient 계산 비활성화, autograd 그래프 생성 안함, 역전파 비용 없음, intermediate tensors 저장 최소화
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,

                # Length
                max_new_tokens=cfg.TOKEN_LENGTH_INFER,
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

        #generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # 입력 토큰 + 생성 토큰 전체
        generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True) # 입력 제외 + 생성된 토큰만

        return generated_text

    def reset_history(self) -> None:
        self.dialogue = ""
    
    def append_history(self, text) -> None:
        self.dialogue += f"{text}\n"
        #self.dialogue += f"{self.username}: {text}\nAI:"

    def infer(self) -> None:
        model: AutoModelForCausalLM | None
        tokenizer: AutoTokenizer | None
        e: Exception | None

        model, tokenizer, e = load_model(cfg.MODEL_NAME, cfg.CACHE_DIR, self.device, self.mode)

        if model is None or tokenizer is None:
            print(f"Error loading model or tokenizer: {e}")
            exit()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"========== Total parameters: {total_params}")

        while True:
            user_input = input(f"{self.username}: ")
            if user_input.lower() in ["exit"]:
                break
            elif user_input.lower() == "reset":
                self.reset_history()
                print("Dialogue history has been reset.")
                continue

            lang_label = detect_lang(self._classifier, user_input)

            if "__label__en" != lang_label:
                user_input = any_to_english(user_input)
                print(f"========== translator: \"{user_input}\"")

            self.append_history(user_input)

            try:
                response = self.generate_response(model, tokenizer)

                ai_response = proc_harness(response)
                
                print(f"AI: \"{ai_response}\"", end="\n\n")
                print(f"AI: \"{english_to_korean(ai_response)}\"", end="\n\n")

                self.append_history(ai_response)

            except Exception as e:
                print(f"Error occurred while running model: {e}")
                tb.print_exc()

    def preprocess(self, example: dict, idx: int, tokenizer: PreTrainedTokenizerBase) -> dict:
        text = serialize_messages(example, idx)

        max_token_length = cfg.TOKEN_LENGTH_TRAIN

        tokens = tokenizer( # 문자열를 토큰으로 변환
            text=text,
            add_special_tokens=False, # BOS/EOS 같은 토큰 자동 추가
            truncation=True, # 길면 자르기
            max_length=max_token_length, # 최대 토큰 길이. (토큰 기준) 1 토큰 ≈ 3~4 영어 글자. 512 토큰 ≈ 1500~2000 글자. (영어 기준)
            padding="max_length", # max_length까지 패딩
            stride=0, # sliding window 없음
            is_split_into_words=False, # 이미 분리된 단어 여부
            return_tensors=None,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_overflowing_tokens=False,
            return_special_tokens_mask=False,
            return_offsets_mapping=True,
            verbose=True # 로그 출력 여부
        )

        input_ids = tokens["input_ids"]
        offsets = tokens["offset_mapping"]

        labels = input_ids.copy()

        # user 영역 문자 위치 기준으로 mask
        cursor = 0
        mask_ranges = []

        for msg in example["messages"]:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                part = f"User: {content}\n"
            elif role == "assistant":
                part = f"Assistant: {content}\n"
            else:
                continue

            start = cursor
            end = cursor + len(part)

            if role == "user":
                mask_ranges.append((start, end))

            cursor = end

        # 토큰 단위로 mask 적용
        for i, (s, e) in enumerate(offsets):
            for ms, me in mask_ranges:
                if s >= ms and e <= me:
                    labels[i] = -100

        # padding 제거
        pad_id = tokenizer.pad_token_id
        if pad_id is not None:
            labels = [t if t != pad_id else -100 for t in labels]

        tokens["labels"] = labels

        return tokens
    
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        mask = labels != -100 # labels에서 -100은 무시 대상

        correct = (preds[mask] == labels[mask]).astype(np.float32) # 유효 토큰만 비교
        acc = correct.mean().item() if correct.size > 0 else 0.0

        return {"accuracy": acc}
    
    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=-1)

    def train(self) -> None:
        model: AutoModelForCausalLM | None
        tokenizer: PreTrainedTokenizerBase | None
        e: Exception | None

        model, tokenizer, e = load_model(cfg.MODEL_NAME, cfg.CACHE_DIR, self.device, self.mode)

        if model is None or tokenizer is None:
            print(f"Error loading model or tokenizer: {e}")
            exit()

        tokenizer.padding_side = "right"

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_config = LoraConfig(
            r=8, # low-rank 차원
            lora_alpha=16, # scaling
            target_modules=["q_proj", "v_proj"], # 적용 레이어
            lora_dropout=0.0, # dropout 없음
            bias="none", # bias 학습 안함
            task_type="CAUSAL_LM", # Causal Language Model. 앞 토큰 보고 다음 토큰 예측하는 GPT 방식 모델.
            inference_mode=False, # 학습 모드
            modules_to_save=None # 추가 저장 모듈 없음
        )

        model = get_peft_model(model, lora_config) # LoRA 기반 미세조정(PEFT) 적용

        dataset_path = os.path.join(cfg.DATASET_DIR_NAME, self.dataset)
        #print(f"========== dataset_path: {dataset_path}")
        dataset = load_dataset("json", data_files=dataset_path)
        train_dataset = dataset["train"]

        # 샘플 채취, 포맷 확인
        sample = train_dataset[0]
        processed = self.preprocess(sample, 0, tokenizer)
        print(f"========== (sampling) json dumps: " + json.dumps(sample, indent=2, ensure_ascii=False))
        print(f"========== (sampling) decoded: {tokenizer.decode(processed['input_ids'])}")
        print(f"========== (sampling) labels: {processed['labels'][:100]}", end="\n\n")

        # 학습 전처리
        train_dataset = train_dataset.map(
            lambda x, i: self.preprocess(x, i, tokenizer),
            with_indices=True,
            remove_columns=train_dataset.column_names
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False, # BERT식 마스킹 학습이 아니라 GPT식 causal LM 학습
            mlm_probability=0.15, # mlm=True 일때 입력 토큰 중 mlm_probability * 100% 를 마스킹
            pad_to_multiple_of=None, # 패딩 배수 없음
            return_tensors="pt" # pytorch tensor 반환
        )
        
        training_args = TrainingArguments(
            output_dir=cfg.LEARNED_DIR,
            do_train=True, # 학습 수행
            do_eval=False, # 평가 여부
            per_device_train_batch_size=1, # 학습 배치 사이즈
            per_device_eval_batch_size=2, # 평가 배치 사이즈
            gradient_accumulation_steps=1, # gradient 누적 없음. 한번의 스텝마다 가중치(weight)에 update. 여러 스텝 모으지 않는다는 뜻.
            learning_rate=5e-5, # 모델 파라미터를 얼마나 크게 업데이트할지 결정하는 계수. 트랜스포머 미세조정에서 일반적인 기본값. weight = weight - learning_rate * gradient
            weight_decay=0.0, # weight decay 없음
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            num_train_epochs=100, # 데이터 전체 반복 횟수
            max_steps=-1, # step 기준 학습 대신 epoch 기준 사용
            lr_scheduler_type="linear",
            warmup_steps=50, # 워밍업
            logging_dir="./logs", # 로그 경로
            logging_steps=500, # 로그 주기
            save_strategy="steps", # 저장 전략
            save_steps=500, # 저장 주기
            save_total_limit=None, # 저장 제한
            eval_strategy="no", # no | steps | epoch | 평가 여부. 평가시 속도 저하 감수.
            eval_steps=500, # 평가가 steps 일때 사용
            fp16=False, # half precision 노사용. 정확도 영향 적은 추론에서는 사용하나 수치 안정성 중요한 학습에서는 노사용.
            bf16=False, # bfloat16 노사용. 정확도 영향 적은 추론에서는 사용하나 수치 안정성 중요한 학습에서는 노사용.
            dataloader_num_workers=0, # 안정성 우선을 위해 멀티프로세싱 대신 싱글프로세싱 사용
            report_to=[], # TensorBoard | Weights & Biases | MLflow
            remove_unused_columns=True, # 불필요 컬럼 제거
            label_names=["labels"],
            disable_tqdm=False # (주의) False가 켜짐이고 True가 꺼짐
        )

        print(f"=========== final dataset check: {len(train_dataset)}")
        print(f"=========== final dataset check: {train_dataset[0]['input_ids'][:20]}")
        print(f"=========== final dataset check: {train_dataset[0]['labels'][:20]}")

        trainer = Trainer(
            model=model,
            args=training_args, # 학습 설정
            train_dataset=train_dataset, # 학습 데이터
            eval_dataset=None, # 평가 데이터
            data_collator=data_collator,
            compute_metrics=self.compute_metrics, # 평가 지표 계산 함수
            callbacks=[TrnCallback()],
            optimizers=(None, None), # 기본 optimizer 사용
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics # 평가 여부 yes 일시 후처리
        )

        #trainer.train()

class TrnCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"========== train start")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"========== train end")

    def on_step_begin(self, args, state, control, **kwargs):
        pass

    def on_step_end(self, args, state, control, **kwargs):
        print(f"========== step: {state.global_step}")

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"========== epoch begin")

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"========== epoch end")

if __name__ == "__main__":
    # gpt2 : 117M(파라미터 약 1.17억 개), 500MB
    # gpt2-medium : 345M(파라미터 약 3.45억 개), 1.5GB
    # gpt2-large : 774M(파라미터 약 7.74억 개), 3GB
    # gpt2-xl : 1.5 billion(파라미터 약 15억 개), 6GB
    # gpt-3 : 175 billion, 약 350GB
    # gpt-4 : 파라미터 수와 모델 크기에 대한 공식적인 정보가 공개되지 않았음

    print(f"========== Kernel: {pf.system()}")
    print(f"========== username: {gp.getuser()}")

    mp.set_start_method("spawn", force=True)

    logging.basicConfig(level=logging.INFO)

    parser = ap.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "infer"])
    parser.add_argument("--dataset", type=str, default=cfg.DATASET_DEFAULT)
    args = parser.parse_args()

    print(f"========== params(--mode): [{args.mode}], params(--dataset): [{args.dataset}]")

    llm = JagalGpt(mode=args.mode, dataset=args.dataset, username=gp.getuser())
    
    if not args.mode or "infer" == args.mode: # Inference mode (파라미터 없을시 추론 모드로 진입)
        llm.infer()
    
    elif "train" == args.mode: # Training mode (학습셋 파일 명시 필수)
        llm.train()

    else:
        pass

def add(a, b):
    return a + b

def divide(a, b):
    return a / b

@pytest.fixture
def sample_data():
    return {"a": 10, "b": 20}

@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (2, 3, 5),
    (10, 20, 30),
])
def test_add(a, b, expected):
    assert add(a, b) == expected

def test_add_with_fixture(sample_data):
    assert add(sample_data["a"], sample_data["b"]) == 30

def test_divide_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)

def fetch_status():
    return rqs.get(cfg.CLASSIFIER_URL).status_code

def test_fetch_status():
    assert fetch_status() == 200

@patch("requests.get")
def test_fetch_status_mock(mock_get):
    mock_get.return_value.status_code = 200
    assert fetch_status() == 200
