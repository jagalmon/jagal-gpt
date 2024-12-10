import torch
#from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
import config as cfg

# GPU가 있는 머신에서는 GPU 사용, GPU가 없는 머신에서는 CPU 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"able device: {device}")

#import warnings
#warnings.filterwarnings('ignore', category=UserWarning, message='Failed to initialize NumPy')

# gpt2 : 117M(파라미터 약 1.17억 개), 500MB
# gpt2-medium : 345M(파라미터 약 3.45억 개), 1.5GB
# gpt2-large : 774M(파라미터 약 7.74억 개), 3GB
# gpt2-xl : 1.5 billion(파라미터 약 15억 개), 6GB
# gpt-3 : 175 billion, 약 350GB
# gpt-4 : 파라미터 수와 모델 크기에 대한 공식적인 정보가 공개되지 않았음

model_name = 'gpt2' # 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
cache_dir = '/Users/hyoje/dev/llm-cache/'

try:
    #model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
    #tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
    #model = GPTNeoForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    #tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    model.to(device) # 연산 대상 device 선택, default: CPU 연산
    model.eval() # 언어 모델을 training mode에서 evaluation mode로 전환
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

# 하나의 토큰은 영어의 경우 약 4자의 텍스트에 해당
# 일반적으로 영어 문장은 약 15-20개의 토큰으로 구성
# 모델 인아웃의 기본 length인 50 토큰은 대략 2-3문장 정도에 해당

dialogue_history = ""

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    elif user_input.lower() == "reset":
        dialogue_history = ""
        print("Dialogue history has been reset.")
        continue

    dialogue_history += f"{user_input}\n"

    try:
        input_ids = tokenizer.encode(dialogue_history, return_tensors='pt').to(device)

        embedding_output = model.transformer.wte(input_ids)
        print(f"Embedding output: {embedding_output}")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        # 토큰 제한을 둬서 최신 대화만 유지 필요
        max_length = input_ids.shape[1] + cfg.TOKEN_LENGTH  # 입력 길이 + 생성할 길이
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        pad_token_id = tokenizer.eos_token_id

        print(f"Token length: {input_ids.shape[1]}")

        with torch.no_grad(): # 그래디언트 계산을 비활성화하여 추론 중 메모리를 절약하고 속도를 높임
            outputs = model.generate(input_ids, max_length=max_length, num_return_sequences=1, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, do_sample=True)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        ai_response = generated_text[len(dialogue_history):]
        print(f"AI: {ai_response}")

        dialogue_history += f"{ai_response}\n"
    except Exception as e:
        print(f"Error occurred while running model: {e}")
