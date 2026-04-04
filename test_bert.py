import torch
from transformers import BertModel, BertTokenizer

#임베딩 벡터 처리

model_name = 'monologg/kobert'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

dialogue_history = ""

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    elif user_input.lower() == "reset":
        dialogue_history = ""
        print("Dialogue history has been reset.")
        continue

    #dialogue_history += f"User: {user_input}\n"
    dialogue_history += f"{user_input}\n"

    inputs = tokenizer(dialogue_history, return_tensors='pt', max_length=128, padding='max_length', truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    print(f"AI: {last_hidden_states}")

    #dialogue_history += f"AI: {last_hidden_states}\n"
    dialogue_history += f"{last_hidden_states}\n"
