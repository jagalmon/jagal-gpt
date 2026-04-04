import torch
import torch.nn as nn
import torch.optim as optim

# 1. 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 초기화
model = SimpleNN()

# 2. 인코딩, 생성 및 디코딩

# 입력 데이터 생성 (여기서는 임의의 숫자 데이터 사용)
input_data = torch.randn(1, 10)  # 1개의 샘플, 10개의 특성

# 인코딩: 입력 데이터를 모델이 처리할 수 있는 형태로 변환 (이미 tensor 형태이므로 추가 작업 필요 없음)
encoded_input = input_data

# 생성: 모델을 사용하여 예측 생성
with torch.no_grad():  # 추론 모드에서 그래디언트 계산 비활성화
    generated_output = model(encoded_input)

# 디코딩: 출력을 사람이 읽을 수 있는 형태로 변환 (여기서는 단순히 텐서를 리스트로 변환)
decoded_output = generated_output.tolist()

print(f"Input: {encoded_input}")
print(f"Generated Output: {decoded_output}")