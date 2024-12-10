# AI 기반 LLM을 활용한 인공신경망 대화 모델 구축

- Creator: 김병민
- 이 코드는 Hugging Face의 Transformers 라이브러리를 사용하여 GPT-2 모델 기반의 챗봇을 구현한 Python 코드입니다. 대화 히스토리를 유지하며 문맥에 맞는 응답을 생성할 수 있는 대화형 인터페이스를 제공합니다. GPU와 CPU 환경 모두에서 동작하도록 설계되었습니다.

## 주요 기능

- 사전학습된 GPT-2 모델 사용: gpt2, gpt2-medium, gpt2-large, gpt2-xl과 같은 다양한 GPT-2 모델을 지원합니다.
- 동적 장치 선택: GPU가 있으면 자동으로 GPU를 사용하며, 없는 경우 CPU를 사용합니다.
- 터미널 기반 대화 인터페이스: 사용자가 터미널에서 챗봇과 실시간으로 상호작용할 수 있습니다.
- 응답 길이 설정 가능: 대화 길이에 따라 생성되는 텍스트의 길이를 동적으로 조정할 수 있습니다.
- 메모리 최적화: 추론 시 그래디언트 계산을 비활성화하여 메모리 사용량을 줄이고 속도를 향상시켰습니다.

## 패키지 요구 사항

- Python 3.7 이상(현재 리파지토리는 3.11.9에서 빌드 및 실행)
- torch
- transformers
- numpy

## 하드웨어 요구 사항

- GPU (권장): 빠른 추론을 위해 CUDA 지원 GPU 사용을 권장합니다.
- CPU: 지원되지만 응답 시간이 느릴 수 있습니다.

## 실행 후 상호작용

- 터미널에 메시지를 입력하고 Enter를 누릅니다.
- reset을 입력하면 대화 히스토리가 초기화됩니다.
- exit 또는 quit을 입력하면 AI가 종료됩니다.

## 설정

- 실행 전, config.py 파일을 수정하여 최대 토큰 길이를 설정합니다.

## 문제 해결

- 모델 다운로드 문제: cache_dir 경로가 쓰기 가능하고 충분한 공간이 있는지 확인하세요.
- CUDA 인식 실패: GPU 사용 가능 여부를 확인하고 적절한 CUDA 드라이버를 설치하세요.
- 성능 저하: 더 작은 모델 크기로 전환하거나 가능한 경우 GPU를 사용하세요.

## 참고 자료

- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- PyTorch Tutorials: https://pytorch.org/tutorials/
- Hugging Face Transformers Documentation: https://huggingface.co/docs/transformers/main/en/index
- Hugging Face Model Hub: https://huggingface.co/models

## License

- This project is licensed under the MIT License. See the LICENSE file for details.