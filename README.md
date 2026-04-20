# AI 기반 LLM을 활용한 인공신경망 대화 모델 구축

- 작성자: 김병민 (로컬 리포지토리 생성 2024년 5월, 리모트 리포지토리 생성 2024년 12월)
- curl https://api.github.com/repos/jagalmon/jagal-gpt

이 프로젝트는 Python 기반의 대화형 인공지능 애플리케이션을 구현하기 위한 코드와 실행 환경을 다룹니다.
터미널 환경에서 사용자와 상호작용하는 인터랙티브 구조를 가지며, GPU와 CPU 환경을 모두 고려하여 동작할 수 있도록 설계되었습니다.
또한 Docker를 통한 실행 환경 구성도 지원하여, 로컬 개발 환경뿐 아니라 컨테이너 기반 실행 환경에서도 일관된 방식으로 프로젝트를 구동할 수 있습니다.

이 프로젝트는 인공지능 모델 실행, 데이터 처리, 자연어 처리, 번역, 최적화, 배포, 코드 품질 관리까지 포함된 종합적인 Python 기반 AI 개발 환경입니다.

## 주요 기능

- 터미널 기반 대화 인터페이스 제공
- GPU 및 CPU 환경 지원
- Docker 기반 실행 지원
- 다양한 AI/NLP 패키지 연동 가능
- 데이터 처리 및 자연어 처리 확장 가능
- 코드 품질 관리 및 소스 보호 지원

## 패키지 구성 및 설명

## 패키지 구성 및 설명

### 1. AI 모델 실행 핵심

- torch  
딥러닝 연산 및 GPU 가속을 담당하는 핵심 프레임워크

- transformers  
자연어 처리 모델 실행 및 텍스트 생성 기능 제공

- tokenizers  
텍스트를 모델 입력 형태로 변환하는 고속 토크나이저

- accelerate  
CPU/GPU 환경 차이를 추상화하여 실행을 단순화

- peft  
경량 파인튜닝을 위한 파라미터 효율화 라이브러리

- safetensors  
모델 가중치 저장/로딩 최적화

- huggingface_hub  
모델 및 관련 파일 다운로드 및 관리

---

### 2. 데이터 처리

- datasets  
데이터셋 로딩 및 전처리

- numpy  
수치 계산 기본 라이브러리

- pandas  
테이블 데이터 처리

- pyarrow  
대용량 데이터 처리 및 컬럼형 포맷 지원

---

### 3. 자연어 처리 확장

- sentencepiece  
서브워드 토크나이징

- sacremoses  
텍스트 전처리 보조

- fasttext  
텍스트 분류 및 임베딩

- spacy  
형태소 분석, 개체명 인식 등 NLP 처리

- stanza  
문법 분석 및 언어학 기반 NLP 처리

---

### 4. 추론 최적화

- onnxruntime  
ONNX 기반 고속 추론 엔진

- ctranslate2  
텍스트 생성 및 번역 모델 추론 최적화

- psutil  
시스템 리소스 모니터링

---

### 5. 텍스트 처리

- regex  
확장 정규표현식

- beautifulsoup4  
HTML 파싱

---

### 6. 네트워크 및 API

- requests  
HTTP 요청

- httpx  
동기/비동기 HTTP 클라이언트

- aiohttp  
비동기 HTTP 통신

---

### 7. 설정 및 유틸

- pydantic  
데이터 검증 및 타입 관리

- PyYAML  
설정 파일 관리

- python-dotenv  
.env 파일 기반 환경변수 로딩

- Jinja2  
템플릿 처리

- rich  
터미널 출력 개선

- typer / click  
CLI 구성

---

### 8. 개발 및 테스트

- ruff  
코드 린트 및 품질 검사

- pytest  
테스트 실행 및 검증 프레임워크

---

## 정리

핵심 패키지 요약:

- 모델 실행: torch, transformers, tokenizers, accelerate, peft
- 데이터 처리: datasets, numpy, pandas
- NLP 확장: sentencepiece, spacy, stanza
- 최적화: onnxruntime, ctranslate2
- 통신: requests, httpx, aiohttp
- 개발: pydantic, ruff, pyarmor

이 프로젝트는 단순 챗봇이 아니라,
AI 모델 실행 + 데이터 처리 + NLP 확장 + 최적화 + 배포까지 포함된 통합 AI 개발 환경이다.

## Docker

- 도커 빌드: docker build -t jagal-gpt .
- 도커 실행: docker run -it jagal-gpt

## 패키지 요구 사항

- Python 3.7 이상(최초 리파지토리는 3.11.9에서 빌드 및 실행)
- torch
- transformers
- numpy

## 하드웨어 요구 사항

- GPU (권장): 빠른 추론을 위해 CUDA 지원 GPU 사용을 권장합니다.
- CPU: 지원되지만 응답 시간이 느릴 수 있습니다.

## 실행 후 상호작용

- 터미널에 메시지를 입력하고 Enter를 누릅니다.
- reset을 입력하면 대화 히스토리가 초기화됩니다.
- exit를 입력하면 AI가 종료됩니다.

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