# nodal-code-surrogate-pipeline

Pipeline for data generation, preprocessing, training, and evaluation of surrogate models for nodal-method-based core simulations.



리포지토리 구성 초안 (250905)

surrogate-model-pipeline/
├── configs/
│   ├── main_config.yaml          # 파이프라인 전체에서 사용할 공통 설정
│   ├── data_generation.yaml      # 데이터 생성 단계에 특화된 설정
│   └── model/
│       ├── regressor_v1.yaml
│       └── regressor_v2.yaml
│
├── data/
│   ├── .gitkeep                  # DVC로 관리될 데이터 폴더 (이 폴더는 .gitignore에 추가)
│   ├── raw/                      # 생성된 원시 데이터가 저장될 위치
│   └── processed/                # 전처리 후 데이터가 저장될 위치
│
├── artifacts/                    # 훈련/평가 결과물이 저장될 폴더 (이 폴더는 .gitignore에 추가)
│   ├── .gitkeep
│   ├── models/                   # 훈련된 모델 파일 (e.g., .pkl, .pt)
│   └── reports/                  # 평가 결과, 지표 (e.g., metrics.json, plots/)
│
├── notebooks/                    # 실험 및 분석용 노트북
│   ├── 1_data_exploration.ipynb
│   └── 2_model_analysis.ipynb
│
├── src/
│   └── surrogate_pipeline/
│       ├── __init__.py
│       ├── data_generation/      # [개선] 원시 데이터 '생성'을 명확히 함
│       │   ├── __init__.py
│       │   └── main.py             # 데이터 생성 파이프라인 실행 로직
│       │
│       ├── preprocessing/        # [개선] 명사 형태로 통일 (preprocess -> preprocessing)
│       │   ├── __init__.py
│       │   ├── preprocessor.py     # 전처리 파이프라인 오케스트레이터 (클래스/함수)
│       │   └── features.py         # 피쳐 엔지니어링, 변환 관련 함수
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── architecture.py     # 모델 구조 정의
│       │   └── factory.py          # 설정 파일을 받아 모델 객체를 생성
│       │
│       ├── training/             # [개선] 명사 형태로 통일 (train -> training)
│       │   ├── __init__.py
│       │   └── trainer.py          # 모델 학습 루프를 포함한 클래스/함수
│       │
│       └── evaluation/           # [개선] 명사 형태로 통일 (evaluate -> evaluation)
│           ├── __init__.py
│           ├── evaluator.py        # [개선] 평가 프로세스 실행 로직 (main.py -> evaluator.py)
│           └── metrics.py          # 평가지표 함수들
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py
│
├── .gitignore
├── dvc.yaml                      # DVC 파이프라인 (data_generation -> preprocessing -> training)
├── pyproject.toml                # 프로젝트 메타데이터 및 의존성 관리
├── run.py                        # [신규] 파이프라인 실행을 위한 단일 진입점
└── README.md
