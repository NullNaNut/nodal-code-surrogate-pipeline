# nodal-code-surrogate-pipeline

```bash
nodal-code-surrogate-pipeline/
│
├── train_dataset/
│
├── src/
│   ├── components/
│   │   ├── data_generation/   // 데이터 생성 파이썬 모듈 (ASTRA, MASTER)
│   │   ├── preprocessing/     // 전처리 파이썬 모듈 (ASTRA, MASTER)
│   │   ├── train_models/      // 모델 생성 및 훈련 코드 (완성/검증된 아키텍처만)
│   │   └── evaluation/        // metric, plot
│   │
│   └── main.py                // component 오케스트레이션
│
├── notebooks/                 // 프로토타입/테스트 코드
│
└── experiments/               // 로컬 산출물 정리 (MLflow artifacts 기록 가능)
    └── ASTRA-optimization/    // <simulation-code>-<campaign> 포맷
        │                      // <operation-condition>-<execution-mode>-<predict-target> 포맷
        ├── CONSTP-DEP_CY1-CYCLE_LENGTH/ 
        │   ├── run-0001/ 
        │   ├── run-0002/
        │   └── ...
        └── LF-BOC_CY2-ASI/
            └── run-0001/
