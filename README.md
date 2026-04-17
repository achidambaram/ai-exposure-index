# AI Exposure Index API

This is a rebuilt version of the course project based on the original repository.

## What changed

The original pipeline exposed raw and model-based exposure values that could be negative or hard to interpret. This version replaces that with a bounded **AI Exposure Index (AEI)** on a **0 to 1** scale.

### New scoring logic

1. Use O*NET ability **importance** and **level** as occupation-specific weights.
2. Transform raw ability-level AI exposure with a **sigmoid** so every ability exposure is in **[0, 1]**.
3. Compute an occupation AEI score as a weighted average of transformed ability exposures.
4. Assign risk bands using the new course thresholds:  
   - **Green**: score < 0.33  
   - **Yellow**: 0.33 <= score <= 0.66  
   - **Red**: score > 0.66

This makes the score easier to explain in class, avoids negative contributions, and supports straightforward API outputs.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m ai_exposure_api.cli fetch-data
python -m ai_exposure_api.cli train-model --model rule_based
python -m ai_exposure_api.cli predict --query "Data Scientist"
python -m ai_exposure_api.cli serve --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## Example API call

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"query": "Data Scientist", "top_k": 8}'
```

## Main outputs

- `aei_score`: bounded 0–1 score
- `risk_band`: Green / Yellow / Red
- `top_contributors`: ability-level explanation using normalized exposure and ability weights

## Project structure

```text
ai_exposure_repo/
├── ai_exposure_api/
│   ├── api.py
│   ├── cli.py
│   ├── config.py
│   ├── data_pipeline.py
│   ├── modeling.py
│   └── utils.py
├── data/
├── models/
├── tests/
├── README.md
└── requirements.txt
```
