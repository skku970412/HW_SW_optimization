# Portfolio Runbook

## Full Reproduction

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce_portfolio.ps1
```

## Fast Path (N6 only)

```powershell
python scripts/generate_portfolio_assets.py
python -m pytest tests/unit/test_portfolio_packaging.py -q
```

## Commit-ready Artifacts

- `README.md`
- `docs/portfolio/final_report.md`
- `docs/portfolio/manifest.json`
- `docs/portfolio/figures/*.png`

## Logs

- `results/step_validation_runs.csv`
- `results/boardless_progress_log.csv`
- `logs/boardless_execution_log.md`
