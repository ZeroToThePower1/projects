@echo off
python eurusd_predictor.py ingest_prices --symbol EURUSD=X --interval 1d
python eurusd_predictor.py ingest_calendar
python eurusd_predictor.py featurize
python eurusd_predictor.py train
python eurusd_predictor.py predict
python eurusd_predictor.py backtest
pause