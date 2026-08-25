.PHONY: all data elasticity forecast optimize experiment figures test clean ui

all: data elasticity forecast optimize experiment figures

data:
	python src/generate_data.py

elasticity:
	python src/elasticity.py

forecast:
	python src/forecast.py

optimize:
	python src/optimize_prices.py

experiment:
	python src/experiment.py

figures:
	python src/figures.py

test:
	pytest -q

ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	rm -rf mlflow.db mlruns data/*.csv reports/*.png __pycache__ src/__pycache__
