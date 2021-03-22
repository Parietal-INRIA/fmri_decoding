lint:
	pylint Experiment/

test:
	pytest -vv --cov-report term-missing --no-cov-on-fail --cov=Experiment/ .
