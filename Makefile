#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tc5061-experimentacion-mlflow
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Entrena el modelo con los valores por defecto
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m telco_churn.modeling.train

## Corre las 8 corridas del experimento (la última repite la primera)
.PHONY: experimentos
experimentos:
	$(PYTHON_INTERPRETER) -m telco_churn.modeling.train
	$(PYTHON_INTERPRETER) -m telco_churn.modeling.train --C 0.001
	$(PYTHON_INTERPRETER) -m telco_churn.modeling.train --C 100
	$(PYTHON_INTERPRETER) -m telco_churn.modeling.train --class-weight balanced
	$(PYTHON_INTERPRETER) -m telco_churn.modeling.train --l1-ratio 1 --solver saga
	$(PYTHON_INTERPRETER) -m telco_churn.modeling.train --C 0.1 --class-weight balanced
	$(PYTHON_INTERPRETER) -m telco_churn.modeling.train --C 0.1 --l1-ratio 1 --solver saga --class-weight balanced
	$(PYTHON_INTERPRETER) -m telco_churn.modeling.train

## Abre la interfaz de MLflow en http://127.0.0.1:5001
.PHONY: ui
ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
