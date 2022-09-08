.ONESHELL:
# Oneshell means I can run multiple lines in a recipe in the same shell, so I don't have to
# chain commands together with semicolon
# Need to specify bash, fish, tcsh, xonsh, zsh, or powershell
# in order for conda activate to work as of conda 4.12.0.
SHELL=/bin/bash
#.SHELLFLAGS = -e

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################




ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = brats_2018_on_intel
PYTHON_INTERPRETER = python3
BASE_ENV = jupyter_launch
ENV1 = explore_data
ENV2 = data_pipeline
ENV3 = train_model
ENV4 = optimize_model
ENV5 = nncf_model
ENV6 = results_comp
DATA_FILENAME = Task01_BrainTumour.tar

ifeq (,$(shell which conda))
	HAS_CONDA=False
else
	HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
install_miniconda:
ifeq (False,$(HAS_CONDA))
	@echo ">>> Conda not detected, installing latest miniconda."
	mkdir -p ~/miniconda3
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
	bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
	rm -rf ~/miniconda3/miniconda.sh
	~/miniconda3/bin/conda init bash
	~/miniconda3/bin/conda init zsh
else 
	@echo ">>> Conda already installed." ; conda -V 
	@echo ">>> If you want to check for an available update, run the command:"
	@echo ">>> conda update conda"
endif

## Build base conda environment
base_env:
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environments."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda remove --name $(BASE_ENV) --all -y
	cp ./environments/builds/env_$(BASE_ENV).yml environment.yml
	conda env create -f environment.yml
	rm environment.yml
	$(CONDA_ACTIVATE) $(BASE_ENV)
	ipython kernel install --user --name=$(BASE_ENV)
	conda deactivate
else
#	conda create --name $(PROJECT_NAME) python=2.7
endif
	@echo ">>> New conda env created. Activate with:"
	@echo
	@echo "conda activate $(BASE_ENV)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif



## Build EDA and Data Viz conda environment
explore_env: 
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda remove --name $(ENV1) --all -y
	cp ./environments/builds/env_$(ENV1).yml environment.yml
	conda env create -f environment.yml
	rm environment.yml
	$(CONDA_ACTIVATE) $(ENV1)
	ipython kernel install --user --name=$(ENV1)
	conda deactivate
else
#	conda create --name $(PROJECT_NAME) python=2.7
endif
	@echo ">>> New conda environment created. Activate with: conda activate $(ENV1)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif


## Build Data Pipeline conda environment
data_env: 
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda remove --name $(ENV2) --all -y
	cp ./environments/builds/env_$(ENV2).yml environment.yml
	conda env create -f environment.yml
	rm environment.yml
	$(CONDA_ACTIVATE) $(ENV2)
	ipython kernel install --user --name=$(ENV2)
	conda deactivate
else
#	conda create --name $(PROJECT_NAME) python=2.7
endif
	@echo ">>> New conda environment created. Activate with: conda activate $(ENV2)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif


## Build TF Model Training conda environment
train_env: 
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda remove --name $(ENV3) --all -y
	cp ./environments/builds/env_$(ENV3).yml environment.yml
	conda env create -f environment.yml
	rm environment.yml
	$(CONDA_ACTIVATE) $(ENV3)
	ipython kernel install --user --name=$(ENV3)
	conda deactivate
else
#	conda create --name $(PROJECT_NAME) python=2.7
endif
	@echo ">>> New conda environment created. Activate with: conda activate $(ENV3)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif


## Build OpenVINO Model Optimization conda environment
optimize_env: 
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda remove --name $(ENV4) --all -y
	cp ./environments/builds/env_$(ENV4).yml environment.yml
	conda env create -f environment.yml
	rm environment.yml
	$(CONDA_ACTIVATE) $(ENV4)
	@echo ">>> Starting libgl1 fix...."
	sudo apt-get update && sudo apt-get install libgl1 -y
	@echo ">>> Ending libgl1 fix"
	ipython kernel install --user --name=$(ENV4)
	conda deactivate
else
#	conda create --name $(PROJECT_NAME) python=2.7
endif
	@echo ">>> New conda environment created. Activate with: conda activate $(ENV4)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif


## Build OpenVINO NNCF Optimization conda environment
nncf_env: 
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda remove --name $(ENV5) --all -y
	cp ./environments/builds/env_$(ENV5).yml environment.yml
	conda env create -f environment.yml
	rm environment.yml
	$(CONDA_ACTIVATE) $(ENV5)
	ipython kernel install --user --name=$(ENV5)
	conda deactivate
else
#	conda create --name $(PROJECT_NAME) python=2.7
endif
	@echo ">>> New conda environment created. Activate with: conda activate $(ENV5)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif


## Build Results Comparison conda environment
results_env: 
ifeq (True,$(HAS_CONDA))
	@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda remove --name $(ENV6) --all -y
	cp ./environments/builds/env_$(ENV6).yml environment.yml
	conda env create -f environment.yml
	rm environment.yml
	$(CONDA_ACTIVATE) $(ENV6)
	ipython kernel install --user --name=$(ENV6)
	conda deactivate
else
#	conda create --name $(PROJECT_NAME) python=2.7
endif
	@echo ">>> New conda environment created. Activate with: conda activate $(ENV6)"
else
	$(PYTHON_INTERPRETER) -m pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already installed.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif



## Set up python interpreter environment
create_environments: base_env explore_env data_env train_env optimize_env nncf_env results_env


## Test python environment is setup correctly
test_environment: base_environment
	$(PYTHON_INTERPRETER) test_environment.py

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Download Dataset
download_data: 
	$(CONDA_ACTIVATE) $(BASE_ENV)
	$(PYTHON_INTERPRETER) src/data/download_dataset.py $(PROJECT_DIR) $(DATA_FILENAME)

## Unpack Dataset
unpack_data: data/raw/$(DATA_FILENAME)
	$(CONDA_ACTIVATE) $(BASE_ENV)
	$(PYTHON_INTERPRETER) src/data/make_dataset.py data/raw/$(DATA_FILENAME) data/processed

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
