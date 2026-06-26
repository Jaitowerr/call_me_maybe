# Makefile

.PHONY: install run debug clean lint
 
install:
	@uv sync


RUN_ARGS := $(filter-out run --,$(MAKECMDGOALS))

run:
	@clear
	@$(MAKE) install --no-print-directory
# 	@$(MAKE) clean --no-print-directory
	@echo "\033[1;33m"
	@echo "     ______    ____    __     __             ___   ___  _______ "
	@echo "    |   ___|  / _  \  |  |   |  |           |   \_/   ||   ____|"
	@echo "    |  |     / /_\  \ |  |   |  |           |         ||  |__   "
	@echo "    |  |    /  ___   \|  |   |  |           |  |\_/|  ||   __|  "
	@echo "    |  |___/  /    \  \  |___|  |____       |  |   |  ||  |____ "
	@echo "    |______|_/      \__\______/______/      |__|   |__||_______|"
	@echo ""
	@echo "          ___   ___    ____  __     __  ______   _______ "
	@echo "         |   \_/   |  / _  \ \  \  /  /|   _  \ |   ____|"
	@echo "         |         | / /_\  \ \  \/  / |  |_)  ||  |__   "
	@echo "         |  |\_/|  |/  ___   \ \    /  |   _  < |   __|  "
	@echo "         |  |   |  |  /    \  \ |  |   |  |_)  ||  |____"
	@echo "         |__|   |__|_/      \__\|__|   |______/ |_______|"
	@echo ""
	@echo "                       ~ CALL ME MAYBE ~"
	@echo "\033[0m"
	@echo "\n"
	@uv run python -m src $(RUN_ARGS)
# 	@uv run python -m src
	@echo "\033[1;31m"
	@echo "\nEND OF PROGRAM - SEE YOU SOON!"
	@echo "\033[0m"
	@$(MAKE) clean --no-print-directory

debug:
	@uv run python -m pdb -m src

clean:
	@rm -rf __pycache__ .mypy_cache src/__pycache__ .pytest_cache
	@find . -name '*.pyc' -delete

lint:
	@uv run flake8 . --exclude=.venv,data,llm_sdk,__pycache__
	@uv run mypy . --exclude "(.venv|data|llm_sdk)" --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
# 	@ $(MAKE) clean

%:
	@:
# make run -- --input data/input/function_calling_tests.json --output data/output/function_calling_results.json