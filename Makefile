.PHONY: install run debug clean lint lint-strict

UV := $(shell command -v uv 2>/dev/null || echo "$(HOME)/.local/bin/uv")

check-uv:
	@if ! command -v uv >/dev/null 2>&1 && [ ! -x "$(UV)" ]; then \
		echo "\033[1;33muv no está instalado. Instalando...\033[0m"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "\033[1;32muv instalado correctamente.\033[0m"; \
	fi

install: check-uv
	@$(UV) sync

install:
	@uv sync


RUN_ARGS := $(filter-out run --,$(MAKECMDGOALS))

run:
	@clear
	@$(MAKE) install --no-print-directory
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
	-@uv run python -m src $(RUN_ARGS)
	@echo "\033[1;31m"
	@echo "\nEND OF PROGRAM - SEE YOU SOON!"
	@echo "\033[0m"
	@$(MAKE) clean --no-print-directory

debug:
	@uv run python -m pdb -m src

clean:
	@rm -rf .mypy_cache .pytest_cache
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -name '*.pyc' -delete

lint:
	@clear
	@uv run flake8 . --exclude=.venv,data,llm_sdk,__pycache__
	@uv run mypy . --exclude "(.venv|data|llm_sdk)" --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	@ $(MAKE) clean --no-print-directory

lint-strict:
	@clear
	@uv run flake8 .  --exclude=.venv,data,llm_sdk,__pycache__
	@uv run mypy . --strict --exclude "(.venv|data|llm_sdk)"
	@ $(MAKE) clean --no-print-directory


%:
	@:
