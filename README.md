#  Crea el entorno virtual   "uv init"
	uv init call-me-maybe --python 3.10
Esto genera pyproject.toml, un .python-version, un README.md placeholder y un .gitignore ya preparado para Python.

# Añadir una dependencia (equivalente a pip install + anotar)
	uv add pydantic
	uv add numpy

# Quitar una dependencia
	uv remove numpy

# Instalar todo desde lockfile
	uv sync

# Ejecutar un script
	uv run python ...

# .gitignore
	.venv/
	__pycache__/
	*.pyc
	.mypy_cache/
	.pytest_cache/
	data/output/
























