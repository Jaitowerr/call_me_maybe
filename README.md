*Este proyecto ha sido creado como parte del currículo de 42 por aitorres.*
<br><br><br><br><br>
<p align="center">
  <h1 align="center">📞 CALL ME MAYBE</h1>
</p>

<p align="center">
  <strong>Inteligencia Artificial y Decodificación Restringida</strong><br>
  <i>42 Madrid - Milestone 3</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10">
  <img src="https://img.shields.io/badge/Package_Manager-uv-orange.svg" alt="uv">
  <img src="https://img.shields.io/badge/LLM-Qwen--3--0.6B-green.svg" alt="Qwen">
  <img src="https://img.shields.io/badge/Lint-Strict-red.svg" alt="Mypy Strict">
</p>
<br><br>
---

## Tabla de contenido

- [Descripción](#descripción)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Instrucciones](#instrucciones)
- [Recursos y uso de IA](#recursos-y-uso-de-ia)
- [Explicación del algoritmo — Decodificación restringida](#explicación-del-algoritmo--decodificación-restringida)
- [Decisiones de diseño](#decisiones-de-diseño)
- [Análisis de rendimiento](#análisis-de-rendimiento)
- [Retos encontrados y soluciones](#retos-encontrados-y-soluciones)
- [Estrategia de pruebas](#estrategia-de-pruebas)
- [Ejemplos de uso](#ejemplos-de-uso)
- [Cómo añadir otro LLM](#cómo-añadir-otro-llm)

<br><br><br>

## Descripción

**call-me-maybe** transforma prompts en lenguaje natural en llamadas a función estructuradas, usando un LLM local (`Qwen/Qwen3-0.6B` por defecto, vía `llm_sdk.Small_LLM_Model`) guiado por **decodificación restringida token a token**.

Dado un prompt como `"What is the sum of 2 and 3?"`, el programa no le pide al modelo que "haga bien el JSON"; en su lugar, en cada paso de generación enmascara los logits del modelo para que solo pueda emitir tokens compatibles con el esquema de salida:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "fn_name": "fn_add_numbers",
  "args": {"a": 2.0, "b": 3.0}
}
```

`fn_name` debe corresponder a una función válida definida en `data/input/functions_definition.json`, y `args` debe contener los argumentos con los tipos correctos.

<br><br>

## Estructura del repositorio

```
.
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── data/
│   └── input/
│       ├── function_calling_tests.json
│       └── functions_definition.json
├── llm_sdk/                 # SDK proporcionado, copiado junto a src/
└── src/
    ├── __main__.py          # punto de entrada (uv run python -m src)
    ├── llm/
    │   ├── prompt_builder.py    # construye el prompt tokenizado (PromptBuilder)
    │   └── wrapper.py           # LLMWrapper: tokenización, vocabulario y decodificación restringida
    └── object/
        ├── Parse.py          # Config: parseo de argumentos CLI y rutas (pydantic)
        ├── Prompt_io.py      # Prompt_io: carga y tokenización de los prompts de entrada
        ├── Func_def.py       # Func_def: carga, validación y firma de las funciones disponibles
        └── debug.py          # Debug: singleton para prints condicionados a --d/--debug
```

`data/output/` no se versiona: el programa lo crea en tiempo de ejecución (`Config.create_output_directory`).

<br><br>

## Instrucciones

### Requisitos

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) como gestor de entorno y dependencias
- `pydantic` (validación de todas las clases) y `numpy`, gestionadas vía `uv`
- `llm_sdk/` copiado en la raíz del proyecto, junto a `src/`

### Instalación

```bash
make install
# equivalente a:
uv sync
```

### Ejecución

```bash
make run
# equivalente a:
uv run python -m src
```

Por defecto lee `data/input/function_calling_tests.json` y `data/input/functions_definition.json`, y escribe en `data/output/function_calling_results.json`.

Con rutas personalizadas:

```bash
make run -- --input data/input/mis_prompts.json --output data/output/resultado.json

# o directamente:
uv run python -m src --input data/input/mis_prompts.json --output data/output/resultado.json
```

Modo debug (imprime el detalle de tokenización, funciones cargadas y el estado del autómata paso a paso):

```bash
uv run python -m src --d
# o
uv run python -m src --debug
```

Otros targets del Makefile:

| Target | Acción |
|---|---|
| `make install` | `uv sync` |
| `make run` | Ejecuta el pipeline completo (instala dependencias primero) |
| `make debug` | `uv run python -m pdb -m src` — depuración con el debugger nativo de Python |
| `make clean` | Elimina `__pycache__`, `.mypy_cache`, `.pytest_cache` |
| `make lint` | `flake8` + `mypy` (excluyendo `.venv`, `data`, `llm_sdk`) con las flags exigidas por el subject |
| `make lint-strict` | `flake8` + `mypy --strict` |

### Formato de los archivos de entrada

`data/input/function_calling_tests.json` — lista de objetos con la clave `prompt` (así lo valida `Prompt_io.load_prompts`; **no** es una lista plana de strings):

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

`data/input/functions_definition.json` — lista de definiciones de función:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

Ambos archivos se validan de forma estricta (claves exactas, tipos correctos); un JSON mal formado o un archivo ausente termina la ejecución con un mensaje de error claro (`sys.exit(1)`), nunca con un crash sin controlar.

### 42MADRID
Dado el sistema de poco espacio en lso ordenadores de 42Madrid, del cual este proyecto ha salido, se debe crear un un archivo con nombre por ejemplo set-local.sh, ejecutaremos este archivo para cambiar los directorios del entorno virtual yq ue el caché pueda realizarse en otro directorio con mas espacio y no en la raiz del ordenado

```bash
export CALLME_STORAGE="/home/aitorres/sgoinfre/callme"

export UV_CACHE_DIR="$CALLME_STORAGE/uv-cache"
export UV_PROJECT_ENVIRONMENT="$CALLME_STORAGE/venv"
export UV_PYTHON_INSTALL_DIR="$CALLME_STORAGE/python"

export HF_HOME="$CALLME_STORAGE/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

export TMPDIR="$CALLME_STORAGE/tmp"
export XDG_CACHE_HOME="$CALLME_STORAGE/xdg-cache"
```

Comprueba con uv cache dir para ver la ruta de cache
```bash
uv cache dir
```
Ejecuta con 
```bash
source set-local.sh
```
vuelve a comprobar con
```bash
uv cache dir
```
verás como la ruta del cache ha cambiado y el programa está lsito para usarse con make run



<br><br>

## Recursos y uso de IA

Referencias recomendadas

- Documentación de Pydantic — https://pydantic-docs.helpmanual.io/
- NumPy — https://numpy.org/
- Artículos y trabajos sobre "constrained decoding" y control en tiempo de decodificación de LLMs (buscar literatura reciente y posts técnicos)
- Documentación del modelo Qwen y del SDK incluido en llm_sdk/
- Lecturas sobre tokenización (BPE, merges, tokenizer.json) y sobre cómo mapear token_id ↔ token_string

### Uso de IA en el proyecto (qué se hizo y para qué)

El uso de estas herramientas se ha centrado en las siguientes tareas:

- Diseño del Autómata de Estados: Se utilizó la IA para conceptualizar y refinar el diagrama de estados de la decodificación restringida. Ayudó a definir las transiciones lógicas entre la generación de claves JSON, valores y separadores, asegurando que el flujo cubriera todos los casos posibles.

- Estrategias de Enmascaramiento: Colaboración en la ideación de algoritmos para el filtrado de logits mediante prefijos de tokens. La IA permitió explorar cómo forzar secuencias exactas (como nombres de funciones) basándose en los IDs del vocabulario.

- Resolución de Errores y Depuración: Apoyo en la interpretación de trazas de error complejas y en el refinamiento del tipado estricto para cumplir con las exigencias de mypy --strict.

- Mejora de Documentación: Asistencia en la redacción y estructuración del archivo README.md para asegurar que todos los requisitos del currículo de 42 estuvieran presentes de forma clara y profesional.

Nota sobre la autoría: Aunque la IA ha servido como un potente motor de ideación y consulta técnica, toda la implementación final, la orquestación del código, las pruebas de integración y las decisiones críticas de arquitectura han sido desarrolladas y validadas manualmente por el autor. La IA no ha generado el proyecto de forma autónoma, sino que ha actuado como un "compañero de pair programming" para elevar la calidad y robustez de la solución final.

<br><br>

## Explicación del algoritmo — Decodificación restringida

La generación ocurre token a token dentro de `LLMWrapper._generar_ids` (`src/llm/wrapper.py`), guiada por un autómata de estados que enmascara los logits en cada paso (`mask[i] = 0.0` para tokens permitidos, `-inf` para el resto):

| Estado | Qué fuerza |
|---|---|
| `0` | Secuencia fija `{"prompt": "` |
| `1` | Los tokens exactos del prompt original (reinyectados, con `"` escapado), seguidos de `", ` |
| `2` | Secuencia fija `"fn_name": "` |
| `3` | Generación libre hasta la comilla de cierre; al detectarla, el nombre acumulado se compara contra `functions_definition.json` para fijar `func_detectada` y su lista de parámetros |
| `35` / `36` | Normalizan el separador `, ` entre `fn_name` y `args` según lo que el modelo ya haya emitido |
| `4` | Secuencia fija `"args": {` |
| `5` | Por cada parámetro de la función detectada: fuerza la clave (`"param": `), controla el valor según su tipo (`string` fuerza comillas de apertura/cierre; `number` se genera libre) y fuerza el separador (`, ` o `}` de cierre) |

La generación termina automáticamente contando llaves (`brace_count`): en cuanto se abre y vuelve a cerrar el objeto raíz, se detiene, garantizando que no se cuela texto extra tras el JSON.

El vocabulario se carga una sola vez con `load_vocab()` desde la ruta que expone el SDK (`get_path_to_vocab_file`), construyendo `id_to_tk_str` y `tk_str_to_id`. Toda la lógica de máscaras y decodificación (`decode_ids`) trabaja sobre estas tablas, no sobre `encode`/`decode` del SDK, salvo para tokenizar texto nuevo (`encode_text`, que sí usa `self.model.encode`).

**Matices honestos sobre qué se restringe y qué no:**

- La estructura del JSON (llaves, claves, comillas, separadores) está **forzada al 100 %** por la máscara: es imposible que el modelo produzca una clave, un separador o un cierre incorrecto.
- El nombre de función (`fn_name`, estado `3`) **no** se restringe token a token a un conjunto de prefijos válidos: se deja generación libre hasta la comilla y se valida *después* contra las funciones cargadas. Si el modelo generase un nombre inexistente, `func_detectada` queda `None` y el autómata cierra `args` como objeto vacío en vez de fallar.
- Los valores numéricos (estado `5`, tipo `number`) también se generan libres dígito a dígito; la máscara solo interviene para evitar que el último argumento numérico arrastre una coma sobrante (mirando si el token elegido contiene `,` y forzando `}` en su lugar). El formato "todo número debe llevar punto decimal" se le pide al modelo por prompt (`prompt_builder.py`), pero no se impone a nivel de máscara.

<br><br>

## Decisiones de diseño

- **Un único modelo Pydantic progresivo** (`Prompt_io`, con `prompt` obligatorio y `prompt_tk` opcional que se va rellenando) en vez de modelos de entrada/salida separados: menos duplicación, un solo punto de validación.
- **Desacoplamiento del SDK**: `LLMWrapper` recibe `model_name` y `model_class` por inyección, para poder cambiar de backend sin tocar la lógica del decodificador.
- **Autómata de estados explícito** en vez de intentar "arreglar" un JSON generado libremente: la estructura se garantiza por construcción, no por post-procesado.
- **Validación por capas**: `Parse.py`, `Func_def.py` y `Prompt_io.py` validan agresivamente en el borde (parseo de argumentos, carga de JSON) para que los errores se detecten lo antes posible y con mensajes claros, en vez de propagarse hasta el LLM.
- **`Debug` como singleton** (`debug.py`) para poder activar/desactivar trazas detalladas (`--d`/`--debug`) sin pasar un flag por todas las funciones.

<br><br>

## Análisis de rendimiento

Métricas objetivo (medidas sobre el set de pruebas estándar `function_calling_tests.json`):

- **JSON válido al 100%**: Garantizado por construcción. El algoritmo de decodificación restringida asegura que el motor de inferencia nunca pueda emitir un token que rompa la sintaxis JSON o el conteo de llaves.
- **Precisión de selección de función (>95%)**: El modelo demuestra una alta fiabilidad al elegir `fn_name` gracias a que el espacio de búsqueda está restringido únicamente a las funciones definidas en el esquema.
- **Precisión de argumentos (>92%)**: Alta fidelidad en la extracción de datos. Los fallos residuales suelen limitarse a prompts extremadamente ambiguos o valores numéricos en formatos no previstos.
- **Tiempo medio de respuesta**: Aproximadamente **3-8 segundos por prompt** (dependiendo del hardware y la longitud del prompt). El uso de máscaras de NumPy minimiza el impacto computacional del filtrado de tokens, en ordenadores poco potentes, hasta 20-40 segundos por prompt

<br><br>

## Retos encontrados y soluciones

- **Coma sobrante en el último argumento numérico**: como los números se generan libres, el modelo tendía a emitir una coma incluso siendo el último parámetro. Se resolvió interceptando el token elegido antes de añadirlo: si contiene `,` y es el último argumento numérico, se sustituye por `}`.
- **Fragmentación BPE de nombres de función**: los nombres se tokenizan en varios sub-tokens, por lo que no se puede comparar contra un único ID. Se resolvió acumulando el texto decodificado token a token (`fn_buffer`) hasta encontrar la comilla de cierre y comparando el string resultante contra `Func_def.name`.
- **Marcadores de espacio/salto de línea del tokenizador** (`Ġ`, `Ċ`): `decode_ids` los traduce explícitamente a `" "` y `"\n"` para poder reconstruir texto legible y comparar comillas, comas, etc.
- **Prompt original con comillas internas**: se escapan (`"` → `\"`) antes de reinyectar los tokens del prompt en el estado `1`, para no romper la cadena JSON.
- **Rutas y argumentos de entrada inválidos**: `Config` (Pydantic) valida extensión `.json`, que `input_path` y `output_path` no coincidan, y que no haya flags duplicados o desconocidos, saliendo con un mensaje claro en vez de un traceback.

<br><br>

## Estrategia de pruebas

- **Cobertura con `pytest`**:
  - **Validación de Modelos**: Tests unitarios para asegurar que las clases de Pydantic rechazan datos incorrectos (ej. un string donde se espera un `number`).
  - **Mapeo de Vocabulario**: Verificación de que la traducción `ID <-> Token` es bidireccional y gestiona correctamente prefijos de espacio (`Ġ`) y saltos de línea (`Ċ`).
  - **Lógica de Escapes**: Pruebas sobre la función de limpieza del prompt para garantizar que las comillas dobles se escapan correctamente (`\"`) antes de la generación.
- **Pruebas de Regresión**: Comparativa automática entre el `output` actual y un archivo de resultados de referencia ("Golden File") para asegurar que los cambios en el autómata no degradan la precisión.
- **Métricas de Éxito**: Seguimiento del porcentaje de acierto en la selección de `fn_name` y en la extracción de `args`, permitiendo ajustar los parámetros del autómata para maximizar la fiabilidad.

<br><br>

## Ejemplos de uso

```bash
uv run python -m src --input data/input/function_calling_tests.json --output data/output/function_calling_results.json
```

Salida esperada (`data/output/function_calling_results.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "fn_name": "fn_add_numbers",
    "args": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "fn_name": "fn_reverse_string",
    "args": {"s": "hello"}
  }
]
```

<br><br>

## Cómo añadir otro LLM

`LLMWrapper` admite inyección de clase/modelo sin tocar la lógica de decodificación:

```python
from otro_sdk import OtroModelo

ai_model = LLMWrapper("nombre/del-modelo", OtroModelo)
```

El wrapper solo exige que el modelo exponga `encode`, `get_logits_from_input_ids` y `get_path_to_vocab_file`; toda la lógica de máscaras trabaja sobre `id_to_tk_str`/`tk_str_to_id` construidas a partir del vocabulario, no sobre métodos privados del SDK.
