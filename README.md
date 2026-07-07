A CONTROLAR!!!!!
  PARSEO:
        "prompt": ""  Esto controlarlo como error, el prompt está vacío no contiene nada
        eliminar espacios antes del prompt es decir la cadena str si tiene espacios al principio o al final, limpiarla para prevenir de cosas extrañas o posibles errores



LO QUE FALTA



jaitowerr@Jaitowerr:~/42madrid/milestone_3/call_me_maybe$ tree
.
├── Makefile
├── README.md
├── data
│   ├── input
│   │   ├── function_calling_tests.json
│   │   └── functions_definition.json
│   └── output
│       └── function_calling_results.json
├── llm_sdk
│   ├── llm_sdk
│   │   └── __init__.py
│   ├── pyproject.toml
│   └── uv.lock
├── pyproject.toml
├── set-local.sh
├── src
│   ├── __init__.py
│   ├── __main__.py
│   ├── llm
│   │   ├── __init__.py
│   │   ├── prompt_builder.py
│   │   └── wrapper.py
│   ├── object
│   │   ├── Func_def.py
│   │   ├── Parse.py
│   │   ├── Prompt_io.py
│   │   └── __init__.py
│   └── src.zip
└── uv.lock

9 directories, 24 files
jaitowerr@Jaitowerr:~/42madrid/milestone_3/call_me_maybe$

































## ENTORNO VIRTUAL UV

###  Crea el entorno virtual   "uv init"
	uv init call-me-maybe --python 3.10
Esto genera pyproject.toml, un .python-version, un README.md placeholder y un .gitignore ya preparado para Python.

### Añadir una dependencia (equivalente a pip install + anotar)
	uv add pydantic
	uv add numpy

### Quitar una dependencia
	uv remove numpy

### Instalar todo desde lockfile
	uv sync

### Ejecutar un script
	uv run python ...
	
### .gitignore
	.venv/
	__pycache__/
	*.pyc
	.mypy_cache/
	.pytest_cache/
	data/output/




# Resumen del Proyecto: Call-Me-Maybe

## 1. data/input — Qué contiene exactamente

### Archivo: `callmemaybe_extracted/data/input/function_calling_tests.json`
**Contenido:** Contiene un total de 11 prompts (casos de prueba):
1. *What is the sum of 2 and 3?*
2. *What is the sum of 265 and 345?*
3. *Greet shrek*
4. *Greet john*
5. *Reverse the string 'hello'*
6. *Reverse the string 'world'*
7. *What is the square root of 16?*
8. *Calculate the square root of 144*
9. *Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS*
10. *Replace all vowels in 'Programming is fun' with asterisks*
11. *Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'*

**Propósito:** Cada uno de estos prompts es un caso de prueba que el programa debe procesar y traducir a una llamada de función estructurada (`fn_name` + `args`).

### Archivo: `callmemaybe_extracted/data/input/functions_definition.json`
**Contenido:** Lista de 5 funciones disponibles que actúan como la "API" que el sistema puede invocar. Cada una cuenta con `name`, `description`, `parameters` y `returns`.

#### Funciones y Parámetros:
* **`fn_add_numbers`**
    * **Descripción:** Suma dos números y devuelve la suma.
    * **Parámetros:**
        * `a`: `type = number`
        * `b`: `type = number`
    * **Retorno:** `number`
* **`fn_greet`**
    * **Descripción:** Genera un saludo para una persona por su nombre.
    * **Parámetro:**
        * `name`: `type = string`
    * **Retorno:** `string`
* **`fn_reverse_string`**
    * **Descripción:** Invierte una cadena y devuelve el resultado.
    * **Parámetro:**
        * `s`: `type = string`
    * **Retorno:** `string`
* **`fn_get_square_root`**
    * **Descripción:** Calcula la raíz cuadrada de un número.
    * **Parámetro:**
        * `a`: `type = number`
    * **Retorno:** `number`
* **`fn_substitute_string_with_regex`**
    * **Descripción:** Reemplaza todas las ocurrencias que casen con una regex en una cadena.
    * **Parámetros:**
        * `source_string`: `string`
        * `regex`: `string`
        * `replacement`: `string`
    * **Retorno:** `string`

> ⚠️ **Importante:** La salida JSON debe usar exactamente esos nombres y tipos. 
> 
> **Ejemplo de salida para el prompt *"What is the sum of 2 and 3?"*:**
> ```json
> {
>   "prompt": "What is the sum of 2 and 3?",
>   "fn_name": "fn_add_numbers",
>   "args": {"a": 2.0, "b": 3.0}
> }
> ```

---

## 2. llm_sdk — Qué hay y qué ofrece

**Fichero:** `callmemaybe_extracted/llm_sdk/llm_sdk/__init__.py`  
Define la clase `Small_LLM_Model` y utiliza `transformers` de Hugging Face.

### Clase: `Small_LLM_Model(model_name="Qwen/Qwen3-0.6B", device=None, dtype=None, trust_remote_code=True)`

* **`__init__`**: Carga el tokenizer y el modelo de HF usando `AutoTokenizer` y `AutoModelForCausalLM`. Selecciona automáticamente el dispositivo (`mps` / `cuda` / `cpu`) y pone el modelo en modo de evaluación (`eval()`).
* **`encode(text: str) -> torch.Tensor`**:
    * Tokeniza el texto utilizando el tokenizer de HF.
    * Devuelve un tensor 2-D (batch size = 1) con `input_ids` en el dispositivo correspondiente (tipo `dtype long`).
    * *Nota:* La salida **NO** es una simple lista de enteros, sino un tensor de PyTorch con forma `[1, N]`.
* **`decode(ids: Tensor | list[int]) -> str`**: Decodifica los `ids` a string omitiendo los tokens especiales (`skip_special_tokens=True`). Acepta tanto un tensor como una lista de enteros.
* **`get_logits_from_input_ids(input_ids: list[int]) -> list[float]`**:
    * Recibe una lista plana de token IDs (no un tensor) y construye el tensor de entrada para el modelo.
    * Ejecuta el modelo bajo el contexto de `no_grad()` y devuelve los **logits del último token** como una lista de floats.
    * Los logits representan los valores sin aplicar la función softmax (puntuación para cada token del vocabulario).
    * La longitud de la lista es igual al tamaño del vocabulario del modelo (`vocab_size`).
* **`get_path_to_vocab_file() -> str`**: Descarga (`hf_hub_download`) y devuelve la ruta local al fichero de vocabulario (ej. `vocab.json`). Es sumamente útil para inspeccionar la correspondencia `token_id ↔ token_string`.
* **`get_path_to_merges_file()`, `get_path_to_tokenizer_file()`**: Descargan y retornan las rutas a los archivos de merges (para BPE) y `tokenizer.json` respectivamente.

### Notas prácticas:
1. El SDK depende directamente de `transformers` y `torch`. Se espera que el entorno de evaluación o la *moulinette* cuente con estos paquetes instalados.
2. `get_logits_from_input_ids` devuelve una lista de floats (`list[float]`), la cual se utilizará para implementar la **decodificación restringida** (decidir qué tokens están permitidos en cada paso).
3. Para visualizar tokens legibles, es más fiable inspeccionar el JSON del vocabulario obtenido mediante `get_path_to_vocab_file()` para mapear de manera exacta `ids ↔ strings`.

---

## 3. src/main.py

* **Contenido actual:** Un script `main` minimalista que únicamente imprime `"Hello from call-me-maybe!"`.
* **Estado actual:** No hay lógica de negocio o del proyecto implementada en `src`. Será necesario estructurar nuevos módulos encargados de:
    * Cargar y procesar los archivos de entrada.
    * Interactuar con la clase `Small_LLM_Model`.
    * Ejecutar la lógica de decodificación restringida token a token.
    * Validar las estructuras mediante **Pydantic** y volcar los resultados en el archivo de salida.

---

## 4. ¿Por qué necesitas Pydantic y NumPy?

### Pydantic
* **Validación de entrada:** Permite validar la estructura y el contenido del archivo `functions_definition.json` al leerlo, garantizando que campos como `name` o `parameters` cumplan con las restricciones del sistema.
* **Validación de salida:** Permite validar de forma estricta cada objeto final obtenido (`prompt`, `fn_name`, `args`) antes de su serialización y escritura.
* **Robustez:** Proporciona mensajes de error claros y estructurados si un campo está ausente o si los tipos de datos no coinciden, previniendo fallos silenciosos durante el proceso de evaluación.

### NumPy
* **Manipulación de Logits:** Ideal para trabajar eficientemente con las listas de floats de gran tamaño devueltas por el modelo.
* **Operaciones vectorizadas:** Facilita la aplicación de máscaras vectorizadas complejas, permitiendo establecer los logits de tokens inválidos a `-inf` (o un valor significativamente bajo) de forma casi instantánea.
* **Eficiencia:** Agiliza el cálculo del `argmax` sobre arrays extensos. Aunque `get_logits` devuelve una lista nativa de Python, su conversión a un array de NumPy optimiza las operaciones matemáticas.

---

## 5. Qué significa “token” en este repositorio

* Un **token** es la unidad mínima de procesamiento textual que utiliza el tokenizador del modelo Hugging Face cargado por `Small_LLM_Model`.
* **No equivalen a caracteres individuales:** Pueden representar subpalabras, caracteres únicos, símbolos de puntuación o incluso espacios en blanco.
* En este proyecto es imperativo **operar a nivel de tokens (IDs)** debido a que la decodificación restringida se ejecuta directamente sobre las distribuciones de probabilidad (logits) de los tokens del vocabulario.

### Herramientas del repositorio:
* `m.encode(text)` $\rightarrow$ Convierte una cadena de texto en una secuencia de token IDs.
* `m.decode(ids)` $\rightarrow$ Transforma una secuencia de IDs de nuevo a texto legible.
* `m.get_path_to_vocab_file()` $\rightarrow$ Proporciona la ubicación del archivo de vocabulario, clave para auditar cómo se tokenizan símbolos, palabras o prefijos específicos.

---

## 6. Acciones concretas que puedes ejecutar ahora

Comprobaciones y experimentos recomendados en un entorno interactivo (REPL de Python) donde `uv` o el modelo estén disponibles:

1.  **Inicializar el wrapper y localizar el vocabulario:**
    ```python
    from llm_sdk.llm_sdk import Small_LLM_Model
    m = Small_LLM_Model()
    print(m.get_path_to_vocab_file()) # Muestra la ruta local a vocab.json
    ```
    *Acción:* Abre el archivo `vocab.json` generado y busca la codificación de los siguientes elementos: `'{'`, `'}'`, `'"'`, `':'`, `','`, dígitos (`'0'`, `'1'`, etc.) y los prefijos de los nombres de las funciones (ej. `'fn'`, `'fn_add'`, `'fn_add_numbers'`).

2.  **Probar la tokenización de cadenas clave:**
    ```python
    print(m.encode('fn_add_numbers')) # Observa cómo se fragmenta en sub-tokens
    print(m.encode('"'))
    print(m.encode('{'))
    print(m.encode('123'))
    print(m.encode('a'))
    ```
    *Acción:* Evalúa si las comillas, llaves o números se consolidan como tokens independientes o si forman parte de bloques más grandes.

3.  **Inspeccionar la estructura de los logits:**
    ```python
    ids = m.encode('What is the sum of 2 and 3?').tolist()[0]
    logits = m.get_logits_from_input_ids(ids)
    print(len(logits)) # Debería coincidir con el vocab_size del modelo
    print(logits[:20]) # Examina las primeras 20 puntuaciones sin procesar
    ```

4.  **Estrategia de restricción de tokens:**
    Para limitar la generación exclusivamente a los 5 nombres de función válidos, es necesario mapear de antemano cada nombre a su secuencia de tokens correspondiente (usando `encode`). Durante la generación paso a paso, se deberán enmascarar todos los tokens excepto aquellos que sigan extendiendo un prefijo válido de dichas secuencias.

---

## 7. Diseño conceptual (Construcción de la decodificación restringida)

Lógica estructurada para la implementación del algoritmo de generación:

1.  **Objetivo:** Generar de manera secuencial (token a token) un JSON válido que contenga las llaves `"prompt"`, `"fn_name"` y `"args"`. Alternativamente, se puede restringir únicamente la generación del valor de `fn_name` y sus `args`, construyendo posteriormente la estructura final mediante código Python estándar.
2.  **Flujo del Autómata / Estados de Tokenización:**
    * **Estado Inicial (Selección de Función):**
        * Precomputar las secuencias de tokens para cada una de las 5 funciones válidas (`m.encode(name)`).
        * Generar token por token aplicando una máscara sobre los logits: solo se permiten tokens que coincidan con la continuación legítima de alguna de las secuencias de nombres precomputadas.
        * Una vez que se completa una secuencia exacta, se bloquea la selección y se transiciona al siguiente estado.
    * **Estado de Argumentos (`args`):**
        * Tomando como base el esquema de la función seleccionada, se identifican los parámetros requeridos y sus tipos de datos.
        * Se restringe la generación para obligar a cumplir con la sintaxis de un JSON estructurado. El autómata debe controlar transiciones como: apertura de llave `{`, escritura de claves de parámetros `"campo"`, dos puntos `:`, el valor según su tipo de dato, y separadores `,` o cierre `}`.
        * *Manejo de tipos de datos en la máscara:*
            * **`number`**: Solo se habilitan tokens numéricos (dígitos, punto decimal, signos `+`/`-`).
            * **`string`**: Se permiten tokens que construyan texto delimitado por comillas, gestionando secuencias de escape.
            * **`boolean`**: Solo se habilita la generación de los literales `true` o `false`.
    * **Consulta y Enmascaramiento:** En cada paso de la generación, se invocará `get_logits_from_input_ids`, y se aplicará una máscara utilizando NumPy para vetar (fijar a $-\infty$) los tokens que violen el estado sintáctico actual.
3.  **Validación y Mitigación:**
    * Al alcanzar el estado final de generación, se parsea el JSON resultante y se valida formalmente con **Pydantic**.
    * En caso de fallo en la validación, se debe capturar la excepción y ejecutar una estrategia de contingencia (ej. reintento de generación o aplicación de heurísticas de corrección).
4.  **Estrategia de Implementación por Fases (Recomendado):**
    * *Fase 1:* Implementar de forma estricta la selección guiada de la función (`fn_name`), dado que la lista de opciones es cerrada y acotada.
    * *Fase 2 (Base de referencia):* Permitir que el modelo genere los argumentos (`args`) de forma libre en formato JSON. Posteriormente, delegar en Pydantic el parseo y forzar la coerción de tipos (ej. convertir una string numérica a number si es posible).
    * *Fase 3 (Refinamiento):* Sustituir la generación libre de la Fase 2 por una decodificación restringida token a token aplicada a los tipos de los argumentos.
    * *Criterio de éxito:* Asegurar una precisión superior al 95% en la selección de la función y robustecer el esquema de Pydantic para garantizar la integridad del output.

---

## 8. Checklist del Mínimo Viable (MVP)

- [ ] Implementar la lectura y el parseo seguro de `functions_definition.json` mapeando los datos a un modelo Pydantic (`FunctionDef`).
- [ ] Implementar el bucle de lectura e iteración sobre los prompts de prueba.
- [ ] Inicializar la instancia de `Small_LLM_Model` controlando los tiempos de carga del modelo.
- [ ] **Desarrollar la selección de funciones:**
  - [ ] Mapear los nombres de las funciones a sus respectivas secuencias de tokens.
  - [ ] Implementar el bucle de generación restringido token a token para `fn_name`.
- [ ] **Desarrollar el extractor de argumentos:**
  - [ ] Diseñar el mecanismo de extracción de `args` dado el prompt y la función (Fase 2: Libre + Coerción / Fase 3: Restringido por tipo).
- [ ] Validar la estructura del objeto de salida de cada iteración utilizando Pydantic antes de su consolidación.
- [ ] Serializar y guardar la colección de resultados en `data/output/function_calling_results.json`.

---

## 9. Posibles problemas prácticos y soluciones

* **Descargas lentas o almacenamiento insuficiente en Hugging Face:** El SDK requiere descargar los pesos del modelo y el tokenizador localmente. Si existen restricciones de red o espacio, se puede desarrollar temporalmente un tokenizador "mock" para pruebas unitarias de flujo, aunque cabe destacar que la decodificación restringida legítima requiere de los logits reales provistos por el modelo.
* **Tokenización inesperada en identificadores (Fragmentación):** Al evaluar y restringir los nombres de las funciones, es crítico operar con secuencias de tokens (listas de enteros resultantes de `encode`) y no asumir que los nombres se traducirán en un único token uniforme.
* **Gestión de comillas y caracteres de escape:** Durante la generación sintáctica de strings en los argumentos, el autómata debe prever caracteres de escape para evitar cierres prematuros de cadenas. La validación con Pydantic actuará como red de seguridad identificando JSONs mal formados.
* **Interpretación del vocabulario:** Usar de manera sistemática `get_path_to_vocab_file` para examinar de forma precisa la representación interna exacta de caracteres de control como espacios y delimitadores.












PROMPT MAESTRO:

He leído ambos documentos en detalle. Aquí tienes el prompt maestro:

PROMPT MAESTRO — call-me-maybe (42 Madrid)
# PROJECT: call-me-maybe — Function Calling with Constrained Decoding
# 42 Madrid | Python 3.10+ | uv | Qwen/Qwen3-0.6B

## CONTEXT
I'm building a function-calling system that translates natural language prompts into
structured JSON function calls using a local LLM with CONSTRAINED DECODING.
This is NOT about prompting the model to output JSON — it's about intercepting
the token generation process and masking invalid tokens at each step.

## TECH STACK & HARD CONSTRAINTS
- Python 3.10+
- Package manager: uv (uv sync to install, NO pip install manually)
- Allowed packages: pydantic, numpy, json (stdlib), llm_sdk (local package)
- FORBIDDEN: dspy, pytorch, huggingface, transformers, outlines, or any similar lib
- All classes MUST use pydantic for validation
- Type hints everywhere + mypy passing (--warn-return-any --warn-unused-ignores
  --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs)
- flake8 compliant
- Docstrings on all functions/classes (PEP 257, Google or NumPy style)
- No private methods/attributes of llm_sdk (no _underscore access)
- Model: Qwen/Qwen3-0.6B (default, must work with it)

## PROJECT STRUCTURE (mandatory)
call-me-maybe/
├── src/
│   ├── __init__.py
│   └── __main__.py          # entry point
├── llm_sdk/                 # copied local package (NOT installed via pypi)
│   └── __init__.py          # contains Small_LLM_Model class
├── data/
│   └── input/
│       ├── function_calling_tests.json
│       └── function_definitions.json
├── pyproject.toml
├── uv.lock
├── Makefile
├── README.md
└── .gitignore
# NOTE: data/output/ is NOT committed to git — generated at runtime

## SDK API (llm_sdk.Small_LLM_Model) — PUBLIC METHODS ONLY
- get_logits_from_input_ids(input_ids: Tensor) -> Tensor
  Returns raw logits for all vocab tokens given current token sequence
- get_path_to_vocabulary_json() -> str
  Returns path to JSON file mapping token_id -> token_string
- encode(text: str) -> List[int]
  Tokenizes text to list of token IDs
- decode(token_ids: List[int]) -> str   [optional]
  Converts token IDs back to text

## INPUT FILES
### function_calling_tests.json
Array of natural language prompts:
["What is the sum of 2 and 3?", "Reverse the string 'hello'", ...]

### function_definitions.json
Array of function specs:
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  },
  {
    "name": "fn_reverse_string",
    "description": "Reverse a string",
    "parameters": {"s": {"type": "string"}},
    "returns": {"type": "string"}
  }
]
# Supported types: number, string, boolean (possibly more at evaluation time)
# Input files may be missing or contain invalid JSON — handle gracefully

## OUTPUT FILE
Path: data/output/function_calling_results.json (default)
         or custom path via --output flag
Format:
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
VALIDATION RULES:
- 100% valid parseable JSON (no trailing commas, no comments)
- Exactly keys: prompt, fn_name, args — no extras
- fn_name must match a name from function_definitions.json exactly
- args keys and types must match the function spec exactly
- number -> float, string -> str, boolean -> bool
- All required args present, no extra args

## EXECUTION
uv run python -m src [--input <path>] [--output <path>]
# Defaults: input=data/input/function_calling_tests.json
#           output=data/output/function_calling_results.json
# The output directory must be created if it doesn't exist

## MAKEFILE (mandatory targets)
install:     uv sync
run:         uv run python -m src
debug:       uv run python -m src (with pdb)
clean:       remove __pycache__, .mypy_cache
lint:        flake8 . && mypy . --warn-return-any --warn-unused-ignores
             --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
lint-strict: flake8 . && mypy . --strict

## THE CORE ALGORITHM — CONSTRAINED DECODING (mandatory, not optional)
This is the heart of the project. DO NOT just prompt the model and hope for JSON.

### How it works:
1. Build a prompt that gives the LLM context: available functions + natural language query
2. Call model.encode(prompt) -> input_ids (List[int])
3. Convert to Tensor
4. Loop token by token:
   a. Call model.get_logits_from_input_ids(input_ids) -> logits (Tensor, shape [vocab_size])
   b. Determine which tokens are VALID at the current generation position
      (based on what JSON has been generated so far and the expected schema)
   c. Set logits of ALL invalid tokens to -infinity (float('-inf'))
   d. Select the highest-scoring remaining token (argmax)
   e. Append selected token_id to input_ids
   f. Repeat until JSON is complete (closing } detected)
5. Decode the generated token sequence to get the final JSON string
6. Parse and validate against the function schema

### Schema enforcement (not just JSON validity):
The constrained decoder must enforce the EXACT output schema:
{
  "fn_name": <one of the known function names>,
  "args": {
    "<param1>": <value of correct type>,
    ...
  }
}
At each generation step, only tokens that keep the output valid
AND schema-compliant are allowed. Example:
- When generating fn_name value: only tokens that form one of the known function names
- When generating a number arg: only digit/decimal tokens
- When generating a string arg: any token until closing quote
- When generating a boolean arg: only "true" or "false" tokens

### Vocabulary file usage:
- Load get_path_to_vocabulary_json() once at startup
- Build a dict: token_id (int) -> token_string (str)
- Use this mapping to check what string each token would add to the output,
  enabling prefix-based valid token filtering at each step

## FUNCTION SELECTION (must use LLM, not heuristics)
The LLM must decide which function to call. Approach:
- Include function names + descriptions in the prompt
- During constrained decoding of fn_name field, restrict tokens to only
  those that form valid prefixes/completions of known function names
- The model's logits (after masking) determine which function wins
- This is NOT keyword matching or string similarity — it's the model choosing

## ERROR HANDLING (all must be handled, no crashes)
- Missing input file -> clear error message, exit cleanly
- Invalid JSON in input -> clear error message, skip or exit cleanly  
- Model loading failure -> clear error message
- Unknown function in output -> log warning, skip entry
- Output directory doesn't exist -> create it
- Any unexpected exception -> catch at top level, print message, exit with code 1

## PERFORMANCE TARGETS
- >95% correct function selection
- >90% correct argument extraction
- 100% valid JSON output (guaranteed by constrained decoding)
- All prompts processed in <5 minutes

## README.md (mandatory sections)
First line (italic): *Este proyecto ha sido creado como parte del currículo de 42 por <login>.*
Sections required:
1. Descripción — what the project does and why
2. Instrucciones — how to install and run
3. Recursos — references + AI usage description (what tasks, which parts)
4. Explicación del algoritmo — detailed constrained decoding explanation
5. Decisiones de diseño — key implementation choices and why
6. Análisis de rendimiento — accuracy, speed, reliability results
7. Retos encontrados — difficulties and how they were solved
8. Estrategia de pruebas — how the implementation was validated
9. Ejemplos de uso — concrete usage examples with real input/output

## BONUS FEATURES (for maximum score)
1. Support multiple LLM models (not just Qwen3-0.6B) — switchable via arg or config
2. RECODE THE TOKENIZER: implement encode() and optionally decode() yourself
   using get_logits_from_input_ids + get_path_to_vocabulary_json ONLY
   (do NOT use the SDK's encode/decode in main logic — implement your own)
3. Advanced error recovery (retry logic, fallback strategies)
4. Performance optimizations: token prefix caching, vocabulary pre-filtering
5. Comprehensive pytest test suite with edge cases
6. Visualization/logging of the generation process (which tokens were masked, why)
7. Support for complex nested function arguments
8. Public implementation of tokenizer encode + decode methods integrated
   with constrained decoding pipeline

## EVALUATION CHECKLIST (what the corrector checks)
- [ ] uv sync works cleanly
- [ ] uv run python -m src runs without crash
- [ ] output JSON is 100% parseable
- [ ] constrained decoding is actually implemented (not just prompting)
- [ ] pydantic used for all classes
- [ ] no private SDK methods used
- [ ] Qwen/Qwen3-0.6B is the default model
- [ ] handles missing/invalid input files gracefully
- [ ] handles prompts that match no function
- [ ] >90% function selection accuracy on test set
- [ ] moulinette passes (uv run python -m moulinette grade_student_answers ...)
- [ ] README has all required sections
- [ ] type hints + mypy passes
- [ ] flake8 passes

## WHAT TO BUILD NOW
Implement src/__main__.py and supporting modules with:
1. CLI argument parsing (--input, --output)
2. Input file loading with error handling (pydantic models for validation)
3. LLM wrapper/initialization
4. Vocabulary loading and token->string mapping
5. Prompt builder (includes function descriptions for LLM context)
6. Constrained decoder (the core loop)
7. Schema-aware token masker (JSON state machine + schema enforcement)
8. Output writer (validated JSON)
9. Full error handling at every layer

Current project state: uv project initialized, src/__init__.py and src/__main__.py
exist as entry point, pydantic+numpy as prod deps, flake8+mypy+pytest as dev deps,
llm_sdk installed as local package via uv add ./llm_sdk.
Next step needed: examine llm_sdk/__init__.py to understand exact method signatures
and Tensor type before implementing the constrained decoder.



La estructura que tiene sentido para este proyecto es algo así:
src/
├── __init__.py
├── __main__.py          # orquesta todo, punto de entrada
├── object/              # (ya tienes esto)
│   ├── __init__.py
│   └── parse.py         # Config, paths — ya hecho
├── models/              # modelos pydantic de datos
│   ├── __init__.py
│   ├── function_def.py  # FunctionDefinition, Parameter...
│   └── result.py        # FunctionCallResult (el output)
├── io/                  # lectura/escritura de archivos
│   ├── __init__.py
│   ├── reader.py        # cargar y validar los JSONs de entrada
│   └── writer.py        # escribir el JSON de salida
├── llm/                 # todo lo del modelo
│   ├── __init__.py
│   ├── wrapper.py       # inicializar Small_LLM_Model, vocabulario
│   ├── prompt.py        # construir el prompt para el LLM
│   └── decoder.py       # EL CORAZON: constrained decoding loop
└── pipeline.py          # une todo: recibe paths, devuelve resultados

El flujo completo de arriba a abajo
__main__.py → parsea args → llama a pipeline.py → que:

Lee functions_definition.json → lista de FunctionDefinition (pydantic)
Lee function_calling_tests.json → lista de prompts
Inicializa el modelo + carga vocabulario
Para cada prompt: construye prompt → constrained decode → obtiene FunctionCallResult
Escribe la lista de resultados al JSON de salida


Lo que tienes que entender del SDK antes de continuar
Hay una cosa importante que has de saber ahora que leí el SDK real:

get_logits_from_input_ids recibe una lista de ints y devuelve una lista de floats (no tensores — ya los convierte internamente)
encode devuelve un tensor 2D (batch de 1), no una lista — tendrás que hacer .tolist()[0] para trabajar con él
El vocabulario está en get_path_to_vocab_file() (no get_path_to_vocabulary_json() como dice el subject — el nombre real es diferente), es un JSON tipo {"token_string": token_id}
También hay get_path_to_merges_file() y get_path_to_tokenizer_file() que son para el bonus de reimplementar el tokenizador


Orden de construcción recomendado

Modelos pydantic (models/) — define cómo se ve un FunctionDefinition y un FunctionCallResult. Sin lógica, solo estructura y validación.
IO (io/reader.py) — leer los dos JSONs con manejo de errores (archivo no existe, JSON inválido), devolver los modelos pydantic.
LLM wrapper (llm/wrapper.py) — inicializar el modelo, cargar el vocabulario en memoria como dict {token_id: token_string}.
Prompt builder (llm/prompt.py) — dado un prompt de usuario + lista de funciones disponibles, construir el texto que le vas a dar al LLM.
Constrained decoder (llm/decoder.py) — el núcleo duro.
Pipeline + writer — unirlo todo y escribir el output.