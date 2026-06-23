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