# Retrieval & Memory Lab — Informe

**Corpus:** `python-docs-es` (documentación de Python en español), versiones 3.13 y 3.12, por la
ruta del **HTML compilado**.
**Generador:** `qwen3:8b` en Ollama, `num_ctx = 8192` fijo en todas las etapas.
**Pregunta del proyecto:** cómo se reparte una ventana finita entre lo que dice el corpus y lo que
se dijo en la conversación.
**Etapas hechas:** 0, 1, 2, 3 (acotada a 3 filas), 4, 5, 6. Sin Etapa 7.
**Perfil de los números de este informe:** `full` salvo donde se diga `smoke`.

> **Estado de este documento.** Las secciones 1-3 y 8 están cerradas: sus números salen de corridas
> que ya existen (`artifacts/full/run_report.json`). Las secciones 4-7 son el esqueleto de las
> etapas que todavía no corrieron en GPU: cada hueco dice **de qué archivo y de qué campo** sale el
> número, para que llenarlo sea mecánico y no una segunda interpretación. Ningún hueco se llena a
> ojo: si una corrida no existe, la fila dice «no corrió».

---

## 0 · Cómo leer este informe

Tres convenciones que valen para todas las tablas:

1. **Las comparaciones son pareadas.** Con 45-53 ítems, el IC95 de una fila sola abarca ~±10
   puntos: dos filas que difieren en 4 puntos no se distinguen leyendo promedios. Se informa
   `gana / pierde / empata` sobre los ítems **discordantes** y el test de signos bilateral
   (`noise.paired`). Donde un resultado da nulo se dice **cuántos** discordantes hacían falta para
   detectarlo (`noise.detectable` ⇒ 6 todos para el mismo lado para bajar de 0,05).
2. **«No llegó» no es lo mismo que «llegó y no se usó».** Toda tabla de generación separa el techo
   de recuperación del uso que el modelo hizo de ese techo.
3. **Las cuentas se informan como cuentas** (3 de 4), no como porcentajes, cuando el denominador
   es chico. Un 75 % sobre 4 ítems es un 3.

---

## 1 · Decisiones de diseño y su justificación

El detalle, con fechas y alternativas descartadas, está en `decision_log.md` (7 entradas). Acá el
resumen de las que condicionaron todo lo demás:

| Decisión | Alternativa descartada | Qué nos llevó ahí |
|---|---|---|
| **Corpus `python-docs-es` por HTML compilado** | los `.po` del repo de traducción | el `.po` es la unidad de traducción, no la de lectura: entradas sueltas, `fuzzy`, obsoletas. El HTML es lo que un usuario lee (Entrada 2) |
| **Dos versiones (3.13 y 3.12), no una** | una sola versión | el corpus necesita documentos confundibles para que la recuperación tenga trabajo; las páginas gemelas comparten de 0,35 a 1,00 de 5-gramas (MinHash, Etapa 0) |
| **Todo en una RTX 3050 de 6 GB** | el reparto híbrido con una T4 de Colab (plan original, Entrada 1) | cada etapa que cruza la frontera entre dos máquinas puede desalinear el `documents_sha256` o el `index_key`, y el chequeo que lo detecta corre *después* de gastar las llamadas. El costo: `qwen3:8b` no entra en 6 GB (~5,2 de pesos + ~1,2 de caché KV a 8.192), así que la corrida se reporta con lo que diga `llm.placement()` (Entrada 7) |
| **Etapa 3 acotada a 3 filas** | el barrido completo de piezas | el presupuesto de la Entrada 1 no daba para más filas *y* llegar a la Etapa 6, que es donde está la pregunta del proyecto |
| **Sin Etapa 7** | juez LLM + framework | «menos etapas bien razonadas» (rúbrica §5.2); el presupuesto se gastó en memoria |
| **`num_ctx = 8192` en TODAS las etapas** | ajustarlo por etapa | una fila que cambia la ventana no mide su estrategia de contexto, mide la ventana |
| **El conjunto de evaluación congelado antes de cualquier resultado** | congelarlo al final | es lo único que hace que la división dev/test signifique algo (Entrada 5) |

**Lo que el repositorio hace y lo que es nuestro.** El andamiaje (`llm.py`, `store.py`,
`console.py`, `environment.py`, `schema.py`) viene del repo base. Nuestro: `00b_corpus_audit_ours`,
`01_documents`, `02_eval_set`, `03_retrieval`, `04_generation`, `05_dialogue`, `06_memory`,
`retrieval.py`, `prompting.py`, `lexical.py`, `noise.py`, `eval/`, `corpus_card.md` y
`decision_log.md`.

### Las preguntas de la Etapa 0

- **De las siete propiedades, cuáles no cumple el corpus y dónde se paga.**
  → `00b_corpus_audit_ours.ipynb`, §final. *(resumen: PENDIENTE de pegar — sale de
  `artifacts/full/stage0_ours.json`)*
- **¿El subconjunto entra en la ventana?** No: 3.13 solo son **1.884.998 tokens** estimados en 385
  páginas (`run_report.json → 00b_corpus_audit_ours.notes`). Contra `num_ctx = 8192`, el corpus es
  **230 ventanas**. Por eso hay recuperación y no un prompt largo.
- **Cuánto de esto ya leyó el generador, y cómo distinguimos recuperación de recuerdo.** Las 5
  preguntas de **fuera del corpus** (términos de 3.14 verificados ausentes con conteo literal) y la
  fila de **libro cerrado** de la Etapa 4 son la medida: lo que el modelo contesta sin pasajes es
  recuerdo.
- **Qué hace confundibles a dos documentos, y si queremos conservarlo.** La misma página en dos
  versiones. Sí: es la propiedad que hace que el gold tenga que ser un **conjunto** y que un
  acierto a nivel documento no signifique nada.

---

## 2 · La ficha del corpus (corpus card)

El documento completo es `corpus_card.md` (va en el zip). Lo que no puede faltar acá:

- **695 documentos**, `documents_sha256 = 6c58d2cf300c1ff61649139369495ff70ad8766b15acabd8af74a0d5228aa7c6`.
- **Qué cambió la limpieza**, en caracteres (`run_report.json → 01_documents.notes.cleaning_chars`):
  marca `¶` → salto de línea **53.440**; firma partida (`pathlib. Path` → `pathlib.Path`)
  **14.148**; etiqueta de nota de versión al español **−10.314**; NFC **0**.
- **La regla que cambió más, y el hallazgo que la amplió:** la de firmas partidas. Derivar las
  preguntas de la Etapa 2 obligó a parsear la firma como la página la muestra, y ahí se vio que la
  regla cubría solo la forma calificada (`pathlib. Path. open`) y no la corta (`Path. open`), que
  es la habitual: quedaban **3.364 ocurrencias sin tocar**, más del doble de las que había
  arreglado. Se extendió a todos los sufijos del `anchor_id` y se iteró **a punto fijo** (las
  firmas con dos puntos partidos, `socket. socket. type`, no se arreglan en una pasada). Contador
  final de restos: **0**.
- **El defecto que un loader ingenuo pasa sin error:** exactamente ese. No tira excepción, no
  rompe el parseo: deja el identificador partido en dos tokens, lo que degrada BM25 justo en las
  preguntas de identificador, que son las que más dependen de él.
- **Documento y sección:** un documento es una página; una sección es un bloque con `anchor_id`.
  `section_id = doc_id#anchor`. Si una etapa posterior no está de acuerdo, el gold deja de ser
  comparable: por eso el gold se etiqueta **a nivel sección** y la contención se verifica por
  spans.
- **`doc_id` ante una descarga nueva:** los `doc_id` son `py3.13:library/itertools`, estables
  mientras la ruta del archivo no cambie. El **contenido** sí cambia, y ahí muerde `FROZEN.json`:
  el hash del corpus ya cambió dos veces (`09c6862a…` → `ca94ffe0…` → `6c58d2cf…`) y las dos veces
  lo detectó la carga.
- **Casi duplicados:** se conservan a propósito (ver §1). Compromete al conjunto de evaluación a
  tener gold de conjunto: **48 de 48** ítems con respuesta lo tienen de las dos versiones
  (`gold_set_fraction = 1.0`), y los dos únicos casos que colapsan a una versión están registrados
  (`itertools.batched`, que solo existe en 3.12).

---

## 3 · El conjunto de evaluación y su chequeo de contaminación

**Qué se congeló** (`eval/` + `FROZEN.json`, antes de cualquier resultado):

| Pieza | Cantidad | Detalle |
|---|---|---|
| ítems de un solo turno | **53** | temáticos 20, identificador 18, metadatos 10, fuera del corpus 5 |
| diálogos | **9** | 36 turnos |
| escenarios de memoria | **4** | 11 sesiones, 25 sondas |
| división | dev 8 / test 45 | la dev existe para mirarla y romperla; la test no se miró hasta la Etapa 3 |

**Habilidades de memoria cubiertas** (`run_report.json → 02_eval_set.notes.probes_by_ability`):
extracción 6, preferencia 3, temporal 4, actualización de conocimiento 4, multi-sesión 4,
abstención 4. **Lo que quedó afuera:** memoria procedural aprendida de corregir al asistente
(requiere varias sesiones con feedback), y conflictos entre dos hechos igual de recientes.

**El chequeo de contaminación.** Cada pregunta declara su procedencia por ítem, no promediada:
**derivadas** (36: identificador desde anclas únicas y desde la firma que la página muestra;
metadatos desde las notas de versión) y **a mano** (25: las temáticas, escritas desde el título de
sección **con el pasaje cerrado**, y las 5 de fuera del corpus). El solapamiento se mide con **el
mismo tokenizador del canal léxico** (`lexical.py`: un solo stemmer para el chequeo y para BM25, o
el chequeo mide un solapamiento distinto del que infla a BM25). Resultado: el n-grama compartido
más largo es **3 tokens** en las temáticas y 2 en las derivadas, con el umbral en 5 ⇒
**`contamination_flagged = 0`**. Ninguna pregunta importó la redacción del pasaje.

**La prueba que podía salir mal:** BM25 solo resuelve **29 de 48** (60 %) a nivel sección
(temáticas 35 %, identificador 72 %, metadatos 90 %). Si hubiera dado 95 % el conjunto no tendría
margen para que la Etapa 3 mueva nada y había que reescribirlo.

**El candado se probó rompiéndolo:** agregar un renglón a `single_turn.jsonl` hace que `load_eval`
levante `EvalChanged` («53 → 54 rows»). Un congelado que no muerde es un archivo.

**Una pregunta de usuario real que este conjunto no puede medir:** «¿cómo migro este script de
3.12 a 3.13?». Requiere leer varias páginas, decidir qué cambios aplican al código que el usuario
tiene, y priorizarlos; una clave de respuesta de una palabra no dice nada de eso.

**Si `topical` e `identifier` no coinciden sobre una técnica, ¿en cuál confiamos?** En
`identifier`, y queda decidido **ahora**: son las preguntas cuya respuesta es verificable por
contención exacta contra una ancla única, así que su señal es menos discutible. Las temáticas
tienen una clave conjuntiva de una o dos palabras elegida por nosotros, y ahí el desacuerdo puede
ser de la clave y no del sistema.

---

## 4 · Etapa 3 · La tabla de recuperación, con cotas de ruido

**Estado: NO CORRIÓ en perfil `full`.** Código listo y limpio en humo.
**Fuente cuando corra:** `artifacts/full/stage3_rows.json` y `artifacts/full/results/`.

| Fila | hit@k doc | hit@k sección | clave entregada | MRR doc | tokens entregados | ms |
|---|---|---|---|---|---|---|
| 1 · solo BM25 (línea de base) | | | | | | |
| 2 · + denso (fusión) | | | | | | |
| 3 · + reranker | | | | | | |

- **Pareado contra la línea de base** (`noise.paired`): fila 2 vs 1 → `gana/pierde/empata`, `p` =
  ; fila 3 vs 2 → .
- **Qué pieza se ganó su costo por encima de la cota, y qué sacamos:** PENDIENTE.
- **La caja abierta del movimiento más grande:** PENDIENTE — los ítems que cambiaron de resultado
  están en `stage3_rows.json`; hay que leer 3 y decir si ganó por la razón que da su paper.
- **Brecha documento vs sección:** PENDIENTE, y qué predice para la Etapa 4.
- **Tokens igualados: muchos hijos o pocos padres:** PENDIENTE (las filas se llenan al mismo
  `budget_tokens = 3000`, contados en el tokenizador del generador; la tabla imprime los tokens
  realmente entregados y advierte si dos filas difieren).
- **El filtro de metadatos:** PENDIENTE, por categoría. Nota de diseño que ya está decidida: el
  filtro de la Etapa 3 **cae al listado sin filtrar** (`fused = kept or fused`) porque una versión
  que el usuario tipeó puede ser un error de tecleo. Comparar con §7.

---

## 5 · Etapa 4 · Respuestas fundamentadas, y la tabla 2×2

**Estado: NO CORRIÓ.** `04_generation.py` escrito y verificado estáticamente.
**Fuente cuando corra:** `artifacts/full/stage4_rows.json`.

| Fila | correctas | abstenciones | citas ⊆ entregados | citó el gold | fallos | tokens de prompt |
|---|---|---|---|---|---|---|
| libro cerrado (sin pasajes) | | | | | | |
| recuperación (la mejor fila de la Etapa 3) | | | | | | |
| oráculo (la sección gold directo) | | | | | | |

**La tabla 2×2** — llegó el pasaje × respondió bien:

| | respondió bien | respondió mal / se abstuvo |
|---|---|---|
| **el pasaje llegó** | | |
| **el pasaje no llegó** | | |

- **De las preguntas donde llegó el pasaje, cuántas se contestaron bien y cómo eran las fallas:**
  PENDIENTE (las fallas se clasifican con las columnas `inconsistent`, `verdict_ok`,
  `key_contained` de `prompting.score`, sin juez).
- **Brecha oráculo − mejor recuperación**, y **mejor recuperación − libro cerrado:** PENDIENTE.
  Es la medida de cuánto del techo alcanza la generación.
- **Citas inventadas y abstenciones con la respuesta en contexto:** PENDIENTE
  (`cited_not_delivered`, y `abstained` con `key_contained = True`).
- **Un prompt demasiado grande:** PENDIENTE — nuestra cuenta (`expected_prompt_tokens`), la del
  servidor (`prompt_tokens`), y qué contestó. La guarda de pre-vuelo convierte el desborde en un
  `ChatResult` con `error`, que es **su propia categoría**: ni abstención ni respuesta mal.
- **La posición:** PENDIENTE, y si la prueba podía mostrarlo a nuestra escala.

---

## 6 · Etapa 5 · Una ventana, dos fuentes

**Estado: NO CORRIÓ.** `05_dialogue.py` escrito y verificado estáticamente.
**Fuente cuando corra:** `artifacts/full/stage5_rows.json`, `stage5_rewrites.txt`,
`stage5_summaries.txt`.

### 5A · La consulta dentro de la conversación, por dependencia y por turno

| Configuración | sin dependencia | correferencia | elipsis | cambio de tema | turno 1 | turno 4 |
|---|---|---|---|---|---|---|
| último turno solo | | | | | | |
| últimos k turnos concatenados | | | | | | |
| reescritura por el modelo | | | | | | |
| reescritura oráculo | | | | | | |

- **Qué ganó en qué dependencia, y si el ranking cambia con el número de turno:** PENDIENTE.
- **Dónde perdió la reescritura oráculo contra la del modelo, o le ganó:** PENDIENTE (hay 20
  reescrituras volcadas para leer a mano).

### 5B · El reparto de la ventana

| pasajes / historial | correctas | tokens de prompt | llamadas por turno |
|---|---|---|---|
| | | | |

- **Dónde quedó la mejor división, y si la diferencia pasa la cota de ruido:** PENDIENTE.
- **Qué hicieron mal los resúmenes**, en las tres categorías de Wang et al. (hechos inventados,
  relaciones equivocadas, detalles faltantes): PENDIENTE **de una persona** —
  `stage5_summaries.txt` → `stage5_summary_audit.json`.
- **Costo por turno de cada estrategia, y si el ganador vale la pena:** PENDIENTE.

---

## 7 · Etapa 6 · Memoria entre sesiones

**Estado: NO CORRIÓ.** `06_memory.py` escrito; verificado con 32 controles sobre las partes que no
necesitan GPU (vigencia, vecinos, presupuesto de lectura, filtro de versión, solapamiento con la
fuente). **El diseño y sus justificaciones están cerrados** (Entrada 6 del registro) y no dependen
del resultado: eso es lo que sigue.
**Fuente cuando corra:** `artifacts/full/stage6_rows.json`, `stage6_records.txt`,
`stage6_filter.txt`, `memory/records.jsonl`, `memory/ops.jsonl`.

### El régimen, declarado antes de medir

A nuestra escala **la reproducción entra siempre**: 56 tokens por sesión en promedio, y el estado
más largo (`s-etl` tras `s3`) son 206 tokens de transcripción — **286** contando lo que el espacio
de memoria cobra encima (32 de encabezado con la fecha simulada, 16 por cada fecha de sesión).
Contra `num_ctx = 8192` o contra el espacio de 300-1.000 tokens que sugiere §7.3, no hay nada que
no entre. Así que el experimento se corre en **dos regímenes del mismo espacio**:

| Régimen | Tokens | Sondas donde la transcripción no entra |
|---|---|---|
| `SLOT_WIDE` | 400 | 0 de 19 |
| `SLOT_TIGHT` | 120 | **15 de 19** |

`SLOT_TIGHT = 120` **no es un despliegue realista** y no se presenta como tal: 120 − 32 − 16 ≈ 72
tokens deja entrar **una sesión** (que mide 56). A 80 tokens también recorta (19 de 19) pero el
corte cae en mitad de un turno, y eso mide dónde cayó la tijera, no qué sesiones conservó la
política. La unidad que la reproducción tira es la sesión; el espacio tiene que poder contener al
menos una.

### Los brazos, y los resultados

| Brazo | correctas | tokens de prompt | llamadas de escritura | sesiones en el espacio |
|---|---|---|---|---|
| sin memoria | | | 0 | 0 |
| reproducción · 400 | | | 0 | |
| reproducción · 120 | | | 0 | |
| almacén · 400 | | | | |
| almacén · 120 | | | | |
| almacén · 400 + filtro de versión | | | | |

**Por habilidad** (en cuentas, nunca en porcentaje):

| Habilidad | sin memoria | reproducción · 400 | almacén · 400 |
|---|---|---|---|
| extracción | | | |
| temporal | | | |
| actualización de conocimiento | | | |
| multi-sesión | | | |
| abstención | | | |
| preferencia | | | |

- **En qué habilidades la memoria le ganó a no tener memoria, y en cuáles la reproducción le ganó
  al almacén, con qué costo en tokens:** PENDIENTE. El costo se informa como **punto de
  equilibrio**: `tokens_de_escritura / tokens_ahorrados_por_sonda` = cuántas sondas hay que hacer
  para amortizar la escritura.
- **Una actualización de conocimiento de punta a punta:** PENDIENTE — el notebook la imprime
  entera (la línea de tiempo del registro, las operaciones que lo escribieron, y qué contestó cada
  brazo). Lo que **ya** está decidido: se **invalida, no se borra** (UPDATE le pone `valid_to` al
  viejo y escribe uno nuevo con `supersedes`), así que la vista «al momento de la sonda» sale de
  comparar timestamps simulados, sin una segunda copia del almacén.
- **Qué guardó el extractor que no era cierto o que no se dijo:** PENDIENTE **de una persona** —
  `stage6_records.txt` → `stage6_record_audit.json`. Ningún script puede decidir si un recuerdo es
  cierto. Lo que el script sí cuenta: marcas de razonamiento en el texto del registro, coberturas
  («probablemente», «supongo»), y el solapamiento de palabras de contenido con la sesión de
  origen. El campo `analysis` existe en los dos schemas, va **primero**, y **no se copia nunca** al
  registro: es la trampa de guardar una hipótesis como hecho.
- **¿La memoria mejoró la recuperación como filtro, o la perjudicó?** Lo perjudicó, y está medido
  antes de correr el generador: el usuario de `s-etl` arranca en Python **3.11**, el corpus solo
  tiene 3.12 y 3.13, así que `allow_for("3.11")` devuelve el conjunto **vacío** y `retrieve`
  entrega **cero** pasajes. Es la señal, no un error: un filtro de memoria que deja cero debe dejar
  cero. Lo que elimina es justamente el pasaje de `itertools` de 3.12 con el que el asistente
  podría decir «eso no existe en tu versión». Es la política **opuesta** a la del filtro de la
  Etapa 3 (§4), y las dos se justifican: una versión que el **usuario tipeó** puede ser un error de
  tecleo; una versión que el **sistema guardó** es una afirmación propia sobre su usuario.
- **¿En qué punto el escenario deja de entrar en la ventana, y qué cambia?** No deja de entrar. A
  56 tokens por sesión, la transcripción cruza el espacio de 400 a las ~7 sesiones y el lugar que
  queda en la ventana (`8192 − 700 de reserva − prompt vacío − 1000 de pasajes`) a las ~90. Por eso
  hay dos regímenes en vez de una afirmación sobre un cruce que estos escenarios no tienen.

### El hallazgo que no esperábamos (contribución original)

La fórmula de puntaje que sugiere el enunciado —`minmax(recencia) + minmax(importancia) +
minmax(relevancia)`— **se degrada a dos términos y un bonus** cuando las sesiones están separadas
por meses, que es el caso realista de un asistente personal. Con `0.995/hora`, el registro más
nuevo puntúa ~1 de recencia y el segundo ~2e-4: después del min-max eso **no es un orden por
antigüedad, es un one-hot del último registro**. Los rangos de los tres términos se imprimen
**antes** de normalizar (el chequeo que el enunciado pide) y el informe lo dice así, en vez de
hablar de «tres señales combinadas».

---

## 8 · Cómo usamos IA en este trabajo

PENDIENTE DE ESCRIBIR A MANO (§2.3 del enunciado pide un párrafo breve y honesto). Lo que hay que
cubrir: qué se usó, para qué, y qué se verificó a mano antes de creerlo. Ya está en el registro de
decisiones la frase correspondiente (Entrada 1, «Sobre el uso de IA en este trabajo») — hay que
reconciliar las dos y que digan lo mismo.

---

## 9 · Lo que este trabajo no puede decir

- **Si el asistente le sirve a alguien.** 53 ítems con clave de respuesta miden si la respuesta
  contiene la palabra correcta; no si el párrafo alrededor es útil, ni si la cita es la que un
  humano habría elegido entre dos pasajes igual de buenos.
- **Nada con 19 sondas de 3 escenarios es 19 observaciones independientes.** Por eso la Etapa 6 no
  publica intervalo por sonda, y las comparaciones van pareadas con la cota declarada.
- **Las 25 preguntas escritas a mano las escribimos nosotros**, que ya decidimos qué nos parece una
  buena pregunta de la documentación de Python.
- **Lo pendiente de una persona está listado, no escondido:** las 40 atribuciones nota→definición
  (10 a leer), 2-3 diffs de limpieza, las diez páginas de `stage0_read_me/`, las 5 preguntas de
  libro cerrado a corregir, las 20 reescrituras de la Etapa 5, la auditoría de resúmenes y la
  auditoría de registros de memoria.

---

## Anexo · Qué falta correr, y en qué orden

Todo lo de abajo necesita **Ollama con `qwen3:8b`** y GPU. Los notebooks son el entregable (con
salidas), así que el orden de la corrida es el orden de los notebooks.

```bash
# 0. el servidor, y el modelo cargado una sola vez
ollama serve &
ollama pull qwen3:8b

# 1. perfil full, etapa por etapa, cada una en su proceso (y su log)
SMOKE_TEST=0 python tools/run_stages.py --from 03 --to 06

# 2. los notebooks CON salidas (el zip los pide así): ejecutarlos, no regenerarlos
#    (regenerar con py_to_notebook.py borra las salidas: eso es para commitear limpio)
#    VS Code / Jupyter / Colab: abrir cada .ipynb y correr todo, de 03 a 06.

# 3. empaquetar
python -c "import store, environment as e; store.pack_for_review(e.artifacts_dir(False))"
#    y agregar a mano al zip: eval/, corpus_card.md, decision_log.md, informe.md,
#    y los .ipynb CON salidas
```

**Antes de la corrida larga, tres chequeos de 5 minutos:**

```bash
python verify_setup.py                      # servidor, modelo, VRAM, tokenizador
SMOKE_TEST=1 python tools/run_stages.py --from 03 --to 06   # humo de punta a punta
python tools/py_to_notebook.py --check      # los .ipynb al día con los .py

# y el que decide si la corrida dura 2 horas o 10 (mirar la línea `placement`):
python -c "import llm; llm.setup('qwen3:8b', num_ctx=8192)"
```

`placement` es el único que distingue las dos corridas posibles en esta placa. Si dice `100% GPU`,
nada que hacer. Si dice `SPLIT x% CPU / y% GPU`, el modelo no entró en los 6 GB: **las respuestas
son las mismas y no hay error**, solo varias veces más lento — de 1,5-2,5 h a 5-12 h sobre ~480
llamadas. La Entrada 7 del registro decide **aceptar el split y reportarlo**, porque la comparación
entre brazos es interna a cada etapa y el split afecta a todos los brazos igual: cuesta horas, no
validez. Dos cosas que **no** se tocan: `num_ctx` (a 4.096 la Etapa 4 recortaría su entrega de
pasajes en silencio) y el embebedor (moverlo a CPU cambia el `dtype` del `encoder_spec`, y con él la
clave de `store.cached_matrix`: re-encodearía las dos matrices de hijos desde cero).

**Riesgos conocidos de la corrida, con su mitigación ya en el código:** la caché de llamadas
(`artifacts/full/llm_cache/`) hace que una desconexión no cueste la etapa entera; `ResultsLog`
retoma por fila; `FROZEN.json` corta la corrida si el corpus cambió; y el pre-vuelo de contexto
convierte un prompt demasiado grande en una fila con `error` en vez de tirar la etapa.
