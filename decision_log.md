# Registro de decisiones

Retrieval & Memory Lab · IA Generativa Avanzada y Sistemas Multi-Agente · UTDT

Una entrada por decisión, en orden cronológico. Lo que el §2.4 pide: el presupuesto que
estimamos al principio y el plan, qué probamos, y qué recortamos y por qué. Terminar en un
lugar distinto del plan es esperable; cuando pase, queda escrito acá junto con qué nos hizo
cambiar de opinión.

---

## 2026-10-24 · Entrada 1 — El presupuesto, antes de escribir código

### El hardware que realmente tenemos

El enunciado asume una T4 de Colab (15 GB). Nuestro hardware de escritorio es una **RTX 3050
de 6 GB**. Esa diferencia no es un detalle de logística: decide qué modelo podemos correr y
con qué `num_ctx`, así que la cuenta va primero.

La caché KV de qwen3 cuesta ~144 KiB por token en fp16 (36 capas × 8 cabezas KV × 128
dimensiones × 2 por K y V × 2 bytes), según §7.2:

| Pieza | VRAM |
|---|---|
| qwen3:8b q4_K_M, pesos | ~5,2 GB |
| Caché KV @ `num_ctx` 8192 | ~1,2 GB |
| Buffer de cómputo del servidor | ~0,4 GB |
| **Total @ 8192** | **~6,8 GB** |
| Total @ 4096 (el mínimo que no sería ridículo) | ~6,2 GB |

**qwen3:8b no entra en 6 GB a ningún `num_ctx` útil.** Y el modo en que no entra es el peor
posible: Ollama no falla, offloadea capas a la CPU y sigue andando. El docstring de
`environment.py:265` describe exactamente ese fallo — *"same answers, ~10x slower, no error"*
— y §6.3 lo nombra como una de las trampas del lab. Una corrida así no da ningún número
equivocado; da los mismos números, muchísimo más tarde, y arruina cualquier columna de
latencia.

En cambio, la **Etapa 3 entra sobrada en la 3050**: `BAAI/bge-m3` en fp16 (~2,3 GB) más el
reranker `mmarco-mMiniLMv2-L12` (~0,5 GB) suman ~2,8 GB, y ahí no hay generador compitiendo
por la placa. Y las pasadas de embeddings son justamente la parte que más se beneficia de no
depender de una sesión de Colab que se desconecta: `store.cached_matrix` las cachea en disco
por spec del encoder, así que correrlas en una máquina que no se apaga nos deja reusarlas sin
límite.

### La decisión: reparto híbrido

- **Etapas 0-3 en la 3050 local.** Limpieza del corpus, conjunto de evaluación, índice y toda
  la medición de recuperación. No necesitan generador, y el embedder y el reranker entran sin
  pelear.
- **Etapas 4-6 en Colab T4**, con `qwen3:8b`, `think=False`, temperatura 0 y `num_ctx`
  explícito en cada llamada, tal como lo especifica §7.2. Con `USE_DRIVE = True` desde el
  primer día, no el último.
- **Esta notebook/laptop** (sin GPU NVIDIA, 8 GB de RAM) es solo para escribir código e iterar
  con `SMOKE_TEST = True` sobre CPU.

El costo de este reparto es un traspaso: `artifacts/full/index/` y `artifacts/full/embeddings/`
se construyen local y tienen que llegar a Colab. Los manifiestos de `store.py` hacen que ese
traspaso sea verificable — una etapa que carga un índice chequea de qué `documents.jsonl` salió
y con qué embedder, y aborta si no coincide, en vez de puntuar otra cosa en silencio.

**Lo que NO vamos a hacer, y por qué:** bajar el generador a `qwen3:4b-q4_K_M` entraría 100%
en la 3050 (~2,5 GB de pesos + 1,2 de KV = ~4,1 GB) y sería ~3x más rápido. Lo descartamos
porque §7.2 especifica `qwen3:8b` y porque cambiar el generador cambia todas las superficies
que el lab mide — la abstención, la fidelidad al pasaje, el seguimiento del schema. Preferimos
pagar la logística del híbrido antes que medir otro modelo. (Si la T4 se vuelve impracticable,
esta es la primera palanca a mover, y pasa a ser una entrada nueva de este registro.)

### Cuántas llamadas al LLM cuesta esto

La multiplicación que §2.1 pide hacer *antes* de correr: ítems × turnos × llamadas por turno ×
brazos.

| Etapa | Cuenta | Llamadas |
|---|---|---|
| 0 · chequeo a libro cerrado | 5 preguntas | 5 |
| 4 · respuestas fundamentadas | 30 ítems × 3 filas (libro cerrado / mejor config / oráculo) | 90 |
| 5A · reescritura de consulta | 10 guiones × ~4 turnos (salida corta, un solo campo) | 40 |
| 5B · ventana compartida | 10 guiones × 4 turnos × 3 divisiones, + ~40 resúmenes | 160 |
| 6 · memoria | escritura: 4 escenarios × 3 sesiones × 3 turnos ≈ 36; lectura: 24 sondas × 3 brazos = 72 | 108 |
| | **Total** | **~400** |

A 10-20 s por respuesta fundamentada en una T4 (§7.3), son **1,5 a 2,5 horas de generación**,
sin contar recargas del modelo. `llm.use_cache` hace que volver a correr una celda salga
gratis, así que el riesgo real no es el total sino repetir una corrida entera por haber movido
dos variables a la vez.

Dos cosas que presupuestamos aparte:
- **Recargas por `num_ctx`.** Cada cambio de `num_ctx` recarga el modelo (~11 s). Mantenemos
  `num_ctx = 8192` fijo dentro de cada experimento; un brazo que lo cambie convierte la columna
  de latencia en un conteo de recargas.
- **Pasadas de embeddings.** Una por cada variante de chunking o de prefijo. Esto es lo que
  acota la Etapa 3 más que las llamadas al LLM, y es la razón por la que la grilla está
  recortada (abajo).

### El plan: hasta dónde llevamos cada etapa

El §2.2 avisa que el recorte más discutible es "una grilla completa de la Etapa 3 y nada
después". Vamos por lo contrario.

| Etapa | Hasta dónde | Por qué |
|---|---|---|
| 0 · corpus | Completa | Elegir ruta de entrada es la decisión que la auditoría resuelta deja abierta a propósito |
| 1 · documentos | Completa (obligatoria) | Todo cuelga del registro de documento |
| 2 · vara de medir | Completa (obligatoria), congelada antes de cualquier resultado | Alimenta todo; una vara escrita después de ver resultados no mide nada |
| 3 · recuperación | **Acotada**: 2 líneas de base + 3 filas acumulativas + 2 ramas etiquetadas | Ver abajo |
| 4 · respuestas | Completa, con las 3 filas y la tabla 2×2 | Construye el ensamblador del que dependen 5 y 6 |
| 5 · conversación | **Completa, es donde ponemos el tiempo** | Es el corazón del lab (§2.2) |
| 6 · memoria | Completa | |
| 7 · juez y framework | **Recortada entera** | Ver abajo |

**Etapa 3, las filas que corremos.** Dos líneas de base (solo BM25 — que además es el detector
de contaminación de la Etapa 2 — y solo denso), y tres filas acumulativas, cada una agregando
exactamente una pieza:

1. denso **+ prefijo contextual** (título · sección · versión)
2. **+ léxico vía RRF** (`k = 60`, la constante del paper, no algo para ajustar con 30 preguntas)
3. **+ rerank**

`+ padre` y `+ filtro de metadatos` **no** van como filas acumulativas sino como **ramas
etiquetadas** a presupuesto de tokens igualado. Razón: ninguna de las dos es una mejora del
ranking, y meterlas en la escalera acumulativa confunde dos preguntas distintas. El padre
responde "con tokens igualados, ¿muchos hijos o pocos padres?", que es una pregunta sobre
*entrega*, no sobre recuperación — y además, como avisa la trampa de §3, el hit@k a nivel de
documento no puede ver su efecto, porque el padre siempre está en el mismo documento que el
hijo. El filtro de metadatos solo tiene sentido sobre la categoría de versión/temporales, y es
la pieza que la Etapa 6 va a alimentar desde la memoria; medirlo sobre las 30 preguntas lo
diluiría.

Cada fila se entrega **por presupuesto de tokens del generador, no por k**, y reportamos los
tokens realmente entregados por fila. Con ~30 ítems, un intervalo del 95% sobre una sola fila
cerca del 85% abarca ±12 puntos (§3): no afirmamos nada que no sobreviva a una comparación
pareada con test de McNemar.

**Etapa 7, por qué la recortamos.** El juez (7a) y la reimplementación en un framework (7b)
son segundas lecturas de lo que construimos. Las dos son valiosas, y las dos cuestan una
pasada completa de generación extra sobre las respuestas guardadas. Con ~400 llamadas ya
presupuestadas y el reparto híbrido metiendo un traspaso entre máquinas, preferimos que la
Etapa 5 esté bien hecha a que la 7 esté empezada. Si sobra tiempo después de que la Etapa 6
cierre, la pieza que más nos interesa es **7a**, porque es la única que nos deja puntuar las
respuestas abiertas que la contención no alcanza a juzgar — y porque auditar a mano 30 juicios
es barato comparado con volver a correr un loop entero. Esto queda como una entrada pendiente,
no como una promesa.

### Sobre el uso de IA en este trabajo

Usamos un asistente (Claude) para escribir e iterar el código de las etapas y para discutir
el diseño. La **regla específica del lab (§2.3)** nos obliga a ser precisos sobre un punto:
los ítems de evaluación, los guiones de conversación y las sondas de memoria de la Etapa 2.
El criterio que fijamos **ahora, antes de escribirlos**:

- Ningún ítem se escribe con el pasaje gold abierto (esa es la trampa de "las preguntas
  escritas leyendo el pasaje miden copia"). Se escriben a partir de una lista de *temas*.
- Un ítem redactado por un modelo es **una propuesta**, no un ítem, hasta que una persona
  verificó la respuesta contra el corpus. En el informe decimos cuáles fueron y cómo se
  verificaron.
- El chequeo de contaminación con BM25 corre sobre *todos* los ítems, sin importar quién los
  redactó, y los marcados se reescriben.

---

## 2026-10-24 · Entrada 2 — La ruta de entrada: HTML compilado, no `.po`

`corpora.md` ofrece dos rutas al mismo corpus y la auditoría resuelta recorre las dos.
Elegimos el **HTML compilado** (`python-{v}-docs-html.zip`).

Lo que decidió: las tres cosas que a `.po` le faltan **no las recupera ninguna etapa
posterior**, y las tres las necesita nuestro plan.

1. **No hay id de documento ni de sección.** Un entry de `.po` se identifica por (archivo,
   línea del `.rst`). La línea se mueve con cada commit de upstream, y además no es única:
   la auditoría resuelta mide **2.209 entries compartidos** por más de una ubicación porque
   gettext fusiona texto idéntico. Sin un id estable no hay gold de sección (Etapa 2) ni
   citas verificables (Etapa 4).
2. **No hay firmas de API.** Medido: `pathlib.po` no contiene `resolve(strict=...)`.
   Las firmas viven en el `.rst` inglés; juntarlas de vuelta significa un join por commit.
3. **No hay números de versión.** Medido: `pathlib.po` no contiene "3.12". Sin eso no hay
   preguntas temporales, que son la mitad del conjunto de evaluación que planeamos.

Lo que cuesta la ruta HTML, con nombre y con etapa asignada:

- **Se pierde la etiqueta `fuzzy`** — la traducción que el proyecto mismo marca como dudosa.
  Pero el *síntoma* sobrevive: una entry sin traducir sale como párrafo en inglés, y eso sí
  lo medimos (`corpus_html.paragraph_language`). Lo paga la **Etapa 3**: 251 de 385 páginas
  tienen más de 10 % de párrafos en inglés, así que la consulta en español contra el pasaje
  en inglés es recuperación cross-lingual, no un caso raro.
- **Se pierde el pin por commit de git.** El zip se recompila en el mismo lugar: de los dos
  sha256 que la auditoría resuelta fijó el 2026-09-27, hoy **no coincide ninguno**. Lo paga
  la **Etapa 1**: pin por sha256 propio + fecha de descarga, y `on_mismatch="raise"` de ahí
  en adelante.

Verificado aparte, porque la auditoría resuelta dice explícitamente que no lo mide: bajamos
3.13 **dos veces** y comparamos. Mismos bytes, mismos ids de página, sección y ancla. El id
es estable *dentro* de una compilación; contra 3.12 se mueven 1.678 anclas (1.211 solo en
3.13, 467 solo en 3.12, 9.548 compartidas).

---

## 2026-10-24 · Entrada 3 — La regla del subconjunto, después de medirla

Entra en `00b_corpus_audit_ours.py`. **El primer borrador era una sola regla y medirlo lo
rompió**, así que la entrada registra las dos versiones.

**Borrador 1: "descartar si inglés > 60 %, descartar si < 100 tokens."** Falla en los dos
sentidos a la vez:

- Descartaba `library/aifc.html` (100 tokens, 100 % inglés) como demasiado corta. Pero
  `aifc` es un **aviso de eliminación**: su contenido es "este módulo se quitó en 3.13".
  Ese texto *es* la respuesta a una pregunta temporal y el mejor material de abstención que
  tiene el corpus. Lo estábamos tirando.
- Rescatarlos con "exceptuar si tiene nota de deprecación" era peor: `builtins/functions.html`
  (17.972 tokens, 98 % inglés) tiene *alguna* nota de deprecación entre sus funciones, así
  que la excepción lo rescataba.

**Versión final: un clasificador de tres clases, y el orden importa.**

```
1. nota de deprecación Y tokens < 500  -> "aviso de eliminación"  (se queda)
2. tokens < 100                        -> "puntero"               (se descarta)
3. resto                               -> "documentación"         (se queda si inglés <= 60 %)
```

La prueba de eliminación va **primera y acotada por largo**; así `aifc` cae en la clase 1 y
`functions.html` no, porque 17.972 > 500.

| clase | páginas | tokens | anclas | efecto |
|---|---:|---:|---:|---|
| 1 · aviso de eliminación | 24 | 2.514 | 0 | se queda, `status=deprecated` |
| 2 · puntero | 7 | 442 | 0 | se descarta |
| 3 · documentación en español | 311 | 1.684.368 | 9.314 | se queda |
| 3 · documentación en inglés | 43 | 197.674 | 1.445 | se descarta |

Corpus final: **335 de 385 documentos, 1.686.882 tokens, 9.314 de 10.759 anclas (86,6 %)**.
La regla cuesta 13 % de las anclas, así que el material de las preguntas de identificador
sobrevive.

**El costo con nombre propio:** `builtins/` entero está sin traducir y la regla lo descarta
(`stdtypes` 99 %, `exceptions` 97 %, `functions` 98 %). Ninguna pregunta nuestra puede ser
sobre `print` o `len`. Es el techo del corpus, no un defecto de la regla, y va dicho así en
`corpus_card.md` y en el informe.

La clase 2 tampoco es cosmética: `library/concurrent.html` son 31 tokens y
`howto/clinic.html` 36. Son páginas índice sin contenido; indexarlas mete documentos que
pueden ganar por similitud de título sin tener nada que citar.

---

## 2026-10-24 · Entrada 4 — Qué es un documento, y el agujero de `builtins`

**Un documento es una página compilada de una versión**: `py3.13:library/pathlib`. No el módulo
`pathlib` (eso uniría 3.12 y 3.13 en un registro con dos respuestas contradictorias), ni una
sección (eso tiraría el título de página, que es lo que hace legible un pasaje suelto). Una
**sección** es un `<section>` de Sphinx con el `id` que la fuente le puso. Resultado:
**695 documentos, 5.614 secciones, 3.474.559 tokens**,
`corpus_sha256 = 09c6862a23ce6d6f18ab31c16814c36ac537e5bc68973b8f5798a03dc0f91193`.
<!-- Ese hash y el conteo de firmas de abajo son los del día: el corpus se volvió a construir dos
     veces después. El linaje completo está en la Entrada 5. El vigente es 6c58d2cf300c1ff6…. -->


**La corrección importante: la Etapa 0 se equivocó sobre `builtins`.** Había cerrado la regla del
subconjunto diciendo "ninguna pregunta nuestra puede ser sobre `print` o `len`", porque
`builtins/` de 3.13 está sin traducir. Aplicar la misma regla a **las dos** versiones mostró que
3.13 movió esas páginas a `builtins/` dejando un stub en `library/`, y que **3.12 todavía las
tiene en `library/`, traducidas al 82-90 %**: se recuperan 398 anclas (`functions` 67,
`stdtypes` 216, `exceptions` 104, `constants` 11).

Lo que cambia de los planes, que es por lo que esto es una entrada y no una nota al pie:

- esas preguntas tienen gold de **una sola** página, y es `status="superseded"`. No pueden tener
  conjunto gold de dos versiones;
- **confirma el filtro por versión como rama etiquetada, no como escalón acumulativo** (Entrada 1
  lo había decidido por otro motivo: que `version` no discrimina dentro de una versión). Ahora hay
  un caso concreto donde restringir a 3.13 sube la precisión y *pierde la respuesta*;
- da un gancho de memoria mejor para la Etapa 6: "estoy en 3.13" + una pregunta de builtins es un
  conflicto real entre lo que el usuario declaró y lo que el corpus puede probar.

**La limpieza: cinco reglas, dos con cero aciertos, y eso también es un dato.** `¶` → salto de
línea (5.942 bloques), firma partida por `anchor_id` (1.570), etiqueta de nota al español (1.095,
y *agrega* 10.314 caracteres), NFC (4 bloques), espacios (0, porque `corpus_html` ya los
colapsa). Las dos de cero se dejan igual: el día que cambie el extractor, el contador lo dice.

**La decisión de no arreglar algo:** los ejemplos de código quedan con los espacios de los
`<span>` de resaltado (`p . iterdir ()`). Arreglarlo requiere volver al HTML y tratar `<pre>`
aparte; el arreglo por `anchor_id` no llega porque los ejemplos usan nombres locales. Cuesta que
una consulta léxica por `p.iterdir()` no matchee el ejemplo; la firma canónica sí está limpia y es
donde vive la respuesta. Queda anotado como primer sospechoso si las preguntas de identificador
fallan en la Etapa 3.

**Dos bugs nuestros que la verificación encontró** (van acá porque los dos habrían pasado sin
error): la prueba de "metadatos que mienten" daba 18 % de desacuerdo comparando versiones
completas, así que marcaba `os.path` de 3.13 por citar "3.13.4" —un parche de su propia rama—;
corregida a comparar solo (mayor, menor), da **0 %**. Y los offsets de sección: la limpieza corre
**por bloque** con un mapeo exacto de posiciones viejas a nuevas, más un assert de que cada
`start`/`end` cae en un borde de bloque. Sin eso, limpiar el texto desplaza todas las citas en
silencio.

---

## 2026-10-24 · Entrada 5 — El conjunto congelado, y el corpus que cambió por escribirlo

**Qué se congeló:** 53 ítems de un solo turno (20 temáticos, 18 de identificador, 10 de
metadatos, 5 fuera del corpus), 9 diálogos de 4 turnos y 4 escenarios de memoria con 11 sesiones
y 25 sondas, en `eval/` con `FROZEN.json` apuntando a
`documents_sha256 = 6c58d2cf300c1ff61649139369495ff70ad8766b15acabd8af74a0d5228aa7c6`. División
dev/test 8/45: la dev existe para mirar y romperse, la test no se mira hasta la Etapa 3.
**Se congeló antes de que exista cualquier resultado**, que es lo único que hace que la división
signifique algo.

**La decisión central: cómo se escribe cada pregunta se declara por ítem, no se promedia.** Hay
dos procedencias y miden cosas distintas:

- **derivadas** (36 ítems): identificador desde anclas únicas en todo el corpus y desde la firma
  que la página muestra; metadatos desde las notas de versión, atribuidas a la definición
  inmediatamente anterior. Son automáticas y verificadas por contención, así que no pueden copiar
  la redacción del pasaje, pero tampoco pueden preguntar nada interesante;
- **a mano** (25 ítems): las temáticas, escritas **desde el título de sección, con el pasaje
  cerrado**, y las 5 de fuera del corpus, escritas desde la lista de novedades de 3.14.

El enunciado avisa que una pregunta escrita con el pasaje abierto mide copiado, no recuperación.
Acá eso no se promete, se mide: contaminación por categoría con **el mismo tokenizador del canal
léxico** (`lexical.py`, un solo stemmer para el chequeo y para BM25, o el chequeo mide un
solapamiento distinto del que infla a BM25). El n-grama compartido más largo es **3 tokens** en
temáticas y 2 en las derivadas, con el umbral en 5: ninguna pregunta importó la redacción.

**La prueba que podía salir mal: BM25 solo resuelve 29 de 48 (60 %) a nivel sección.** Por
categoría: temáticas 35 %, identificador 72 %, metadatos 90 %. Si hubiera dado 95 % el conjunto
no tendría margen para que la Etapa 3 mueva nada y había que reescribirlo. Los 15 ítems que BM25
no encuentra quedan listados: son la lista contra la que se va a leer cada escalón.

**Las trampas del enunciado, resueltas una por una:**

- **una clave de respuesta no puede decir que no.** 15 renglones llevan `expected_verdict`
  (`yes`/`no`), que se lee por la primera palabra y sobre el texto crudo. La abstención **no**
  cuenta como `no`: son resultados distintos y se cuentan aparte;
- **una actualización solo se ve como par.** `p-batched-antes` / `p-batched-despues` son la misma
  pregunta antes y después de que la sesión diga que el proyecto pasó a 3.13: veredictos
  opuestos, los dos correctos. Igual en los otros tres escenarios;
- **la clave es UNA palabra, y es a propósito.** El campo dice *«short strings a correct answer
  contains»*: se leen en conjunción. Agregar un sinónimo no afloja, endurece — pedir `estable`
  *y* `estabilidad` falla a «el orden es estable», que es correcta. Probé las 5 alternativas que
  sospechaba (`orden`, `una sola vez`, `normalizar`, `estabilidad`, `fracciones binarias`) contra
  el pasaje y las informo, **sin tocar la clave**: 3 también están, 2 no. Correcciones: 0 de 20,
  porque la contención no falló ninguna vez;
- **cinco términos de 3.14 verificados ausentes con conteo literal** (`Path.copy`,
  `compression.zstd`, `template string literal`, `InterpreterPoolExecutor`, `annotationlib`): 0
  ocurrencias cada uno, y el script levanta `RuntimeError` si alguno aparece. `tomllib`,
  `ExceptionGroup`, `asyncio.TaskGroup` y `itertools.batched` **sí** están en el corpus y por eso
  no sirven como no-respondibles, aunque "suenen" nuevos.

**El gold es un conjunto, y 48 de 48 ítems con respuesta lo tienen de las dos versiones.** La
métrica por documento que cuente la otra versión como fallo no mide nada (el MinHash de la
Etapa 0 ya lo decía: las gemelas comparten de 0.35 a 1.00 de 5-gramas). Donde el gold colapsa a
una sola versión queda **registrado, nunca implícito**: los dos únicos casos son el par de
`itertools.batched`, porque `library/itertools.html` existe solo en 3.12.

**Lo que esta etapa le hizo a la anterior, que es el hallazgo que no esperaba.** Derivar las
preguntas de "valor por omisión" obliga a parsear la firma tal como la página la muestra, y ahí
se vio que la regla de firmas partidas de la Etapa 1 cubría solo la forma **calificada**
(`pathlib.Path. open`) y no la **corta** (`Path. open`), que es la que las páginas muestran
casi siempre: quedaban **3.364 ocurrencias sin tocar**, más del doble de las que la regla había
arreglado. Se extendió la regla a **todos los sufijos** del `anchor_id` (1.570 → 1.886 bloques),
y el contador de restos bajó a 4 — todas en `py3.12:library/socket`, firmas con **dos** puntos
partidos (`socket. socket. type`), que `re.sub` no puede arreglar en una pasada porque sigue
leyendo después del reemplazo. Se iteró a **punto fijo** y el contador dio **0**.

Dos cosas que quedan dichas de esto:

1. **el corpus se reconstruyó dos veces y el hash cambió dos veces**
   (`09c6862a…` → `ca94ffe0…` → `6c58d2cf…`). Se congeló contra el último, y
   `corpus_card.md` dice el mismo. Si vuelve a cambiar, `freeze_eval` lo va a gritar;
2. **el riesgo de falso positivo se midió, no se supuso.** 202 ocurrencias tienen la cola en
   mayúscula (`TLSVersion. TLSv1`, `blake2b. SALT_SIZE`); inspeccioné las 20 más frecuentes y
   todas eran firmas, así que la regla queda sin guarda de palabras prohibidas.

**Lo que el conjunto NO puede medir, dicho ahora y no en el informe:** si el asistente le sirve a
alguien. 53 ítems con clave de respuesta miden si la respuesta contiene la palabra correcta, no
si el párrafo alrededor es útil, ni si la cita es la que un humano habría elegido entre dos
pasajes igual de buenos. Los 25 ítems escritos a mano los escribí yo, que ya decidí qué me
parece una buena pregunta de la documentación de Python.

**El candado se probó rompiéndolo**: agregar un renglón de más a `single_turn.jsonl` hace que
`load_eval` levante `EvalChanged` ("53 → 54 rows"), y después se restaura. Un congelado que no
muerde es un archivo.

**Sin perfil smoke para esta etapa, a propósito.** El conjunto se congela contra el corpus
completo; un conjunto distinto por perfil son dos conjuntos. `eval/smoke/` existe aparte —4 ítems
cuyo gold cae en las diez páginas del smoke— **sin** `FROZEN.json`, solo para que el humo corra
rápido.

**Pendiente de una persona** (no lo puede hacer el script): leer diez de las 40 atribuciones
nota→definición en `artifacts/full/stage2_note_owners.txt` y confirmar que la nota habla de la
función de arriba. Es el único supuesto de esta etapa que no está verificado por contención.

---

## 2026-10-24 · Entrada 6 — La memoria entre sesiones: declarar el régimen antes del resultado

Entrada de **diseño**, escrita antes de correr: `06_memory.py` está escrito y verificado en todo
lo que no necesita GPU (32 controles sobre las partes puras: vigencia, vecinos, presupuesto de
lectura, filtro de versión, solapamiento con la fuente). Los números de abajo son **medidas del
conjunto congelado**, no resultados del experimento.

**La decisión incómoda: a nuestra escala la reproducción entra siempre.** Los 3 escenarios de test
tienen 2-3 sesiones de 1-2 turnos: 56 tokens por sesión en promedio, y el estado más largo
(`s-etl` tras `s3`) son 206 tokens de transcripción. Contra `num_ctx = 8192`, o contra el espacio
de memoria de 300-1.000 tokens que sugiere §7.3, no hay nada que no entre. Decirlo es la mitad de
la respuesta; la otra mitad son dos cosas que sí se pueden medir:

1. **la proyección**, con los tokens por sesión que estos escenarios tienen de verdad: cuántas
   sesiones hacen falta para cruzar cada límite;
2. **dos regímenes del mismo espacio**: `SLOT_WIDE = 400` (entra entera) y `SLOT_TIGHT = 120`
   (no entra entera en 15 de las 19 sondas).

**`SLOT_TIGHT = 120` salió de una medición, no de un número redondo.** El espacio de memoria cobra
**32 tokens de encabezado** (el texto que lo presenta más la fecha simulada) y 16 por cada fecha de
sesión: comparar la transcripción pelada contra el límite declara «entra» cuando no entra, así que
la tabla del régimen cuenta el encabezado (`replay_tokens`). Con eso: 400 → 0 sondas recortadas,
120 → 15, 80 → 19. Elegí 120 y no 80 porque 120 − 32 − 16 ≈ 72 tokens deja entrar **una sesión**
(que mide 56) y a 80 el corte cae en mitad de un turno: eso mide dónde cayó la tijera, no qué
sesiones conservó la política, que es lo que el brazo quiere evaluar. La unidad que la reproducción
tira es la sesión, así que el espacio tiene que poder contener al menos una. Queda dicho que
`SLOT_TIGHT` **no es un despliegue realista**: es el valor que hace binaria la pregunta «¿entra?»
con tres sesiones, y la tabla imprime a cuántas sesiones equivale en un espacio de 400.

**Los dos brazos comparten el mismo espacio** (el parámetro `memory=` de `pr.messages`, que aterriza
en el mensaje de sistema). Poner la reproducción como mensajes `user`/`assistant` de verdad y el
almacén como un bloque de texto confundiría representación, posición y marcadores de rol en una
sola columna; la Etapa 5 ya midió mensajes-contra-bloque para el historial intra-sesión.

**La escritura no guarda el razonamiento.** Los dos schemas llevan `analysis` primero (y por eso
también primero en orden alfabético: `analysis` < `decision` < `target_id` < `text`), pero el
registro se arma **campo por campo** y `analysis` no se copia nunca: es la trampa de guardar una
hipótesis como hecho. Se cuenta cuántos registros traen marcas de pensamiento, cuántos traen
coberturas («probablemente», «supongo»), y cuál es el solapamiento de palabras de contenido con la
sesión de origen, con el mismo tokenizador del canal léxico.

**Se invalida, no se borra.** UPDATE le pone `valid_to` al registro viejo y escribe uno nuevo con
`supersedes`. La vista «al momento de la sonda» sale de comparar los timestamps simulados con la
fecha de la sonda, sin una segunda copia del almacén. Un `target_id` que el modelo inventa no se
acepta: se **cuenta y se degrada** (UPDATE → ADD, INVALIDATE → NOOP).

**El hallazgo que no esperaba, y que es de la fórmula del enunciado:** con `0.995/hora` y meses
entre sesiones, la recencia del registro más nuevo vale ~1 y la del segundo ~2e-4. Después del
min-max eso **no es un orden por antigüedad, es un one-hot del último registro**: el puntaje se
reporta como «relevancia + importancia + un bonus para el más nuevo» y no como tres términos. Los
rangos de los tres términos se imprimen **antes** de normalizar, que es el chequeo que el enunciado
pide, y el min-max solo colapsa a ceros cuando el término es exactamente constante.

**La memoria como filtro del recuperador, y el daño que hace.** El usuario de `s-etl` arranca en
Python **3.11** y el corpus solo tiene 3.12 y 3.13: `allow_for("3.11")` devuelve el conjunto
**vacío** y `retrieve` entrega **cero** pasajes. Eso es a propósito y es la señal —un filtro de
memoria que deja cero tiene que dejar cero— y lo que elimina es justamente el pasaje de
`itertools` de 3.12 con el que el asistente podría decir «eso no existe en tu versión». Es la
política **opuesta** a la de `cfg.metadata_filter` de la Etapa 3, que cae al listado sin filtrar
(`fused = kept or fused`), y las dos se justifican: una versión que el **usuario tipeó** puede ser
un error de tecleo; una versión que el **sistema guardó** es una afirmación propia sobre su
usuario, y taparla es mentir sobre lo que la memoria dice. `allow=None` (no hay hecho guardado) y
`allow=set()` (hay un hecho y no hay corpus) son dos casos distintos y el código no los mezcla.

**El costo se informa como punto de equilibrio, no como «es más barato».** La reproducción no paga
ninguna llamada de escritura; el almacén paga extracción + decisión una vez por sesión y ahorra
tokens de prompt en cada sonda. El número que va al informe es
`tokens_de_escritura / tokens_ahorrados_por_sonda`: cuántas sondas hay que hacer para amortizar.

**Lo que este conjunto no puede sostener:** 19 sondas que salen de 3 escenarios no son 19
observaciones independientes, así que no se publica intervalo por sonda. Las comparaciones van
pareadas y con el piso de ruido declarado: hacen falta **6** discordantes todos para el mismo lado
para bajar de 0,05, y la exactitud por habilidad se informa en **cuentas** (3 de 4), nunca en
porcentaje.

**Pendiente de una persona:** auditar los registros del almacén (`stage6_records.txt` →
`stage6_record_audit.json`) y contar los falsos, los que el usuario nunca dijo y los que son
hipótesis. Ningún script puede decidir si un recuerdo es cierto.

---

## 2026-10-25 · Entrada 7 — Todo en la 3050: se cae el reparto híbrido

**Qué cambia.** La Entrada 1 repartía el trabajo: Etapas 0-3 en la 3050 local, Etapas 4-6 en una T4
de Colab, porque 6 GB de VRAM no alcanzan para `qwen3:8b` con `num_ctx` grande. Decidimos correr
**todas las etapas en la 3050**. Esta entrada reemplaza ese reparto; el resto de la Entrada 1 (el
presupuesto de llamadas, los modelos, el `num_ctx` fijo) sigue en pie.

**Por qué el reparto no se sostenía.** La T4 agrega una máquina más que montar, un repositorio
privado que clonar con un token, y artefactos que viajan entre dos sistemas de archivos. Cada
etapa que cruza la frontera es una oportunidad de que el `documents_sha256` o el `index_key` dejen
de coincidir, y el chequeo que lo detecta corre *después* de haber gastado las llamadas.

**La cuenta que hay que mirar de frente.** Con los números del enunciado (§7.2 y su tabla de caché
KV), `qwen3:8b` Q4_K_M son ~5,2 GB de pesos y la caché KV a `num_ctx = 8192` son ~1,2 GB: **~6,4
GB contra 6 de la placa**, y eso *antes* de `bge-m3` en fp16 y el cross-encoder, que en nuestra
estimación suman ~1,9 GB más con el contexto de CUDA. No entra. `llm.placement()` lo va a reportar
como `SPLIT … expect several times slower`, que es exactamente el modo de falla que el enunciado
advierte: **mismas respuestas, varias veces más lento, sin error**. A ~480 llamadas, el
presupuesto de 1,5-2,5 horas de la Entrada 1 pasa a 5-12 horas.

**Las palancas, en orden, y por qué en ese orden.**

1. **Medir antes de tocar.** `llm.placement(MODEL)` después de `llm.setup`. Si dice `100% GPU`, no
   hay nada que decidir. Esa línea va al informe en los dos casos: una corrida partida es un dato
   sobre el experimento, no un detalle de infraestructura.
2. **`OLLAMA_KV_CACHE_TYPE=q8_0` con `OLLAMA_FLASH_ATTENTION=1`.** La caché KV baja de ~1,2 a ~0,6
   GB. Cambia la aritmética de la atención, así que si se activa se activa para *todas* las etapas
   que se comparan entre sí, nunca a mitad de un barrido.
3. **El reranker a CPU** (~0,5 GB de los nuestros). El cross-encoder no tiene nada cacheado, así que
   moverlo no cuesta más que rerankear más lento, y contra 10-20 s de generación no se nota.
4. **`qwen3:4b-q4_K_M`** (~2,5 GB de pesos + 0,6 de KV): entra entera con todo cargado y es ~3x más
   rápida. La Entrada 1 ya la nombraba como primera palanca; acá queda como la **última**, porque
   bajar el generador cambia las respuestas que estamos midiendo, mientras que 2 y 3 solo cambian
   dónde corren los números.

**Una palanca que parecía gratis y no lo es: el embebedor a CPU.** `environment.device()` sale de
`torch.cuda.is_available()`, así que con `CUDA_VISIBLE_DEVICES=""` torch deja la placa entera para
Ollama —Ollama es otro proceso— y se liberan ~1,4 GB. Pero `retrieval.load` arma el
`encoder_spec` con `"dtype": "fp16" if device == "cuda" else "fp32"`, y `store.cached_matrix` hashea
el spec para formar su clave: **en CPU la clave cambia y las dos matrices de hijos se re-encodean de
cero con `bge-m3`**, que es más caro que todo lo que la palanca ahorra. Descartada, y anotada acá
para no volver a proponerla. (El spec tiene razón en registrar el `dtype`: la precisión del cómputo
es parte de qué produjo esa matriz.)

**Lo que esto implica, sin adornos.** Con 2 y 3 aplicadas, Ollama necesita ~5,8 GB y nosotros
seguimos reteniendo ~1,4 GB de `bge-m3` más el contexto de CUDA: en 6 GB no cierra. Es decir que en
esta placa, con la pila de recuperación residente, **`qwen3:8b` no puede quedar entera en VRAM**. La
decisión real es binaria: aceptar el `SPLIT` (5-12 h, con la caché de `llm` protegiendo los
reintentos) o bajar a `qwen3:4b`. Las palancas 2 y 3 solo achican cuánto se ofloadea. Elegimos
**aceptar el split y reportarlo**: la comparación entre brazos es interna a cada etapa y el split
afecta a todos los brazos igual, así que cuesta horas, no validez.

**Lo que decidimos NO mover: `num_ctx`.** Bajarlo a 4.096 haría entrar el modelo, pero la Etapa 4
entrega `BUDGET_TOKENS = 3000` más 700 de reserva de salida: `pr.passage_budget` recortaría la
entrega **en silencio**, y las 3 filas de la Etapa 4 medirían un presupuesto de pasajes distinto
del que midió la Etapa 3. El problema no es la ventana chica, es que el recorte no avisa.
`num_ctx = 8192` queda fijo en las Etapas 4, 5 y 6.

**Qué más se cae con esto.** El arranque para Colab de `tools/py_to_notebook.py` (clonar el
repositorio privado con `GITHUB_TOKEN`) deja de estar en el camino crítico. Queda en el código
porque las notebooks tienen que poder abrirse en otra máquina, pero no es parte de nuestra corrida.

---

## Entradas siguientes

<!-- Formato: ## FECHA · Entrada N — Título breve
     Qué decidimos, contra qué alternativa, y qué número o qué hallazgo nos llevó ahí.
     Cualquier edición del conjunto de evaluación congelado es una entrada obligatoria. -->
