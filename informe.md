# Retrieval & Memory Lab — Informe

**Corpus:** `python-docs-es` (documentación de Python en español), versiones 3.13 y 3.12, por la
ruta del **HTML compilado**.
**Generador:** `qwen3:8b` en Ollama, `num_ctx = 8192` fijo en todas las etapas.
**Pregunta del proyecto:** cómo se reparte una ventana finita entre lo que dice el corpus y lo que
se dijo en la conversación.
**Etapas hechas:** 0, 1, 2, 3 (5 escalones y 2 ramas), 4, 5, 6 y 7a (juez). Sin Etapa 7b.
**Perfil de los números de este informe:** `full` salvo donde se diga `smoke`.

> **Estado de este documento.** Todas las etapas corrieron en perfil `full` en la RTX 3050 el
> 2026-10-31 (`tools/run_stages.py`, 4,7 h; `qwen3:8b` en SPLIT 30 % CPU / 70 % GPU durante las
> Etapas 4-6). Los números de §4-7b salen de `artifacts/full/run_report.json` y de los
> `stage*_rows.json`; las lecturas al lado de cada tabla son nuestras. Las lecturas «a mano» están
> hechas **por Claude, no por un integrante** (§8), y dejan sus cuentas en `artifacts/full/*audit*.json`.
> Lo único PENDIENTE es la línea de §8 sobre qué verificó cada integrante.

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

El detalle, con fechas y alternativas descartadas, está en `decision_log.md` (9 entradas). Acá el
resumen de las que condicionaron todo lo demás:

| Decisión | Alternativa descartada | Qué nos llevó ahí |
|---|---|---|
| **Corpus `python-docs-es` por HTML compilado** | los `.po` del repo de traducción | el `.po` es la unidad de traducción, no la de lectura: entradas sueltas, `fuzzy`, obsoletas. El HTML es lo que un usuario lee (Entrada 2) |
| **Dos versiones (3.13 y 3.12), no una** | una sola versión | el corpus necesita documentos confundibles para que la recuperación tenga trabajo; las páginas gemelas comparten de 0,35 a 1,00 de 5-gramas (MinHash, Etapa 0) |
| **Todo en una RTX 3050 de 6 GB** | el reparto híbrido con una T4 de Colab (plan original, Entrada 1) | cada etapa que cruza la frontera entre dos máquinas puede desalinear el `documents_sha256` o el `index_key`, y el chequeo que lo detecta corre *después* de gastar las llamadas. El costo: `qwen3:8b` no entra en 6 GB (~5,2 de pesos + ~1,2 de caché KV a 8.192), así que la corrida se reporta con lo que diga `llm.placement()` (Entrada 7) |
| **Etapa 3 como escalera: 5 escalones y 2 ramas** | el barrido completo de combinaciones de piezas | cada escalón agrega **una** pieza y se compara pareado contra el anterior, así que cada fila responde una pregunta. El plan de la Entrada 1 eran 3 filas; la Etapa 3 no llama al LLM (948 s en total), así que la escalera completa no le quitó presupuesto a la Etapa 6 |
| **Etapa 7a sí, 7b no** | el framework (7b) | la 7a relee respuestas guardadas y no genera ninguna nueva: con todo en una placa es barata. La 7b es otra corrida completa de las Etapas 5-6 (Entrada 8) |
| **`num_ctx = 8192` en TODAS las etapas** | ajustarlo por etapa | una fila que cambia la ventana no mide su estrategia de contexto, mide la ventana |
| **El conjunto de evaluación congelado antes de cualquier resultado** | congelarlo al final | es lo único que hace que la división dev/test signifique algo (Entrada 5) |

**Lo que el repositorio hace y lo que es nuestro.** El andamiaje (`llm.py`, `store.py`,
`console.py`, `environment.py`, `schema.py`) viene del repo base. Nuestro: `00b_corpus_audit_ours`,
`01_documents`, `02_eval_set`, `03_retrieval`, `04_generation`, `05_dialogue`, `06_memory`,
`retrieval.py`, `prompting.py`, `lexical.py`, `noise.py`, `eval/`, `corpus_card.md` y
`decision_log.md`.

### Las preguntas de la Etapa 0

- **De las propiedades, cuáles no cumple el corpus y dónde se paga** (`00b_corpus_audit_ours`,
  tabla final; `artifacts/full/stage0_ours.json`):

  | Propiedad | Medido | Qué etapa paga |
  |---|---|---|
  | ¿está la respuesta? | 385 páginas, 1.885.104 tokens; **251 páginas con >10 % de párrafos en inglés** | Etapa 3: consulta en español contra pasaje en inglés |
  | ¿está limpio? | **13.757 marcas `¶`**, 4 stubs de redirección, etiquetas de versión en dos idiomas | Etapa 1: cada regla de limpieza, con el antes/después medido |
  | ¿hay estructura para cortar? | 2.996 secciones (todas con id), 10.759 anclas de API, 0 páginas sin sección; 196 hojas pasan el tope | Etapa 3: las hojas grandes necesitan corte por largo |
  | ¿el id sobrevive? | idéntico en dos descargas seguidas; contra 3.12 se mueven 1.678 anclas | Etapa 2: el gold se congela contra nuestro sha256, no contra la URL |
  | ¿hay metadatos? | versión 100 % (por construcción), **fecha 0 %**, nota de versión 66 % | Etapa 6: sin fechas del documento, los timestamps son simulados |
  | ¿es estable? | **no**: el zip se recompila en el mismo lugar (y volvió a pasar el 2026-10-31, Entrada 8) | Etapa 1: pin por sha256 propio y `on_mismatch="raise"` |
  | ¿se puede evaluar? | 10.759 anclas y 4.385 notas de versión: las preguntas de identificador y temporales salen solas | Etapa 2: salen *demasiado* solas — fuga léxica, chequeo BM25 obligatorio |
  | la similitud | Jaccard mediano 0,86 entre una página de 3.13 y su copia en 3.12 | Etapa 3: el hit@k por documento no ve el efecto; gold con las dos versiones |

  Lo que **no** cumple: limpio (el `¶` y la mezcla de idiomas), fechas (no hay), estable (se
  recompila) y la similitud, que es un problema buscado: es la que da trabajo a la recuperación.
- **El chequeo a libro cerrado de la Etapa 0** (5 preguntas a `qwen3:8b` sin pasajes, leídas contra
  las anclas medidas): **1 correcta** (`itertools.batched` desde 3.12), **1 vaga** (`aifc`: dice
  bien que no se importa en 3.13, pero dice que lo eliminaron «después de 3.10»; lo eliminó 3.13) y
  **3 incorrectas**: `PurePath.full_match` «en 3.11» (es de 3.13), `Path.copy` «sí existe en 3.13»
  (no existe en estas versiones) y el parámetro de `Path.resolve` que falla si la ruta no existe
  (es `strict=True`; el modelo inventa otra cosa). El modelo **cree** saber las versiones y se
  equivoca con seguridad: es la razón para medir recuperación contra libro cerrado. *Lectura hecha
  por Claude (ver §8).*
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

- **695 documentos**, `documents_sha256 = 891cf14c62fc496a1809e557bdc55e12059fd8892ed3a2b3a6d78c900822cd38` (re-pin del 2026-10-31, Entrada 8; antes `6c58d2cf…`).
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

**Corrió en `full`** el 2026-10-31 (948 s, 0 llamadas al LLM). **Fuente:** `artifacts/full/stage3_rows.json`,
`logs/03_retrieval.log`. 41 ítems de test con gold; 17.287 hijos y 6.826 padres en el índice;
`budget_tokens = 3000` en todas las filas.

| Fila | carácter | hit doc | hit sección | respuesta entregada | MRR@20 | tokens entregados | unidades | s/consulta |
|---|---|---|---|---|---|---|---|---|
| solo BM25 | base | 76 % | 59 % | 73 % | 0,558 | 2.947 | 11,4 | 0,00 |
| solo denso | base | 78 % | 73 % | 83 % | 0,612 | 2.970 | 11,6 | 0,04 |
| + prefijo | escalón | 76 % | 71 % | 85 % | 0,678 | 2.953 | 11,4 | 0,03 |
| + léxico (RRF) | escalón | 80 % | 76 % | 83 % | 0,640 | 2.968 | 11,2 | 0,03 |
| **+ rerank** (la mejor) | escalón | 78 % | 76 % | 83 % | 0,640 | 2.959 | 10,6 | 0,61 |
| rama: + padre | rama | 76 % | 68 % | 85 % | 0,640 | 2.933 | 4,9 | 0,61 |
| rama: + filtro de metadatos | rama | 78 % | 76 % | 83 % | 0,640 | 2.959 | 10,6 | 0,61 |

- **Pareado, hit a nivel sección** (`noise.paired`, n = 41): denso vs BM25 **7 / 1 / 33, p = 0,070**;
  + prefijo vs denso 1 / 2 / 38, p = 1,0; + léxico vs + prefijo 4 / 2 / 35, p = 0,69; + rerank vs
  + léxico 4 / 4 / 33, p = 1,0; + padre vs + rerank 1 / 4 / 36, p = 0,38; + filtro vs + rerank
  0 / 0 / 41.
- **Qué pieza se ganó su costo por encima de la cota: ninguna.** Hacen falta 6 discordantes todos
  para el mismo lado; el más cerca es el paso de BM25 a denso (7 a 1, p = 0,07), que es también el
  único movimiento grande de la tabla. El rerank cuesta 20 veces más tiempo por consulta (0,61 s
  contra 0,03) y empata 4 a 4: a esta escala no se gana su costo, y la fila «mejor» lo es por
  empate, no por diferencia.
- **El movimiento más grande dentro de los escalones:** «+ padre» vs «+ rerank» (1 ganada, 4
  perdidas en sección). **La caja abierta:** las 4 pérdidas no son la recuperación fallando. En
  `meta-pair-collections-Counter-total-3.9` el denso tenía la sección gold primera
  (`****.*****`) y «+ padre» la reemplazó por su sección padre: el texto con la respuesta sigue
  entregado, pero con el id del padre, y el hit de sección lo cuenta como fallo. Es lo que predice
  la recuperación «de chico a grande» (*small-to-big* / parent-document retrieval): se gana
  contexto alrededor y se pierde granularidad. La tabla lo confirma en agregado: «+ padre» sube la
  respuesta entregada (83 → 85 %) y baja el hit de sección (76 → 68 %). En `t-add-note` la otra
  causa: con padres de ~600 tokens entran 4,9 unidades en vez de 10,6, y la sección de notas de
  excepción quedó afuera del presupuesto. La única ganancia (`t-yield`) es el caso inverso: ningún
  canal tenía la sección `generators` entre sus 10 primeros, y con la expansión a padre entró
  en lo entregado.
- **Brecha documento → sección → respuesta entregada** en la mejor fila: **78 % → 76 % → 83 %**.
  La respuesta llega más seguido que la sección gold, porque la misma clave aparece en la sección
  gemela de la otra versión o en una vecina. Lo que predice para la Etapa 4: el techo de la
  generación es **83 %**, no 78 %.
- **Tokens igualados: muchos hijos o pocos padres.** Al mismo presupuesto, «+ padre» entrega 4,9
  unidades en vez de 10,6: gana 2 puntos en respuesta entregada (85 %) y pierde 8 en sección
  (68 %). Ninguna de las dos diferencias pasa la cota.
- **El filtro de metadatos** se disparó en **0** de las preguntas que nombran una versión que el
  corpus tiene, así que su fila es idéntica a «+ rerank» (0 / 0 / 41). Recordatorio de diseño: el
  filtro de la Etapa 3 **cae al listado sin filtrar** (`fused = kept or fused`). Comparar con §7.
- El reranker no truncó ningún par (0 / 450).

---

## 5 · Etapa 4 · Respuestas fundamentadas, y la tabla 2×2

**Corrió en `full`** (4.970 s, 215 llamadas). `qwen3:8b` en **SPLIT 30 % CPU / 70 % GPU**, 6,0 GB,
contexto 8.192 (`llm.placement`). **Fuente:** `artifacts/full/stage4_rows.json`.

| Fila | correctas (IC95) | abstuvo | abstuvo bien (fuera del corpus) | citas ⊆ entregados | citó el gold | fallos | tokens de prompt | s/resp. |
|---|---|---|---|---|---|---|---|---|
| libro cerrado (sin pasajes) | **0 %** (0-0) | 100 % | 4/4 | — | 0 % | 0 | 216 | 4,6 |
| mejor recuperación («+ rerank») | **59 %** (44-73) | 12 % | 4/4 | 100 % | 66 % | 0 | 3.351 | 44,0 |
| oráculo (la sección gold directo) | **68 %** (54-83) | 5 % | 4/4 | 100 % | 98 % | 0 | 1.455 | 16,4 |

**La tabla 2×2**, en la fila de mejor recuperación (41 ítems con respuesta):

| | respondió bien | respondió mal / se abstuvo |
|---|---|---|
| **la clave llegó** | 23 | **8** («lo tenía y falló») |
| **la clave no llegó** | 1 | **9** (fiel al pasaje equivocado, o se abstuvo) |

- **Mejor recuperación − libro cerrado: +59 puntos**, pareado 24 / 0 / 17, p < 0,001. Sobrevive con
  holgura. Y libro cerrado da **0 %** porque el modelo se abstiene en todos: el prompt le pide
  responder desde el contexto y, sin pasajes, obedece. *No* mide cuánto de Python sabe el modelo;
  mide que la instrucción de abstención funciona.
- **Oráculo − mejor recuperación: +10 puntos**, pareado 8 / 4 / 29, p = 0,39. **No sobrevive** a la
  cota: con 41 ítems no podemos afirmar que arreglar el recuperador gane algo.
- **De las 31 donde llegó la sección, 23 bien y 8 mal** (74 %). Contra el techo de la Etapa 3
  (83 %), la generación alcanza 59 %. **Pero leídas una por una, la mayoría de esas fallas no son
  de generación: son de la métrica.** Contando por clave entregada hay 10 fallas, y en **7** la
  respuesta es correcta y la contención la rechaza porque el modelo contestó corto:
  - 5 preguntas de sí/no contestadas con el sí/no correcto y nada más (`t-else-break` «No»,
    `t-stable-sort` «Sí, se mantiene el orden…», `ChainMap` en 3.2 «No» y en 3.3 «Sí», `Counter.total`
    en 3.9 «No» y en 3.10 «Sí»): la clave exige además la palabra (`estable`, `3.3`, `3.10`);
  - `t-mro`: describe bien el orden (en profundidad, de izquierda a derecha) sin la palabra
    «resolución».

  Las **3** fallas reales: dos abstenciones con la respuesta delante (`t-yield`,
  `id-mod-makeRecord`) y un «Sí» sin sentido a «¿desde qué versión existe `collections.deque`?».
  Conclusión para el informe: la clave conjuntiva (palabra **y** veredicto) es demasiado estricta
  para un generador que responde escueto, y la §7b lo mide en general.
- **Citas inventadas: 0** (todas las citas son de pasajes entregados). **Abstenciones con la
  clave en el contexto: 2** (`t-yield`, `id-mod-makeRecord`). **Inconsistencias** entre el
  booleano y el texto: **0**.
- **Un prompt demasiado grande (§5 del notebook):** nuestra cuenta **3.311** tokens, el servidor
  evaluó **1.026**: `SILENT_TRUNCATION`. Con la guarda prendida el mismo prompt no se manda
  (`ContextOverflow`); con la guarda apagada el servidor descarta el medio sin error. La falla es
  su propia categoría: ni abstención ni respuesta mal.
- **La posición:** primero vs último 3 / 0 / 28, p = 0,25; primero vs medio igual. Hacen falta 6
  discordantes y hubo 3: **a esta escala la prueba no podía mostrarlo** salvo un efecto enorme
  (~10 pasajes en 3.000 tokens).

---

## 6 · Etapa 5 · Una ventana, dos fuentes

**Corrió en `full`** (4.021 s, 197 llamadas; mismo placement que la Etapa 4). **Fuente:**
`artifacts/full/stage5_rows.json`, `stage5_rewrites.txt`, `stage5_summaries.txt`. 29 turnos de test
en 8 guiones.

### 5A · La consulta dentro de la conversación, por dependencia y por turno

hit@10 a nivel **sección**:

| Configuración | todas | sin dependencia (8) | correferencia (10) | elipsis (7) | cambio de tema (2) | vuelta (2) | turno 1 | turno 4 (5) | llamadas |
|---|---|---|---|---|---|---|---|---|---|
| último turno solo | 59 % | 75 % | 50 % | 43 % | 100 % | 50 % | 75 % | 40 % | 0 |
| conversación completa | 62 % | 75 % | 50 % | 57 % | 100 % | 50 % | 75 % | 40 % | 0 |
| reescritura por el modelo | 59 % | 75 % | 50 % | 43 % | 100 % | 50 % | 75 % | 40 % | 32 |
| **reescritura oráculo** | **86 %** | 75 % | **90 %** | **86 %** | 100 % | 100 % | 75 % | **100 %** | 0 |

- **Pareado contra «último turno»**: oráculo **9 / 1 / 19, p = 0,021 (sobrevive)**; conversación
  completa 4 / 3 / 22, p = 1,0; reescritura del modelo **0 / 0 / 29**.
- **Qué ganó en qué dependencia:** la reescritura oráculo gana justo donde el turno depende del
  historial (correferencia 50 → 90 %, elipsis 43 → 86 %) y empata donde no (sin dependencia, 75 %).
  **El ranking cambia con el turno**: en el turno 1 las cuatro empatan (75 %), y en el turno 4 el
  último turno cae a 40 % mientras el oráculo llega a 100 %.
- **La reescritura del modelo no hizo nada.** 0 discordantes contra el último turno, la misma
  mediana de tokens (13). **Leídas las 32** (`stage5_rewrites.txt`): la reescritura del modelo es
  **idéntica carácter por carácter al turno del usuario en las 32**, incluso en «¿Y recursivo?» y
  «¿Y reemplazarlas?». No es un error del código: la caché de llamadas guarda la respuesta cruda
  (`{"query": "¿Y recursivo?"}`), 32 llamadas exitosas. `qwen3:8b` aplicó a todos los turnos la
  regla 4 del prompt («si ya se entiende sola, devolvela igual»). Los 2 casos que el proxy marca
  como «responde la pregunta» (`d-json-csv#4`, `d-sqlite#2`) son falsos positivos del proxy: la
  «reescritura» es el turno original, que ya contiene la clave. Y el único turno en que el modelo
  le «ganó» al oráculo (`d-fechas#2`) es uno donde el turno suelto ya recuperaba bien y la
  reformulación del oráculo no. Hallazgo: la brecha oráculo − modelo (+27 puntos) es lo que vale
  una buena reescritura, y **nuestro prompt no produjo ninguna**. No lo cambiamos después de ver
  el resultado: sería ajustarlo contra el conjunto de test. La próxima versión probaría el prompt
  sin la regla 4, sobre los guiones de dev.

### 5B · El reparto de la ventana

1.400 tokens compartidos; la consulta es la reescritura oráculo (la ganadora de 5A). Nuestra cuenta
y la del servidor coinciden en todos los brazos (máx. 4 tokens de diferencia): nadie truncado.

| pasajes / historial | estrategia | correctas (IC95 por guion) | tokens de historial | pasajes | llamadas/turno | s/turno |
|---|---|---|---|---|---|---|
| 80/20 (1.120 / 280) | completo | 69 % (54-80) | 221 | 4,2 | 1,00 | 29,7 |
| **50/50 (700 / 700)** | completo | **76 %** (55-90) | 404 | 2,3 | 1,00 | 20,1 |
| 30/70 (420 / 980) | completo | 62 % (39-77) | 426 | 1,3 | 1,00 | 16,4 |
| 80/20 | ventana de 2 turnos | 66 % (52-78) | 52 | 4,2 | 1,00 | 24,5 |
| 80/20 | resumen | 69 % (54-80) | 135 | 4,2 | 1,25 | 15,8 |

- **La mejor división es 50/50, y no pasa la cota.** Pareado por guion contra 80/20: 3 / 1 / 4,
  p = 0,63; 30/70 vs 80/20: 0 / 2 / 6, p = 0,5. Con 8 guiones el test de signos no puede bajar de
  0,05 con menos de 6 discordantes para el mismo lado: el nulo es **«la prueba no podía
  mostrarlo»**, no «no hay diferencia». La forma de la curva (sube de 80/20 a 50/50, cae a 30/70)
  es la que predice el enunciado, pero acá es una forma, no un resultado.
- A 80/20 el historial completo se recorta en 21 de 32 turnos; a 50/50 en 7; a 30/70 en 0. A 30/70
  sobra historial y faltan pasajes (1,3 por turno).
- **Estrategias de historial, con la división fija en 80/20** (donde el historial escasea): ventana
  de 2 turnos 0 / 1 / 7; resumen 0 / 0 / 8. Ninguna se distingue del historial completo.
- **Costo por turno:** el resumen agrega 0,25 llamadas por turno (8 resúmenes escritos) y usa 135
  tokens de historial contra 221 del completo, con la misma exactitud. Es el único brazo que
  ahorra ventana sin perder nada medible, pero a esta escala no hay nada que ahorrar: ningún guion
  llena la ventana.
- **Qué hicieron mal los resúmenes** (Wang et al.; leídos los 8 de `stage5_summaries.txt` contra
  los turnos guionados, cuentas en `stage5_summary_audit.json`): sobre **28 afirmaciones**,
  **0 hechos inventados**, **0 relaciones equivocadas** y **26 detalles faltantes**. Los resúmenes
  son fieles a lo que resumen y pierden casi todo lo anterior. El resumen es acumulativo (cada
  llamada recibe el «Resumen previo» y los turnos nuevos), pero el modelo **reescribe** en vez de
  acumular: cada resumen describe solo el último bloque de 6 mensajes, y lo que el anterior decía
  (`datetime`, `ZoneInfo`, `strptime`, `json.load`…) desaparece en el siguiente. 24 de los 26
  faltantes son eso; los otros 2 son nombres de excepción (`CalledProcessError`,
  `TimeoutExpired`), que la regla 1 del prompt pedía conservar. Con un tope de 500 caracteres y
  resúmenes de 43-73 tokens, no es falta de espacio. Eso explica que «resumen» empate con
  «completo» (0 / 0 / 8): en la práctica es una ventana de un bloque reformulada, no una memoria
  del guion. *Lectura hecha por Claude (ver §8).*

---

## 7 · Etapa 6 · Memoria entre sesiones

**Corrió en `full`** (2.736 s, 127 llamadas; mismo placement). **El diseño y sus justificaciones
estaban cerrados antes de correr** (Entrada 6 del registro) y no se tocaron después de ver el
resultado. **Fuente:** `artifacts/full/stage6_rows.json`, `stage6_records.txt`,
`stage6_filter.txt`, `memory/records.jsonl`, `memory/ops.jsonl`. 3 escenarios de test, 8 sesiones,
19 sondas (16 con respuesta, 3 «nunca dicho»).

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

| Brazo | correctas | abstuvo bien | tokens del espacio | recuerdos | tokens de prompt | sondas con toda su evidencia |
|---|---|---|---|---|---|---|
| sin memoria | 5/16 | 3/3 | 0 | 0 | 1.356 | 0/16 |
| reproducción · 400 | **7/16** | 3/3 | 183 | — | 1.539 | 16/16 |
| reproducción · 120 | 4/16 | 3/3 | 103 | — | 1.460 | 8/16 |
| almacén · 400 | 6/16 | 3/3 | 199 | 4,7 | 1.556 | 16/16 |
| almacén · 120 | 6/16 | 3/3 | 107 | 2,3 | 1.464 | 14/16 |
| almacén · 400 + filtro de versión | 2/6 | — | 191 | 4,5 | 1.406 | 6/6 |

Escritura del almacén: **28 llamadas, 12.022 tokens**, 18 registros (14 semánticos, 3
procedimentales, 1 episódico), operaciones ADD 15 · UPDATE 3 · INVALIDATE 1 · NOOP 1, 4 registros
cerrados, 0 rechazados.

**Por habilidad** (en cuentas, nunca en porcentaje):

| Habilidad | sin memoria | reproducción · 400 | reproducción · 120 | almacén · 400 | almacén · 120 |
|---|---|---|---|---|---|
| extracción (4) | 1/4 | 1/4 | 1/4 | 1/4 | 1/4 |
| temporal (4) | 0/4 | 1/4 | 0/4 | 1/4 | 1/4 |
| actualización de conocimiento (3) | 1/3 | 1/3 | 0/3 | 1/3 | 1/3 |
| multi-sesión (3) | 1/3 | 2/3 | 1/3 | 2/3 | 2/3 |
| abstención (3) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| preferencia (2) | 2/2 | 2/2 | 2/2 | 1/2 | 1/2 |

**Por régimen** (el único corte que hace contrastable la predicción de Mem0):

| | sin memoria | reproducción · 400 | reproducción · 120 | almacén · 400 | almacén · 120 |
|---|---|---|---|---|---|
| la reproducción entra en 120 (4) | 2/4 | 2/4 | 2/4 | 1/4 | 1/4 |
| la reproducción NO entra en 120 (12) | 3/12 | 5/12 | 2/12 | 5/12 | 5/12 |

- **Ninguna comparación pasa la cota.** Reproducción · 400 vs sin memoria 2 / 0 / 14 (p = 0,5);
  almacén · 400 vs reproducción · 400 0 / 1 / 15; almacén · 120 vs reproducción · 120 **3 / 1 / 12**
  (p = 0,63). Con 16 sondas hacen falta 6 discordantes para el mismo lado, y las sondas de un
  escenario comparten persona y almacén: no son independientes.
- **Lo que sí se ve, como forma:** en el régimen ancho la reproducción entera es la mejor fila
  (7/16), como en la evaluación de Mem0. En el angosto, donde la transcripción no entra, la
  reproducción cae a 4/16 (por debajo de no tener memoria) y el almacén **no cae** (6/16): conserva
  la evidencia de 14 de 16 sondas contra 8 de 16. Es la dirección que predice Mem0, con 3 a 1
  discordantes.
- **Dónde ganó la memoria:** multi-sesión (1/3 → 2/3) y temporal (0/4 → 1/4). **Dónde perdió el
  almacén contra la reproducción:** preferencia (2/2 → 1/2). Abstención es 3/3 en todos los
  brazos: el «nunca dicho» no se inventa.
- **El costo, como punto de equilibrio:** el almacén **no ahorra** tokens de prompt a esta escala
  (1.556 contra 1.539 de la reproducción: −17 por sonda) y costó 12.022 tokens escribirlo, así que
  **no se amortiza nunca** con transcripciones de 56 tokens por sesión. La ventaja de costo de Mem0
  aparece cuando la transcripción es grande.
- **Una actualización de conocimiento de punta a punta:** el notebook la imprime entera (la línea
  de tiempo del registro, las operaciones que lo escribieron y qué contestó cada brazo). Se
  **invalida, no se borra** (UPDATE le pone `valid_to` al viejo y escribe uno nuevo con
  `supersedes`).
- **Qué guardó el extractor que no era cierto o que no se dijo:** los chequeos mecánicos dan 0
  marcas de razonamiento, 0 coberturas («probablemente»), 0 filtraciones de `</think>`, 2 registros
  con solapamiento bajo con su sesión de origen y **4 `subjects` con dos valores vigentes** (un
  UPDATE que el extractor no hizo). **Leídos los 18 registros contra su sesión de origen**
  (`stage6_records.txt` → `stage6_record_audit.json`):
  - **contenido:** 0 falsos y 0 hipótesis; **1 nunca dicho**: `s-backup#m007` dice «Pasé todo a
    logging, con un archivo de registro **único** (vigente desde 2026-04-10)». En la sesión 3 el
    usuario dijo que *quiere* un registro único, no que lo tiene, y la fecha es la de la sesión 2.
    Es el registro con solapamiento 0,17, el más bajo, así que el chequeo mecánico apuntaba bien.
  - **vigencia — donde está casi todo el daño:**
    - **2 registros siguen vigentes ya superados**: Python 3.11 de `s-etl` con 3.13 al lado, y
      «imprime todo con print» de `s-backup` después de pasar a logging.
    - **3 cerrados por error**, cada uno «reemplazado» por un registro que no tiene nada que ver:
      la preferencia «solo biblioteca estándar» cerrada por «le molesta armar los lotes a mano»;
      «no quiero que el archivo crezca sin límite» cerrada el mismo día por «pasé todo a logging»;
      y «fechas en UTC» cerrada aunque la sesión 2 dice que se guarda la zona **también**, «no
      solo UTC».
    - **1 actualización no guardada**: la zona horaria por turno de `s-turnos` nunca se escribió.
  - **2 `subjects` mal puestos**: «usuarios en Argentina» bajo `sistema_operativo`, y «imprime todo
    con print» bajo `preferencia_api`.

  O sea que el extractor **no inventa hechos**: se equivoca en **cuándo un hecho deja de valer**.
  La preferencia que el almacén pierde (1/2 contra 2/2 de la reproducción) es `s-backup#p-print`
  («¿con qué reemplazo mis print para filtrar por nivel?»). Lo que el almacén puso en el espacio
  fue justo el registro mal etiquetado, «[preferencia_api] Imprime todo con print», y la respuesta
  habló de filtros sin nombrar `logging`. La reproducción le mostró la frase original del usuario
  y acertó. *Lectura hecha por Claude (ver §8).*
- **¿La memoria mejoró la recuperación como filtro, o la perjudicó?** La perjudicó, de las dos
  maneras posibles, medidas: en **1** sonda el filtro dejó **cero** candidatos (el usuario de
  `s-etl` está en 3.11 y el corpus solo tiene 3.12 y 3.13) y en **2** sacó el pasaje gold, que
  estaba en otra versión que la del usuario. En las 6 sondas con gold el brazo filtrado empata con
  el sin filtro (0 / 0 / 6, 2/6 cada uno): a esta escala el daño al recuperador no llegó a la respuesta.
  Es la política **opuesta** a la del filtro de la Etapa 3 (§4), y las dos se justifican: una
  versión que el **usuario tipeó** puede ser un error de tecleo; una versión que el **sistema
  guardó** es una afirmación propia sobre su usuario.
- **¿En qué punto el escenario deja de entrar en la ventana?** No deja de entrar. A 56 tokens por
  sesión, la transcripción cruza el espacio de 120 a las ~1,6 sesiones, el de 400 a las ~6,5 y el
  lugar que queda en la ventana a las ~110. Por eso hay dos regímenes.

### El hallazgo que no esperábamos (contribución original)

La fórmula de puntaje que sugiere el enunciado —`minmax(recencia) + minmax(importancia) +
minmax(relevancia)`— **se degrada a dos términos y un bonus** cuando las sesiones están separadas
por meses, que es el caso realista de un asistente personal. Con `0.995/hora`, el registro más
nuevo puntúa ~1 de recencia y el segundo ~2e-4: después del min-max eso **no es un orden por
antigüedad, es un one-hot del último registro**. Medido en la corrida: la recencia tuvo rango
exactamente 0 en 6 sondas, y en las demás va de 0 a 0,89 con todo el peso en un registro.

---

## 7b · Etapa 7a · Un juez, calibrado antes de leerlo

**Corrió en `full`** (3.876 s, 676 llamadas). **Fuente:** `artifacts/full/stage7_rows.json`,
`results/07_judge.jsonl`, `stage7_audit.txt`. El juez **relee** las respuestas guardadas de las
Etapas 4-6 y no genera ninguna: **396 casos** (135 de la Etapa 4, 160 de la 5, 101 de la 6), todos
con referencia (rúbricas: 327 clave, 42 abstención, 16 actualización, 11 preferencia). Dos jueces
en pasadas separadas, `num_ctx = 4096`, temperatura 0: `qwen3:8b` (**el mismo modelo que
generó**; SPLIT 25 / 75) y `granite3.3:8b` (IBM, **otra familia**; SPLIT 24 / 76).

**Cada juez contra la contención:**

| Juez | n | juez sí · cont. sí | juez sí · cont. no | juez no · cont. sí | juez no · cont. no | acuerdo |
|---|---|---|---|---|---|---|
| mismo modelo (`qwen3:8b`) | 396 | 206 | 41 | 16 | 133 | **85,6 %** |
| otra familia (`granite3.3:8b`) | 396 | 205 | 75 | 17 | 99 | **76,8 %** |

Por etapa, el juez propio acuerda 86,7 % (E4) · 83,1 % (E5) · 88,1 % (E6); Granite 76,3 % · 79,4 % ·
73,3 %. Por rúbrica, el desacuerdo se concentra en **actualización de conocimiento**: 43,8 % de
acuerdo con el juez propio y 25,0 % con Granite, casi todo «juez sí / contención no».

- **Autopreferencia: no aparece en la dirección esperada.** Sobre los mismos 396 casos, el juez
  propio aprueba **247** y el de otra familia **280**. Pareado: 50 casos que solo aprueba Granite
  contra 17 que solo aprueba Qwen, p < 0,001. Si Qwen favoreciera sus propias respuestas, la
  asimetría iría al revés. Lo que la tabla muestra es que **Granite es más indulgente**: 75 de sus
  «sí» van contra la contención, contra 41 del juez propio. Que no haya autopreferencia visible no
  prueba que no exista: puede estar tapada por un juez ajeno más indulgente. La auditoría a mano
  es lo que separa las dos lecturas.
- **Dónde el juez no coincidió con la contención:** sobre todo en «juez sí / contención no» (41 y
  75 casos). En la rúbrica de actualización, leídos, **la que se equivoca es la contención**:
  «No, no debés seguir usando print… usá `logger.debug()`» es la respuesta correcta a «¿Sigo usando
  print?», pero `logging` no está en el texto. Y «Sí» a «¿Y ahora puedo usar `itertools.batched`?»
  es correcto (el usuario pasó a 3.13), pero `3.12` tampoco está.

**La auditoría** (`stage7_audit.txt` → `stage7_audit.json`): **79 casos** (max(30, 20 %)), que
arrancan por los 57 desacuerdos del juez propio con la contención y se completan al azar (semilla
0). Regla de lectura: una respuesta es correcta si con ella el usuario resuelve lo que preguntó. En
sí/no alcanza el sí/no correcto; si falta la función o clase con la que se actúa («usá rotación»,
«el formato `%Y-%m-%d`» sin `strftime`), es incorrecta. *La lectura la hizo Claude, no un
integrante del grupo (ver §8).*

| Fuente | n | sí · auditoría sí | sí · auditoría no | no · auditoría sí | no · auditoría no | acuerdo |
|---|---|---|---|---|---|---|
| juez mismo modelo (`qwen3:8b`) | 79 | 43 | 11 | 11 | 14 | 72,2 % |
| juez otra familia (`granite3.3:8b`) | 79 | 52 | 8 | 2 | 17 | **87,3 %** |
| contención | 79 | 24 | 5 | 30 | 20 | 55,7 % |

- **En cuál confiar.** La muestra está sesgada a propósito hacia los desacuerdos, así que estos
  porcentajes no estiman el acuerdo sobre los 396: dicen **quién tiene razón cuando discrepan**. Y
  ahí la contención pierde: **30 respuestas correctas que rechaza** (sí/no escuetos, paráfrasis,
  `glob('**/*.csv')` por `rglob`) contra 5 incorrectas que acepta. Esas 5 son casos donde la clave
  aparece por accidente: `check_output()` contiene `check`, y la cita
  `[py3.13:library/zoneinfo#…]` contiene `zoneinfo`. El juez propio es el peor de los dos jueces
  por lo contrario a la autopreferencia: rechaza respuestas correctas por «agregar información que
  no está en la clave» y le pide a `can_answer=false` que «explique».
- **Lo que esto cambia de las tablas anteriores:** los números de las Etapas 4-6 son **cotas
  inferiores**. La contención subestima sobre todo las filas que responden escueto, que son las de
  sí/no y las de memoria.
- **El sesgo que no podemos sacar:** el auditor es un tercer modelo de lenguaje (Claude). Que
  coincida más con Granite puede querer decir que Granite tiene razón, o que dos modelos que no
  generaron las respuestas comparten criterio. Una lectura humana de los mismos 79 casos es lo
  que cerraría esto.

---

## 8 · Cómo usamos IA en este trabajo

Usamos **Claude** (Anthropic, a través de Claude Code) en casi todo el trabajo, y lo decimos
etapa por etapa para que se pueda juzgar qué es de quién:

- **Código y diseño.** Como dice la Entrada 1 del registro, Claude escribió e iteró el código de
  las etapas y discutimos el diseño con él. Las decisiones del registro son nuestras en el sentido
  de que las tomamos y las firmamos, pero la mayoría se redactaron con Claude.
- **Conjunto de evaluación.** Rigió la regla de la Entrada 1: ningún ítem se escribió con el pasaje
  gold abierto, un ítem propuesto por un modelo era una propuesta hasta verificarlo contra el
  corpus, y el chequeo de contaminación con BM25 corrió sobre todos. Las preguntas de identificador
  y de versión salen de código que lee el corpus; ese código también lo escribió Claude, y §9
  documenta un error que dejó pasar.
- **La corrida.** Claude armó el entorno en la RTX 3050 y corrió todas las etapas en `full`. En el
  camino arregló tres fallas: el zip recompilado (re-pin y re-congelamiento, Entrada 8), un
  `KeyError` de la Etapa 6 con filas sin JSON parseable, y el CRLF que la prueba del candado de la
  Etapa 2 dejaba en Windows. También escribió la Etapa 7a y llenó las tablas de §4-7b desde los
  archivos de resultados.
- **Las lecturas «a mano».** Las hizo **Claude, no un integrante del grupo**, con las reglas que
  cada sección declara: el libro cerrado y los diez documentos crudos de la Etapa 0, los diffs de
  limpieza, las atribuciones nota→definición, las cajas abiertas de las Etapas 3-5, los 8
  resúmenes, los 18 registros del almacén y los 79 juicios del juez. Son lecturas de un tercer
  modelo, no una auditoría humana. Por eso cada una dice de qué archivo sale y deja el `.json` con
  las cuentas, para que una persona pueda rehacerla y comparar.
- **Lo que verificamos nosotros:** **PENDIENTE** — completar acá qué revisó cada integrante (por
  ejemplo, una muestra de las lecturas de arriba) antes de entregar. Si no se revisó nada, esta
  línea tiene que decirlo.

---

## 9 · Lo que este trabajo no puede decir

- **Si el asistente le sirve a alguien.** 53 ítems con clave de respuesta miden si la respuesta
  contiene la palabra correcta; no si el párrafo alrededor es útil, ni si la cita es la que un
  humano habría elegido entre dos pasajes igual de buenos.
- **Nada con 19 sondas de 3 escenarios es 19 observaciones independientes.** Por eso la Etapa 6 no
  publica intervalo por sonda, y las comparaciones van pareadas con la cota declarada.
- **Las 25 preguntas escritas a mano las escribimos nosotros**, que ya decidimos qué nos parece una
  buena pregunta de la documentación de Python.
- **Las lecturas «a mano» no son humanas.** Todas las lecturas de este informe las hizo Claude
  (§8). Una persona puede rehacerlas sobre los mismos archivos; ninguna cuenta de este informe
  depende de que lo haga, pero las conclusiones de §7b sobre en qué instrumento confiar sí.
- **Un ítem del conjunto congelado tiene la clave equivocada.** Leídas las atribuciones
  nota→definición de `stage2_note_owners.txt` (20 distintas), **6 están mal**: la heurística de
  posición le asigna a una clase la nota de versión de uno de sus métodos o de una variante
  vecina. Ejemplos: `Counter` «nuevo en 3.2» y «en 3.3» (son de sus métodos), `contextlib.closing`
  «en 3.10» (es `aclosing`), `contextmanager` «en 3.7» (es `asynccontextmanager`). Una de esas
  atribuciones se volvió un ítem de test: `meta-ver-collections-deque` pregunta desde qué versión
  existe `collections.deque` y su clave es **3.5**, pero `deque` existe desde 2.4; la nota de 3.5 es
  de sus métodos. No lo editamos después de tener resultados. En la Etapa 4 el modelo contestó un
  «Sí» que es incorrecto con cualquier clave, así que ninguna fila cambia; pero el conjunto tiene
  **un error de etiqueta conocido** y la heurística tiene una tasa de error de ~30 % en esta
  muestra.
- **Los diffs de limpieza no inventan texto** (leídos `library/chunk` y `howto/logging-cookbook`):
  el `¶` pasa a salto de línea donde corresponde. Dejan dos cosas: la etiqueta de versión queda a
  medio traducir («Obsoleto desde la versión 3.11, removed in version 3.13», porque la regla
  traduce solo la primera mitad de la etiqueta) y el código de los ejemplos queda aplanado en la
  prosa con los tokens espaciados (`logger . setLevel ( logging . DEBUG )`).

---

## Anexo · Cómo se corrió, y cómo reproducirlo

Corrida del 2026-10-31, todo en una RTX 3050 de 6 GB (Windows 11, Python 3.13, torch 2.11 + CUDA
12.8, Ollama 0.34.4):

```bash
# entorno
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
ollama pull qwen3:8b ; ollama pull granite3.3:8b      # (y qwen3:1.7b, granite3.3:2b para smoke)
.venv/Scripts/python verify_setup.py --model qwen3:1.7b

# el servidor lo levanta llm.ensure_server con estas dos variables, para TODAS las etapas
export OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0

SMOKE_TEST=1 .venv/Scripts/python tools/run_stages.py   # humo, 00-07
SMOKE_TEST=0 .venv/Scripts/python tools/run_stages.py   # full, 00-07
```

| Etapa | segundos | llamadas al LLM |
|---|---|---|
| 00 · auditoría resuelta | 137 | 5 |
| 00b · nuestra auditoría | 139 | 5 |
| 01 · documentos | 88 | 0 |
| 02 · conjunto de evaluación | 12 | 0 |
| 03 · recuperación | 953 | 0 |
| 04 · respuestas | 4.970 | 215 |
| 05 · conversación | 4.021 | 197 |
| 06 · memoria | 2.736 | 127 |
| 07 · juez | 3.876 | 676 |

`placement` dio **SPLIT 30 % CPU / 70 % GPU** para `qwen3:8b` a 8.192 y **25 / 75** a 4.096 (el
juez); `granite3.3:8b`, 24 / 76. La Entrada 7 del registro decide aceptar el split y reportarlo,
porque afecta a todos los brazos de una etapa igual: cuesta horas, no validez. Un detalle operativo
que no estaba previsto: entre la Etapa 0 y la 3 el `qwen3:8b` quedó cargado (keep-alive de 30 min)
mientras la Etapa 3 encodeaba con `bge-m3`, y se lo descargó a mano (`ollama stop qwen3:8b`) para
que el embebedor tuviera la placa. No cambia ningún número.

Los notebooks con salidas del zip de entrega se ejecutaron **después** de la corrida, sobre las
cachés (`llm_cache`, `embeddings`, `results`). Por eso sus tiempos de celda son de reanudación, no
de cómputo.
