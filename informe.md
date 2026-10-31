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
> `stage*_rows.json`; las lecturas al lado de cada tabla son nuestras. Lo que sigue **PENDIENTE** son las lecturas a
> mano que ningún script puede hacer (cajas abiertas, resúmenes, registros del almacén, auditoría del
> juez) y §8; cada hueco dice de qué archivo sale.

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

El detalle, con fechas y alternativas descartadas, está en `decision_log.md` (8 entradas). Acá el
resumen de las que condicionaron todo lo demás:

| Decisión | Alternativa descartada | Qué nos llevó ahí |
|---|---|---|
| **Corpus `python-docs-es` por HTML compilado** | los `.po` del repo de traducción | el `.po` es la unidad de traducción, no la de lectura: entradas sueltas, `fuzzy`, obsoletas. El HTML es lo que un usuario lee (Entrada 2) |
| **Dos versiones (3.13 y 3.12), no una** | una sola versión | el corpus necesita documentos confundibles para que la recuperación tenga trabajo; las páginas gemelas comparten de 0,35 a 1,00 de 5-gramas (MinHash, Etapa 0) |
| **Todo en una RTX 3050 de 6 GB** | el reparto híbrido con una T4 de Colab (plan original, Entrada 1) | cada etapa que cruza la frontera entre dos máquinas puede desalinear el `documents_sha256` o el `index_key`, y el chequeo que lo detecta corre *después* de gastar las llamadas. El costo: `qwen3:8b` no entra en 6 GB (~5,2 de pesos + ~1,2 de caché KV a 8.192), así que la corrida se reporta con lo que diga `llm.placement()` (Entrada 7) |
| **Etapa 3 acotada a 3 filas** | el barrido completo de piezas | el presupuesto de la Entrada 1 no daba para más filas *y* llegar a la Etapa 6, que es donde está la pregunta del proyecto |
| **Etapa 7a sí, 7b no** | el framework (7b) | la 7a relee respuestas guardadas y no genera ninguna nueva: con todo en una placa es barata. La 7b es otra corrida completa de las Etapas 5-6 (Entrada 8) |
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
  perdidas en sección). *La caja abierta —leer esos ítems y decir si perdió por la razón que
  predice su paper— queda PENDIENTE de una persona* (`stage3_rows.json` y el log).
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
- **De las 31 donde llegó la clave, 23 bien y 8 mal** (74 %). Las 8 fallas se clasifican con
  `key_contained`, `verdict_ok` e `inconsistent`; leerlas una por una queda *PENDIENTE de una
  persona*. Contra el techo de la Etapa 3 (83 %), la generación alcanza 59 %: la mayor parte de lo
  que se pierde se pierde en la generación, no en la recuperación.
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
  mediana de tokens (13): a juzgar por eso, `qwen3:8b` devolvió casi siempre la pregunta tal cual. Contra el oráculo
  perdió en 9 turnos y ganó en 1 (`d-fechas#2`). De las 32 reescrituras, 2 *responden* en vez de
  reformular (`rewrite_proxies`). Leer las 20 volcadas en `stage5_rewrites.txt` para decir **por
  qué** queda *PENDIENTE de una persona*. Hallazgo: la brecha oráculo − modelo (+27 puntos) es lo
  que vale una buena reescritura, y nuestro prompt de reescritura no la captura.

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
- **Qué hicieron mal los resúmenes** (hechos inventados, relaciones equivocadas, detalles
  faltantes; Wang et al.): *PENDIENTE de una persona* — `stage5_summaries.txt` →
  `stage5_summary_audit.json`.

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
  UPDATE que el extractor no hizo). El conteo de falsos y nunca dichos es *PENDIENTE de una
  persona* — `stage6_records.txt` → `stage6_record_audit.json`.
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
  75 casos). Es la celda donde un juez puede corregir paráfrasis que la contención castiga, o
  aprobar respuestas que no dicen la clave. En la rúbrica de actualización la sospecha es lo
  segundo: los dos jueces aprueban respuestas que dan el valor viejo o no lo dan. Hay que leerlo
  antes de afirmarlo.
- **En cuál confiar: todavía en la contención.** La matriz contra una persona es *PENDIENTE*:
  `stage7_audit.txt` tiene **79 casos** (max(30, 20 %)), empezando por todos los desacuerdos del juez
  propio con la contención, y el veredicto propio se carga en `stage7_audit.json`. Hasta que esté,
  el juez es una alarma sobre la contención, no un reemplazo.

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
