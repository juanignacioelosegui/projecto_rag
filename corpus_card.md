# corpus_card.md — python-docs-es

Los números salen de dos corridas del perfil `full`: la auditoría
(`00b_corpus_audit_ours.py` → `artifacts/full/stage0_ours.json`) y la construcción
(`01_documents.py` → `artifacts/full/documents.jsonl` + `stage1_documents.json`). Donde la
Etapa 1 corrigió a la Etapa 0, queda dicho cuál era el error.

**El corpus, en una línea.** 695 documentos (335 de 3.13 + 360 de 3.12), 5.614 secciones,
3.474.559 tokens del generador, `corpus_sha256`
`6c58d2cf300c1ff61649139369495ff70ad8766b15acabd8af74a0d5228aa7c6`.

---

## 1 · Fuente y versión fijada

Documentación oficial de Python traducida al español, en su forma **compilada a HTML**
(no los `.po` del repo de traducción; el por qué está en `decision_log.md`, entrada 2).

| | |
|---|---|
| origen | `https://docs.python.org/es/{v}/archives/python-{v}-docs-html.zip` |
| versiones | 3.13 (corpus principal) y 3.12 (distractor de versión) |
| subdirectorios | `library/`, `builtins/`, `tutorial/`, `howto/`, `reference/`, `faq/` |
| fecha de descarga | 2026-10-24 |
| sha256 3.13 | `59f4ffeee535aacf14786d7960b829aa047ea26d4aa76f9a841fce5d1fb3db67` |
| sha256 3.12 | `ed1d2613c23620f83525a0453495fed896b0812ccc0a3e7393917dd5ba90b59d` |
| parser | `corpus_html.py` (un módulo, no una celda: la Etapa 0 y la Etapa 1 leen las páginas igual) |

**El pin es el sha256, no la URL.** El zip se recompila *en el mismo lugar*: de los dos
sha256 que la auditoría resuelta del curso fijó el 2026-09-27, hoy no coincide ninguno. De
la Etapa 1 en adelante la descarga corre con `on_mismatch="raise"`: si upstream recompila,
el pipeline se cae en vez de evaluar contra otro corpus en silencio.

Verificado: bajamos 3.13 **dos veces seguidas** y comparamos. Mismos bytes y mismos ids de
página, sección y ancla. El id es estable dentro de una compilación. Contra 3.12 se mueven
1.678 anclas (1.211 solo en 3.13, 467 solo en 3.12, 9.548 compartidas).

**Identificadores.** `doc_id = "py3.13:library/pathlib"`; `section_id = doc_id + "#" + id`
del `<section>`, que Sphinx deriva del título del `.rst`; las anclas de API son los `id` de
los `<dt>`. Los tres vienen de la fuente, no los inventamos nosotros.

## 2 · Licencia y atribución

- Texto de la documentación: **PSF License Agreement** (Python Software Foundation).
- Ejemplos de código dentro de la documentación: **0BSD**.
- Las traducciones aportadas a `python-docs-es`: **CC0**.

Uso compatible con el trabajo (educativo, redistribución del derivado). **Cada documento
guarda su URL canónica** (`https://docs.python.org/es/3.13/<rel>`) en el manifiesto y en los
metadatos del chunk, y las respuestas de la Etapa 4 citan esa URL. La atribución no es un
trámite: es también el id que hace verificable una cita.

## 3 · Regla del subconjunto

Clasificador de tres clases, **evaluado en orden** (el orden es la decisión; ver
`decision_log.md`, entrada 3, que registra el borrador que medir rompió):

```
1. tiene nota de deprecación/eliminación  Y  tokens < 500  ->  "aviso de eliminación"  SE QUEDA
2. tokens < 100                                            ->  "puntero"               SE DESCARTA
3. resto                                                   ->  "documentación"         SE QUEDA si inglés <= 60 %
```

El inglés por página se mide con `corpus_html.paragraph_language` (voto por palabras
funcionales sobre cada párrafo, sin decidir si hay menos de 4 marcadores).

| clase | páginas | tokens | anclas de API | efecto |
|---|---:|---:|---:|---|
| 1 · aviso de eliminación | 24 | 2.514 | 0 | se queda con `status=deprecated` |
| 2 · puntero | 7 | 442 | 0 | se descarta |
| 3 · documentación en español | 311 | 1.684.368 | 9.314 | se queda |
| 3 · documentación en inglés | 43 | 197.674 | 1.445 | se descarta |

**En 3.13: 335 de 385 documentos, 1.686.882 tokens, 9.314 de 10.759 anclas (86,6 %).** La regla
cuesta 13 % de las anclas: el material de las preguntas de identificador sobrevive.

Aplicada a las dos versiones (Etapa 1): **695 documentos** = 335 de 3.13 + 360 de 3.12, con
3.474.559 tokens en total. 3.12 pierde menos (14 páginas por inglés contra 43) porque su
traducción está más completa en las páginas que 3.13 reorganizó.

### Mapeo de estado a `DocMetadata.status`

`DocMetadata.status` solo acepta cinco valores; el valor original queda en `source_status`.

| de la fuente | `status` | por qué |
|---|---|---|
| página de 3.13 | `current` | 3.13 es la rama vigente del corpus |
| página de 3.12 | `superseded` | la rama entera quedó atrás; el contenido no es falso, es viejo |
| aviso de eliminación, cualquier versión | `deprecated` | la página dice de sí misma que el módulo se fue |

29 documentos quedan `deprecated` (24 en 3.13, 5 en 3.12). Verificado contra el cuerpo: los 29
contienen una frase de eliminación; **0 desacuerdos**.

Por qué cada cláusula existe:

- **1** rescata los 24 módulos removidos (`aifc`, `asynchat`, `asyncore`, `audioop`, `cgi`,
  `imp`, `telnetlib`, …). Son páginas de ~100 tokens, están en inglés, y su contenido *es*
  la respuesta a "¿puedo usar X en 3.13?". Es el mejor material de abstención del corpus.
- **2** saca páginas índice sin contenido: `library/concurrent.html` tiene 31 tokens,
  `howto/clinic.html` 36. Indexarlas mete documentos que pueden ganar por similitud de
  título sin tener nada que citar.
- **3** evita que la mayoría del corpus sean pasajes en inglés contra consultas en español.
  No es una regla de calidad de traducción: es el techo que ya tiene el corpus.

**El costo, con nombre propio — y la corrección que trajo la Etapa 1.** `builtins/` de 3.13 está
sin traducir y la regla lo descarta (`stdtypes` 99 % inglés, `functions` 98 %, `exceptions`
97 %). La Etapa 0 concluyó de ahí que *"ninguna pregunta nuestra puede ser sobre `print` o
`len`"*. **Eso era falso**, y se vio al aplicar la regla a las dos versiones: 3.13 movió esas
páginas a `builtins/` y dejó un stub de redirección en `library/`, pero **3.12 todavía las tiene
en `library/`, traducidas**, y la regla las conserva.

| página | en 3.13 | en 3.12 | anclas recuperadas | status del único gold |
|---|---|---|---:|---|
| `functions` | stub de redirección | 19.824 tk, 18 % inglés | 67 | `superseded` |
| `stdtypes` | stub de redirección | 47.795 tk, 11 % inglés | 216 | `superseded` |
| `exceptions` | stub de redirección | 8.034 tk, 10 % inglés | 104 | `superseded` |
| `constants` | stub de redirección | 798 tk, 38 % inglés | 11 | `superseded` |

Entonces sí se puede preguntar por `print` o por `len`, **pero el gold es una sola página y es
`superseded`**. Las consecuencias no son cosméticas:

- el gold de esas preguntas **no puede ser un conjunto de dos versiones**: solo existe una;
- el **filtro por versión** de la Etapa 3, si restringe a 3.13, las vuelve incontestables. Es el
  caso en que un filtro de metadatos sube la precisión y pierde la respuesta, y es la razón por
  la que ese filtro es una **rama etiquetada** y no un escalón de la escala acumulativa;
- es un gancho de memoria mejor que el que teníamos: "estoy en 3.13" + una pregunta de builtins
  es un conflicto real entre lo que el usuario declaró y lo que el corpus puede probar.

Lo que **sí** se va sin red: `library/tkinter.html` (87 % inglés, 599 anclas), la página con más
anclas del corpus, y `library/token.html` (76 %).

## 4 · Defectos conocidos, y qué hizo la limpieza

### Lo que la limpieza cambió, regla por regla

Medido sobre los 695 documentos. La unidad es el **bloque** (el texto propio de una sección), no
el documento: la limpieza corre por bloque para poder recalcular los offsets de `start`/`end`.

| regla | bloques que tocó | caracteres | qué arregla |
|---|---:|---:|---|
| `¶` → salto de línea | 5.942 | −53.440 | el marcador de ancla, que es el único separador entre título y cuerpo |
| firma partida (`pathlib. Path` → `pathlib.Path`) | 1.886 | −14.148 | el prefijo de módulo que Sphinx pone en un `<span>` aparte |
| etiqueta de nota al español | 1.095 | **+10.314** | `"Added in version 3.13"` → `"Nuevo en la versión 3.13"`; el número no se toca |
| NFC | 4 | 0 | acentos descompuestos (4 bloques: pocos, pero existen) |
| espacios en blanco | 0 | 0 | nada: `corpus_html` ya los colapsa al extraer |

13.968.785 caracteres crudos → 13.915.708 limpios (0,38 % menos). La etiqueta de nota **agrega**
caracteres y está bien: el largo no es el objetivo, el número de versión sí, y hay un assert que
comprueba que los 2.310 números declarados por las notas siguen literales en el texto limpio.

**El defecto que un loader ingenuo deja pasar sin ningún error: la marca `¶`.** Sphinx le cuelga
un ancla `¶` al final de cada encabezado y de cada firma. En texto plano esa marca es *lo único*
que separa el título del cuerpo: `"Excepciones ¶ exception pathlib. UnsupportedOperation ¶ Una
excepción que…"`. Borrarla —la reacción obvia, porque parece basura— da `"Excepciones exception
pathlib.UnsupportedOperation Una excepción que…"`: pega el encabezado a la firma y la firma a la
prosa. No tira ninguna excepción. El documento se indexa, el pipeline corre, y los pasajes
recuperados tienen una frase inventada en cada borde. La reemplazamos por `"\n"`, que es lo que
la marca significa: acá termina un elemento anclable. Se encontró leyendo el diff antes/después
de `pathlib`, no con un test.

**Las firmas se arreglan con el id, no con una regex.** `r"(\w)\.\s+(\w)"` arreglaría
`pathlib. Path` y rompería toda oración que termine en abreviatura. Lo que usamos es el dato que
la fuente ya da: el `anchor_id` (`pathlib.UnsupportedOperation`); de cada ancla generamos las
variantes con un espacio después de cada punto y reemplazamos solo esas.

Y se generan de **todos los sufijos** del id, no solo del id completo. El primer intento solo
cubría la forma calificada (`pathlib.Path. open`) y dejaba **3.364 ocurrencias** intactas, porque
la firma que la página muestra es casi siempre la corta (`Path. open`). Lo encontró la Etapa 2 al
parsear firmas para derivar preguntas de valor por omisión, no un test de la Etapa 1. Después de
extender la regla quedaban 4, todas en `py3.12:library/socket`: firmas con **dos** puntos
partidos (`socket. socket. type`), que `re.sub` no arregla en una pasada porque retoma la lectura
*después* del reemplazo. Se itera a punto fijo y el contador de restos da **0**. El riesgo de
falso positivo se midió: de las 202 ocurrencias con la cola en mayúscula (`TLSVersion. TLSv1`,
`blake2b. SALT_SIZE`) se inspeccionaron las 20 más frecuentes y todas eran firmas, así que la
regla no lleva guarda de excepciones.

### Lo que decidimos NO arreglar

**Los espacios dentro de los ejemplos de código.** El resaltado de Sphinx parte cada token en un
`<span>`, así que el texto plano queda `>>> [ x for x in p . iterdir () if x . is_dir ()]`. No lo
tocamos: distinguir un bloque de código de la prosa que lo rodea requeriría volver al HTML y
tratar `<pre>` aparte, y el arreglo de firmas por `anchor_id` no llega ahí porque los ejemplos
usan nombres locales (`p`, `q`), no identificadores calificados. **Lo que cuesta:** una consulta
léxica por `p.iterdir()` no matchea el ejemplo. **Por qué es tolerable:** la firma canónica
(`pathlib.Path.iterdir`) sí está limpia, y es donde vive la respuesta. Queda anotado para que si
la Etapa 3 ve fallar las preguntas de identificador, este sea el primer sospechoso y no una
sorpresa.

### Defectos de la fuente

Medidos sobre las 385 páginas de 3.13 antes de la regla del subconjunto.

| defecto | cuánto | quién lo paga |
|---|---:|---|
| marcas `¶` de ancla pegadas al texto | 13.757, en 385 páginas | Etapa 1 las borra, con el antes/después medido |
| stubs de redirección (cuerpo vacío) | 4 (`library/{constants,exceptions,functions,stdtypes}.html`) | el parser devuelve `None`; no entran |
| párrafos en inglés (>10 % de la página) | 251 páginas | Etapa 3: la recuperación es **cross-lingual**, no monolingüe |
| páginas que Sphinx nunca mandó a traducir | 9 (100 % sin marcar) | Etapa 1 las marca; la regla del subconjunto decide |
| etiquetas de nota de versión en dos idiomas | 214 páginas, 163 con las dos mezcladas | Etapa 1 normaliza la etiqueta y **conserva el número** |
| firmas con espacios de más (`pathlib. UnsupportedOperation`) | en los `<dt>` | Etapa 1, antes de que la firma sea texto indexable |
| **sin fecha de documento** | 0 % de cobertura | Etapa 6: los timestamps de memoria son **simulados** |
| la versión no discrimina *dentro* de una versión | 100 % de cobertura, por construcción | Etapa 3: el filtro por metadatos es una **rama etiquetada**, no un escalón de la escala |
| páginas sin sección con id | 0 | — (la estructura resultó regular) |

La etiqueta `fuzzy` —la traducción que el proyecto mismo marca como dudosa— **no existe en
esta ruta**. Lo que sobrevive es el síntoma: la entry sin traducir sale como párrafo en
inglés, y eso sí se mide. Es el costo que la entrada 2 del log acepta explícitamente.

### Casi-duplicados

Dos métricas, dos propósitos distintos (confundirlas fue un error medido; ver el docstring
de `vocabulary()`):

- **Entre versiones** (near-duplicación, MinHash de 5-gramas): 375 páginas tienen gemela en
  3.12, Jaccard mediano **0,86**, y 227 por encima de 0,8. **La página de 3.12 no es una
  copia sucia: es otra versión del mismo documento.** Consecuencia para la Etapa 2: cuando
  la respuesta no depende de la versión el gold lista **las dos** páginas; cuando sí
  depende, la pregunta tiene que decir de qué versión habla.
- **Dentro de una versión** (confusabilidad, contención exacta de vocabulario): el par más
  confundible es `asynchat ~ asyncore` con **0,975** de contención y 0,672 de Jaccard de
  5-gramas. No son duplicados; comparten casi todo el vocabulario. Eso es lo que hace que un
  recuperador traiga el vecino equivocado.

**La política: versiones, no copias.** No se deduplica nada. Las dos páginas llevan
`canonical_id` = el `doc_id` de 3.13, `duplicate_of = null` y `versions_of` con las ramas en que
existe. La Etapa 1 reverificó la medición de la Etapa 0 con **otro método** (Jaccard exacto sobre
una muestra de 20, sin MinHash): mediana **0,859**, idéntica a la del MinHash. 335 de 360 rutas
tienen gemela; **25 páginas existen solo en 3.12 y ninguna solo en 3.13.**

**Las gemelas que no se parecen son las interesantes.** La menos parecida de la muestra es
`library/dialog.html` con 0,346, y el caso extremo del corpus es `library/aifc.html`: en 3.12 es
la documentación completa del módulo, en 3.13 es un aviso de 100 tokens que dice que se eliminó.
Mismo `rel`, misma `canonical_id`, respuestas opuestas. Son las páginas cuyo gold **no puede** ser
un conjunto de dos versiones: ahí la pregunta tiene que nombrar la versión.

### Metadatos que mienten: la verificación

`version` tiene 100 % de cobertura y eso es precisamente lo sospechoso — no lo leímos de la
página, lo pusimos según de qué zip salió. La Etapa 3 va a filtrar por él, así que se contrasta
contra el cuerpo con una prueba de un solo lado: **una página de 3.12 no puede hablar de algo que
llegó en 3.13**. Sobre los 476 documentos con alguna nota de versión, **0 desacuerdos (0,00 %)**.

El primer intento de esta prueba dio 18 % de desacuerdo y era un bug nuestro: comparaba la
versión completa, así que `library/os.path` de 3.13 quedaba marcado por decir "3.13.4" —una nota
de parche de su propia rama—. La etiqueta de versión es una **rama**, así que la comparación usa
solo (mayor, menor).

## 5 · Qué pone a prueba bien el corpus

| pone a prueba bien | de dónde sale |
|---|---|
| **preguntas de identificador** | 10.759 anclas de API, todas dentro de una sección con id |
| **preguntas temporales** ("¿desde qué versión…?") | 4.385 notas de versión; 66 % de las páginas declara algo sobre su propio estado |
| **abstención, en los dos sentidos** | 1.211 anclas existen solo en 3.13 y 467 solo en 3.12; más los 24 avisos de eliminación |
| **veredicto sí/no** | "¿puedo usar X en 3.12?" se contesta con un booleano, así que el ítem necesita `expected_verdict`, no solo una clave de respuesta |
| **recuperación cross-lingual** | 251 páginas con más de 10 % de párrafos en inglés |
| **distinguir vecinos confundibles** | contención de vocabulario hasta 0,975 dentro de 3.13 |
| **chunking por estructura** | 2.996 secciones, 100 % con id **y** título; 2.415 hojas, p50 355 y p90 1.304 tokens |

### Lo que **no** pone a prueba

- **Nada que dependa de la fecha del documento**: no hay fechas. La Etapa 6 usa timestamps
  simulados, que es lo que `schema.Session.date` espera de todas formas.
- **Los builtins en su versión vigente**: están, pero solo por la puerta de 3.12 (ver §3). Una
  pregunta sobre `print` se contesta con una página `superseded`, y bajo filtro de versión a 3.13
  no se contesta.
- **`tkinter` y `token`**: fuera del corpus por inglés, sin red de 3.12.
- **Conflictos reales entre fuentes**: la documentación oficial no se contradice consigo
  misma. Lo más cercano es 3.12 vs 3.13, que es versionado, no desacuerdo.

### Por qué la recuperación no es opcional

| | |
|---|---|
| tokens del generador en el subconjunto (3.13) | 1.884.998 |
| presupuesto de prompt (`num_ctx` 8192 − `num_predict` 512) | **7.680** |
| ventanas completas que haría falta | **245** |
| páginas que **no entran solas** en una ventana | 75 de 385 |
| largo de página: p50 / p90 / máx | 2.403 / 12.974 / 48.297 tokens |
| largo de sección hoja: p50 / p90 / máx | 355 / 1.304 / 17.972 tokens |
| hojas sobre el tope de padre (1.500 tokens) | 8,1 % |

Las hojas tienen el largo adecuado para ser hijos, pero el 8,1 % que se pasa del tope
obliga a un **fallback por largo** en la Etapa 3: cortar solo por estructura deja padres que
no entran en el presupuesto.

---

## Pendiente (una persona, no una tabla)

1. **Leer diez documentos crudos de principio a fin.** Están volcados en texto plano en
   `artifacts/full/stage0_read_me/`, elegidos por diez extremos distintos (la más larga, la
   más corta, la más en inglés, la de más anclas, la de estructura más profunda, la de hoja
   más grande, la vecina confundible, …). Lo que haya que anotar va en la sección 4 de esta
   corpus card: el punto del ejercicio es encontrar lo que ninguna de las tablas de arriba
   nombró.
2. **Corregir el chequeo a libro cerrado.** Las cinco preguntas están escritas en la última
   celda de `00b_corpus_audit_ours.py`; necesitan un Ollama con `qwen3:8b`, así que corren en
   la máquina de escritorio (3050) o en Colab. Marcar cada respuesta correcta / incorrecta /
   vaga contra las anclas medidas y anotar el conteo. `corpora.md` avisa que para este corpus
   el libro cerrado va a salir **alto**: eso cambia qué preguntas vale la pena escribir.
