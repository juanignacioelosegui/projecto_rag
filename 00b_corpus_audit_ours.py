# %% [markdown]
r"""
# Etapa 0b · El corpus como decisión: nuestra parte

`00_corpus_audit` audita python-docs-es por las dos rutas de entrada y **no elige ninguna**:
esa elección es nuestra, y es lo que hace esta notebook. Arriba de eso, el §3 de la Etapa 0 pide
siete cosas que la auditoría resuelta deja afuera, y acá se miden una por una:

| Lo que falta | Dónde está acá |
|---|---|
| cantidad de documentos | *Documentos y largos* |
| largos **en tokens del generador** (la resuelta solo estima el total) | *Documentos y largos* |
| marcadores de estructura, y qué tan regulares son | *Estructura* |
| si el id sobrevive a una nueva descarga | *¿El id sobrevive a una nueva descarga?* |
| cobertura de metadatos (fecha / versión / estado) | *Metadatos* |
| casi-duplicados sobre una muestra (la resuelta cuenta solapamiento exacto de párrafos) | *Vecinos confundibles* |
| diez documentos crudos leídos por una persona | *Diez documentos para leer a mano* |

La extracción vive en `corpus_html.py`, no en una celda, porque la Etapa 1 tiene que leer las
mismas páginas de la misma forma. Si la auditoría mide una extracción y el índice se construye
sobre otra, estos números no describen el corpus que terminamos indexando.
"""

# %%
import environment  # noqa: F401  (efectos secundarios: guarda de JAX, caché de HF, bundle TLS, UTF-8)

environment.summary()

import collections
import json
import os
import re
import statistics
import zlib

import numpy as np

import corpus_html
import store
from console import StageRun, clip, print_table, verdict

SMOKE_TEST = os.environ.get("SMOKE_TEST", "1").lower() not in ("0", "false", "no")
SEED = 1337
ART = environment.artifacts_dir(SMOKE_TEST)
PROFILE = "smoke" if SMOKE_TEST else "full"
GENERATOR = "qwen3:1.7b" if SMOKE_TEST else "qwen3:8b"
NUM_CTX = 8192           # la ventana que nos podemos permitir (ver decision_log.md)
NUM_PREDICT = 512        # la reserva de salida: el prompt entra en NUM_CTX - NUM_PREDICT
VERSIONS = ("3.13", "3.12")

# En smoke solo estas páginas, elegidas para que cada defecto aparezca al menos una vez:
# una página muy traducida (pathlib), una sin traducir (builtins/functions), una con
# identificadores literales (itertools), una vecina confundible de otra (os.path vs pathlib).
SMOKE_PAGES = {"library/pathlib.html", "library/json.html", "library/itertools.html",
               "library/os.path.html", "library/functions.html", "builtins/functions.html",
               "library/pickle.html", "library/shutil.html", "faq/general.html",
               "tutorial/introduction.html"}

run = StageRun.start("00b_corpus_audit_ours", ART, profile=PROFILE,
                     config={"seed": SEED, "generator": GENERATOR, "num_ctx": NUM_CTX})

# %% [markdown]
r"""
## La decisión: la ruta del HTML compilado

La auditoría resuelta cierra diciendo: *"La ruta `.po` tiene las marcas de calidad de los propios
traductores y un historial de git; la ruta HTML tiene firmas, notas de versión e ids estables, y
esconde la división fuzzy/sin traducir detrás de texto en inglés. Tomes la que tomes, nombrá lo
que te cuesta más adelante."*

**Vamos por el HTML compilado.** El argumento no es que sea más limpio -- no lo es -- sino que lo
que el `.po` pierde, ninguna etapa posterior lo puede recuperar, y lo que el HTML pierde, se
puede medir.

Tres cosas que el `.po` no tiene y que nuestro plan necesita:

1. **Un identificador de documento y de sección.** Un `.po` es una lista plana de pares; lo único
   que ubica una entrada es *(archivo, línea del `.rst` en inglés)*. Esa línea se mueve con
   cualquier edición upstream, así que no es un id estable; y la auditoría resuelta midió que
   **2.209 entradas están compartidas por más de una ubicación** (gettext fusiona textos
   idénticos), así que tampoco es único. Sin un id de sección no hay etiquetas gold de sección
   en la Etapa 2, ni citas verificables en la Etapa 4. El HTML trae los dos ids puestos por la
   fuente: `<section id>` y `<dt id>`.
2. **Las firmas de la API.** Son el material de las preguntas de identificador, que es
   exactamente donde el canal denso solo falla y el léxico gana (Etapa 3). La celda de abajo
   vuelve a verificar que no están en el `.po`.
3. **Los números de versión.** `.. versionadded:: 3.13` vive en el `.rst` en inglés: el `.po`
   traduce el *cuerpo* de la directiva, no su argumento. Sin esos números no hay preguntas
   temporales, ni filtro de versión, ni el gancho de memoria ("estoy en 3.12") que la Etapa 6
   necesita.

**Lo que esta elección nos cuesta, y dónde lo pagamos:**

- **Perdemos la etiqueta `fuzzy`.** El traductor marca qué traducciones sospecha
  desactualizadas; Sphinx no las usa, así que la página compilada muestra el inglés. Perdemos el
  *motivo* pero conservamos el *síntoma*, y es medible: un párrafo en inglés dentro de una página
  en español es, casi siempre, una entrada fuzzy o sin traducir. Lo pagamos en la Etapa 3: una
  consulta en español contra un pasaje en inglés es recuperación cross-lingual. `bge-m3` es
  multilingüe, así que es sobrevivible, pero es una fuente de error que no habríamos tenido.
- **Perdemos el pin por commit.** El zip se recompila en el mismo lugar, a veces a diario. El
  `.po` se fija con un commit de git que cualquiera puede reproducir; el zip solo se fija con el
  sha256 de *nuestra* descarga. Y ya está roto: los dos sha256 que la auditoría resuelta fijó el
  2026-09-27 **ya no coinciden**. Lo pagamos en la Etapa 1, declarando la fecha de descarga y
  nuestro propio hash en el manifiesto y en `corpus_card.md`, y poniendo `on_mismatch="raise"`
  de ahí en adelante: después de fijarlo, un hash distinto tiene que parar la corrida, no
  avisar.
"""

# %%
import polib

po_root = environment.corpora_dir() / "python-docs-es-3.13"
if not po_root.exists():
    raise RuntimeError("falta el repo .po: corré primero 00_corpus_audit.py, que lo clona "
                       "fijado por commit")

po_pathlib = polib.pofile(str(po_root / "library" / "pathlib.po"))
po_text = "\n".join(e.msgid + "\n" + e.msgstr for e in po_pathlib)
po_has_signature = "resolve(strict" in po_text
po_has_version = bool(re.search(r"\.\. versionadded|\bversionadded::", po_text))
# Lo único que ubica una entrada .po: (archivo, línea) del .rst en inglés.
po_locations = [e.occurrences for e in po_pathlib]
po_multi = sum(len(occ) > 1 for occ in po_locations)

zips = {}
for v in VERSIONS:
    path, digest = corpus_html.fetch_archive(v, on_mismatch="warn")
    zips[v] = {"path": path, "sha256": digest}
    print(f"   {v}: {path.name}  sha256 {digest[:16]}…")

print()
print("pathlib.po tiene la firma `resolve(strict=...)`:        ", po_has_signature)
print("pathlib.po tiene una directiva de versión:              ", po_has_version)
print(f"entradas de pathlib.po con más de una ubicación fuente:  {po_multi} de {len(po_pathlib)}")
print(verdict(not (po_has_signature or po_has_version),
              "confirmado: la ruta .po no tiene firmas ni números de versión -> vamos por el HTML",
              "revisar: el .po SÍ tiene firmas o versiones, el argumento de la ruta se cae"))

# %% [markdown]
r"""
## ¿El id sobrevive a una nueva descarga?

Es la pregunta 4 de las ocho, y la auditoría resuelta dice explícitamente que **no** la mide
(*"NOT measured here: compare two downloads"*). Es la que decide si una etiqueta gold escrita hoy
sigue apuntando a algo la semana que viene. La medimos de las dos formas que tenemos:

- **Descarga contra descarga.** Bajamos el zip de 3.13 otra vez, ahora, a otro archivo, y
  comparamos. Si el sha256 cambió, comparamos los tres conjuntos de ids que una etiqueta gold
  puede nombrar: páginas, secciones y anclas de la API.
- **Versión contra versión.** 3.13 contra 3.12 es el peor caso de una "nueva descarga": no es
  una recompilación, es la siguiente versión. Pone un techo a cuánto puede moverse un id.
"""

# %%
again_path, again_digest = store.fetch_url(
    corpus_html.ARCHIVE_URL.format(v="3.13"),
    environment.corpora_dir() / "python-3.13-docs-html-es.recheck.zip",
    sha256=None)
same_bytes = again_digest == zips["3.13"]["sha256"]
print(f"descarga 1: {zips['3.13']['sha256'][:16]}…")
print(f"descarga 2: {again_digest[:16]}…")
print(verdict(same_bytes, "byte por byte idéntico en esta ventana de tiempo",
              "el zip se recompiló entre nuestras dos descargas"))

redownload = {"same_bytes": same_bytes, "sha256_1": zips["3.13"]["sha256"], "sha256_2": again_digest}
if not same_bytes:
    a = corpus_html.ids_of("3.13", zip_path=zips["3.13"]["path"])
    b = corpus_html.ids_of("3.13", zip_path=again_path)
    rows = []
    for kind in ("pages", "sections", "anchors"):
        kept = len(a[kind] & b[kind])
        rows.append((kind, {"descarga 1": len(a[kind]), "descarga 2": len(b[kind]),
                            "sobreviven": kept,
                            "% sobreviven": round(100 * kept / max(len(a[kind]), 1), 2)}))
        redownload[kind] = {"before": len(a[kind]), "after": len(b[kind]), "kept": kept}
    print_table("ids que sobreviven a una recompilación del zip",
                ["descarga 1", "descarga 2", "sobreviven", "% sobreviven"], rows, smoke=SMOKE_TEST)
again_path.unlink(missing_ok=True)   # 52 MB que no hacen falta una vez medido

# %% [markdown]
r"""
## La pasada por página

Una sola lectura de los dos zips produce, por página: el texto, las secciones con sus offsets,
las anclas de la API, las notas de versión, las marcas de traducción por párrafo, y **el largo
real en tokens del generador** -- no una estimación a partir de caracteres, que es lo único que
da la auditoría resuelta. Tokenizamos con el tokenizer de `qwen3:8b` (no hace falta el servidor
de Ollama, solo el tokenizer de HuggingFace), porque cada modelo divide el mismo texto distinto
y el único largo que importa es el del modelo que lo va a leer.

El resultado se cachea en disco: la pasada cuesta minutos y todas las celdas de abajo la reusan.
"""

# %%
import llm

NPERM = 128
MERSENNE = (1 << 31) - 1
_rng = np.random.default_rng(SEED)
_A = _rng.integers(1, MERSENNE, NPERM, dtype=np.int64)
_B = _rng.integers(0, MERSENNE, NPERM, dtype=np.int64)


def _signature(shingles: set[int]) -> list[int]:
    if not shingles:
        return [MERSENNE] * NPERM
    h = np.fromiter(shingles, dtype=np.int64, count=len(shingles))
    return ((_A[:, None] * h[None, :] + _B[:, None]) % MERSENNE).min(axis=1).tolist()


def shingle_signature(text: str, k: int = 5) -> list[int]:
    """Firma MinHash sobre shingles de k palabras, para detectar casi-duplicados.

    Es la medida correcta para "¿la página de 3.12 es una copia de la de 3.13?", donde el
    Jaccard verdadero es alto (0,8-0,95) y 128 permutaciones lo estiman con sobra.

    `zlib.crc32` y no `hash()`: el hash de Python está salado por proceso y daría una firma
    distinta en cada corrida.
    """
    words = re.findall(r"\w+", text.lower())
    return _signature({zlib.crc32(" ".join(words[i:i + k]).encode()) % MERSENNE
                       for i in range(max(len(words) - k + 1, 1))})


def vocabulary(text: str) -> set[str]:
    """El vocabulario de la página. La contención se calcula EXACTA sobre estos conjuntos, no
    con MinHash: con 128 permutaciones la resolución del Jaccard es 1/128, y para un par de
    páginas de tamaños muy distintos el Jaccard verdadero es chiquito, así que la intersección
    derivada de él tiene más error que señal (nuestro primer intento daba contenciones de 3,06,
    que es imposible). Exacto cuesta un bitset por página y vale la pena."""
    return set(re.findall(r"\w+", text.lower()))


def scan(_dir) -> list[dict]:
    rows, vocabs = [], []
    only = SMOKE_PAGES if SMOKE_TEST else None
    for v in VERSIONS:
        for page in corpus_html.iter_pages(v, zip_path=zips[v]["path"], only=only):
            tokens = llm.count_tokens(page.text, GENERATOR) or 0
            per_char = tokens / max(len(page.text), 1)
            leaves = [s for s in page.sections
                      if not any(o["parent_id"] == s["section_id"] for o in page.sections)]
            vocab = vocabulary(page.text)
            vocabs.append(vocab)
            rows.append({
                "version": v, "rel": page.rel, "page_id": page.page_id, "top": page.top,
                "title": page.title, "url": page.url,
                "characters": len(page.text), "tokens": tokens,
                "sections": len(page.sections),
                "section_ids_real": sum(not s["section_id"].startswith("_") for s in page.sections),
                "titled_sections": sum(bool(s["title"]) for s in page.sections),
                "max_depth": max((s["level"] for s in page.sections), default=0),
                "leaf_tokens": sorted(int((s["end"] - s["start"]) * per_char) for s in leaves),
                "anchors": len(page.anchors),
                "anchors_in_section": sum(bool(a["section_id"]) for a in page.anchors),
                # Los ids, no solo su conteo: la deriva de ids entre versiones se calcula abajo
                # sin volver a parsear 100 MB de zip.
                "anchor_ids": [a["anchor_id"] for a in page.anchors],
                "section_ids": [s["section_id"] for s in page.sections],
                "notes": len(page.notes),
                "note_kinds": sorted({n["kind"] for n in page.notes if n["kind"]}),
                "note_versions": sorted({n["version"] for n in page.notes if n["version"]}),
                "note_labels": sorted({n["label"] for n in page.notes if n["label"]}),
                "english_label_notes": sum(n["label"][:1].isascii() and n["label"][:5] in
                                           ("Added", "Chang", "Depre", "Soft ") for n in page.notes),
                "paragraphs": len(page.paragraphs),
                "english_ratio": round(page.english_ratio, 4),
                "unmarked_ratio": round(page.unmarked_ratio, 4),
                "pilcrows": page.text.count("¶"),
                "minhash_grams": shingle_signature(page.text),
                "vocab_size": len(vocab),
            })
            if len(rows) % 50 == 0:
                print(f"   {len(rows)} páginas…", flush=True)
    store.write_jsonl(_dir / "pages.jsonl", rows)

    # Los vocabularios, como un bitset por página sobre un índice de términos compartido. Va a
    # un .npy aparte y no a pages.jsonl: son millones de enteros y no se leen a mano.
    terms = sorted(set().union(*vocabs)) if vocabs else []
    index = {t: i for i, t in enumerate(terms)}
    bits = np.zeros((len(vocabs), (len(terms) + 7) // 8), dtype=np.uint8)
    for r, vocab in enumerate(vocabs):
        row = np.zeros(len(terms), dtype=bool)
        row[[index[t] for t in vocab]] = True
        bits[r] = np.packbits(row)
    np.save(_dir / "vocab_bits.npy", bits)
    print(f"   vocabulario: {len(terms):,} términos, bitset {bits.nbytes / 1e6:.1f} MB")
    return rows


spec = {"stage": "00b", "profile": PROFILE, "versions": VERSIONS, "subset": corpus_html.SUBSET,
        "generator": GENERATOR, "nperm": NPERM, "seed": SEED,
        # Subir esto cuando cambien los campos que `scan` escribe: el hash del parser no lo ve,
        # y una caché vieja con campos faltantes rompe las celdas de abajo.
        "scan_fields": 4,
        "smoke_pages": sorted(SMOKE_PAGES) if SMOKE_TEST else None,
        "zips": {v: zips[v]["sha256"] for v in VERSIONS},
        # El hash del parser entra en la spec: si cambia la extracción, estos números dejan de
        # describir el corpus y la caché tiene que reconstruirse sola.
        "parser": store.sha256_file(corpus_html.__file__)}
scan_dir = store.build_or_load(ART / "stage0_pages", spec, scan)
pages = store.read_jsonl(scan_dir / "pages.jsonl")
vocab_bits = np.load(scan_dir / "vocab_bits.npy")
rows13 = [i for i, p in enumerate(pages) if p["version"] == "3.13"]
P13 = [pages[i] for i in rows13]
P12 = [p for p in pages if p["version"] == "3.12"]
print(f"\npáginas parseadas: {len(P13)} en 3.13, {len(P12)} en 3.12")

# %% [markdown]
r"""
## Documentos y largos, en tokens del generador

Dos decisiones viven en esta tabla.

**Qué es un documento.** Una página compilada (`library/pathlib.html`) -- no un módulo, no una
sección. Es el grano que tiene id estable, URL, título y metadatos propios, y es el grano al que
apuntan las citas. El costo: una página de `library/` es enorme comparada con la ventana, así que
el documento **nunca** es la unidad de entrega; se entrega por chunk y se cita por documento.

**Si hace falta recuperación.** La línea de base honesta de cualquier RAG es *si entra, pegalo*.
La reserva de salida (`num_predict = 512`) sale de la ventana, así que el presupuesto real de
prompt es `num_ctx - num_predict`.
"""

# %%
def describe(values: list[int]) -> dict:
    """`method="inclusive"`: el default ("exclusive") extrapola más allá del rango de los datos
    y con pocas páginas devuelve un p90 mayor que el máximo (`builtins`, n=5, lo mostró)."""
    v = sorted(values)
    q = statistics.quantiles(v, n=100, method="inclusive") if len(v) > 1 else [v[0]] * 99
    return {"n": len(v), "min": v[0], "p50": int(q[49]), "p90": int(q[89]),
            "p99": int(q[98]), "max": v[-1], "total": sum(v)}


rows = []
for top in sorted({p["top"] for p in P13}):
    sub = [p for p in P13 if p["top"] == top]
    d = describe([p["tokens"] for p in sub])
    rows.append((top, {"páginas": d["n"], "tokens p50": d["p50"], "tokens p90": d["p90"],
                       "tokens max": d["max"], "tokens total": d["total"]}))
all13 = describe([p["tokens"] for p in P13])
rows.append(("TODO 3.13", {"páginas": all13["n"], "tokens p50": all13["p50"],
                           "tokens p90": all13["p90"], "tokens max": all13["max"],
                           "tokens total": all13["total"]}))
print_table("largo de las páginas en tokens de " + GENERATOR,
            ["páginas", "tokens p50", "tokens p90", "tokens max", "tokens total"],
            rows, smoke=SMOKE_TEST, name_header="directorio")

budget = NUM_CTX - NUM_PREDICT
print(f"\npresupuesto de prompt: {NUM_CTX} - {NUM_PREDICT} = {budget:,} tokens")
print(f"el corpus 3.13 entero:  {all13['total']:,} tokens = {all13['total'] / budget:,.0f} prompts llenos")
print(f"páginas que NO entran solas en el presupuesto: "
      f"{sum(p['tokens'] > budget for p in P13)} de {len(P13)}")
print(verdict(all13["total"] > 10 * budget,
              "el corpus es órdenes de magnitud más grande que la ventana: la recuperación no es opcional",
              "el corpus está cerca de una ventana: 'pegá todo' es un competidor real y hay que medirlo"))

# %% [markdown]
r"""
## Estructura, y qué tan regular es

"Tiene estructura" no alcanza: lo que decide si se puede cortar por secciones es **qué tan
regular** es esa estructura. Tres cosas que rompen el corte, y que hay que contar antes de
confiar en él:

- páginas **sin ninguna `<section>`**: el cuerpo entero es un solo bloque sin id, así que sus
  chunks no pueden citar una sección;
- secciones **sin id propio** (las marcamos `_anon…` / `_body`): no sirven como etiqueta gold;
- **secciones hoja demasiado grandes**: si la hoja no entra en el techo del padre, el padre no es
  una unidad de entrega y hay que cortar dentro de la sección.

El techo de padre que vamos a usar en la Etapa 3 es 1.500 tokens del generador (§7.1).
"""

# %%
PARENT_CAP = 1500
leaf_tokens = [t for p in P13 for t in p["leaf_tokens"]]
no_sections = [p for p in P13 if p["section_ids_real"] == 0]
leaf = describe(leaf_tokens) if leaf_tokens else {}

rows = [
    ("páginas con al menos una sección con id", {"valor": len(P13) - len(no_sections),
                                                 "% de páginas": round(100 * (len(P13) - len(no_sections)) / len(P13), 1)}),
    ("secciones, total", {"valor": sum(p["sections"] for p in P13), "% de páginas": "—"}),
    ("secciones con id propio", {"valor": sum(p["section_ids_real"] for p in P13),
                                 "% de páginas": round(100 * sum(p["section_ids_real"] for p in P13) / max(sum(p["sections"] for p in P13), 1), 1)}),
    ("secciones con título", {"valor": sum(p["titled_sections"] for p in P13),
                              "% de páginas": round(100 * sum(p["titled_sections"] for p in P13) / max(sum(p["sections"] for p in P13), 1), 1)}),
    ("anclas de la API", {"valor": sum(p["anchors"] for p in P13), "% de páginas": "—"}),
    ("anclas ubicadas en una sección con id", {"valor": sum(p["anchors_in_section"] for p in P13),
                                               "% de páginas": round(100 * sum(p["anchors_in_section"] for p in P13) / max(sum(p["anchors"] for p in P13), 1), 1)}),
    (f"secciones hoja sobre el techo de {PARENT_CAP} tokens",
     {"valor": sum(t > PARENT_CAP for t in leaf_tokens),
      "% de páginas": round(100 * sum(t > PARENT_CAP for t in leaf_tokens) / max(len(leaf_tokens), 1), 1)}),
]
print_table("regularidad de la estructura (3.13)", ["valor", "% de páginas"], rows,
            smoke=SMOKE_TEST, name_header="marcador")

print(f"\nprofundidad de anidamiento de secciones: "
      f"{collections.Counter(p['max_depth'] for p in P13).most_common()}")
if leaf:
    print(f"secciones hoja en tokens: p50 {leaf['p50']}, p90 {leaf['p90']}, max {leaf['max']}")
print("páginas sin ninguna sección con id:", [p["rel"] for p in no_sections][:10] or "ninguna")
print(verdict(len(no_sections) / len(P13) < 0.05 and
              sum(t > PARENT_CAP for t in leaf_tokens) / max(len(leaf_tokens), 1) < 0.25,
              "la estructura es regular: cortar por sección es defendible",
              "la estructura es irregular: el corte por sección necesita un fallback por largo"))

# %% [markdown]
r"""
## Metadatos: qué hay, y qué tan lejos llega

La pregunta 5 pide la proporción de documentos con **una fecha, una versión y un estado
utilizables**. Hay que ser preciso sobre qué cuenta como "utilizable", porque es fácil inflar
esta fila:

- **versión**: la del archivo (3.13 / 3.12). Está en el 100 % de los documentos *por
  construcción*, no porque el documento la declare. Como filtro sirve igual, pero no distingue
  nada **dentro** de una versión, y por eso el filtro de metadatos de la Etapa 3 va como rama
  etiquetada y no como fila de la escalera: solo tiene sentido contra la otra versión.
- **fecha**: **no hay ninguna.** La página compilada no lleva fecha de publicación. Lo único
  parecido es la fecha de *nuestra descarga*. Esto es un "no" de los siete, y lo pagamos en la
  Etapa 6: las sondas temporales no pueden apoyarse en fechas del documento, así que los
  escenarios de memoria usan timestamps **simulados** (que es lo que `schema.Session.date`
  espera de todas formas).
- **estado**: lo único que el documento declara de sí mismo son sus notas de versión. Una página
  con `deprecated` dice algo sobre su propio estado; una sin notas no dice nada.

La fila honesta es la de las notas de versión, porque es la única que varía documento a documento.
"""

# %%
with_notes = [p for p in P13 if p["notes"]]
with_dep = [p for p in P13
            if {"deprecated", "deprecated-removed"} & set(p["note_kinds"])]
mixed_labels = [p for p in P13 if len({("es" if not l[:5] in ("Added", "Chang", "Depre", "Soft ") else "en")
                                       for l in p["note_labels"]}) > 1]
n = len(P13)
rows = [
    ("versión (del archivo)", {"documentos": n, "cobertura %": 100.0,
                               "comentario": "por construcción, no la declara el documento"}),
    ("fecha", {"documentos": 0, "cobertura %": 0.0,
               "comentario": "la página compilada no lleva fecha; la Etapa 6 usa fechas simuladas"}),
    ("estado (nota de versión)", {"documentos": len(with_notes),
                                  "cobertura %": round(100 * len(with_notes) / n, 1),
                                  "comentario": "lo único que varía documento a documento"}),
    ("estado (obsolescencia explícita)", {"documentos": len(with_dep),
                                          "cobertura %": round(100 * len(with_dep) / n, 1),
                                          "comentario": "candidatos a status='deprecated'"}),
    ("notas con etiqueta en los dos idiomas", {"documentos": len(mixed_labels),
                                               "cobertura %": round(100 * len(mixed_labels) / n, 1),
                                               "comentario": "'Added in version' junto a 'Distinto en la versión'"}),
]
print_table("cobertura de metadatos (3.13)", ["documentos", "cobertura %", "comentario"], rows,
            smoke=SMOKE_TEST, name_header="metadato")

all_note_versions = collections.Counter(v for p in P13 for v in p["note_versions"])
print("\nversiones nombradas por las notas, las 12 más frecuentes:",
      all_note_versions.most_common(12))
print(verdict(len(with_notes) / n > 0.5,
              "la mayoría de los documentos declara algo sobre su propio estado",
              "la mayoría de los documentos no declara nada: el único metadato real es la versión del archivo"))

# %% [markdown]
r"""
## Cuánto está en inglés, y la regla del subconjunto

Acá está el techo del corpus, y es el número que más decisiones condiciona. Un párrafo sin
traducir o fuzzy aparece en inglés en la página compilada. La auditoría resuelta lo midió por
directorio; lo que decide la **regla del subconjunto** es la distribución **por página**: un
promedio de directorio del 25 % puede ser 300 páginas al 10 % o 100 páginas al 75 %, y son dos
corpus distintos.

El primer borrador de la regla era de una sola línea: **descartar las páginas con más del 60 % de
párrafos en inglés.** No es un umbral de limpieza, es uno de honestidad -- una página al 98 % en
inglés no es un documento en español, y dejarla adentro convierte cualquier pregunta sobre ella
en recuperación cross-lingual disfrazada.

Medirla antes de adoptarla la rompió, y de dos formas que valen la pena:

- **se llevaba justo el mejor material de abstención.** `library/aifc.html`, `library/asynchat.html`
  y compañía son páginas de módulos que 3.13 **eliminó**: su cuerpo entero es un aviso de 100
  tokens, en inglés, que dice "este módulo ya no es parte de la biblioteca estándar, se eliminó en
  3.12, se había declarado obsoleto en 3.6, ver PEP 594". Es inglés, sí, pero es **exactamente la
  respuesta correcta** a "¿puedo usar asynchat en 3.13?", que es la pregunta de abstención que la
  Etapa 2 necesita y la Etapa 6 usa con el gancho de versión. Y para contestarla no hay que leer
  prosa en español: hay que leer una nota de obsolescencia, que es estructura, no idioma.
- **no veía las páginas vacías.** `library/concurrent.html` son 31 tokens ("en este paquete solo
  existe un módulo: concurrent.futures") y `howto/clinic.html` son 36 ("esta documentación se
  trasladó a la Guía de Desarrolladores"). Están en español, así que el umbral de idioma las deja
  pasar, pero no contienen ninguna respuesta: son punteros. En el índice son ruido con un título
  muy buscable.

La regla final clasifica cada página en una de tres clases, **en este orden** -- y el orden es
la mitad de la regla. Nuestro primer intento ponía el umbral de largo antes de la excepción de
obsolescencia, y hacía las dos cosas mal a la vez: descartaba `library/aifc.html` por tener menos
de 100 tokens (siendo un aviso de eliminación, justo lo que queríamos conservar) y rescataba
`builtins/functions.html` -- 18.000 tokens al 98 % en inglés -- por llevar *alguna* nota de
obsolescencia suelta entre sus funciones.

1. **aviso de eliminación**: lleva una nota `deprecated` / `deprecated-removed` **y** mide menos
   de 500 tokens. Se queda siempre, con `status="deprecated"`, sin importar el idioma: para
   contestar "¿puedo usar asynchat en 3.13?" no hay que leer prosa en español, hay que leer una
   nota de obsolescencia, que es estructura.
2. **puntero**: menos de 100 tokens y sin nota de obsolescencia. Se descarta: no contiene
   ninguna respuesta, y tiene un título muy buscable.
3. **documentación**: todo lo demás. Se queda si tiene a lo sumo **60 % de párrafos en inglés**.
"""

# %%
# La regla vive en `corpus_html`, no acá: la Etapa 1 la aplica y esta celda mide lo que cuesta.
# Con los umbrales escritos en las dos notebooks, esta tabla describiría otro corpus.
ENGLISH_CAP = corpus_html.ENGLISH_CAP
MIN_TOKENS = corpus_html.MIN_TOKENS
STUB_MAX_TOKENS = corpus_html.STUB_MAX_TOKENS
DEPRECATION_KINDS = corpus_html.DEPRECATION_KINDS


def page_class(p: dict) -> str:
    return corpus_html.page_class(p["tokens"], p["note_kinds"])


buckets = [(0.0, 0.10), (0.10, 0.25), (0.25, 0.40), (0.40, 0.60), (0.60, 0.90), (0.90, 1.01)]
rows = []
for lo, hi in buckets:
    sub = [p for p in P13 if lo <= p["english_ratio"] < hi]
    rows.append((f"{lo:.0%}-{hi if hi <= 1 else 1:.0%} en inglés",
                 {"páginas": len(sub), "tokens": sum(p["tokens"] for p in sub),
                  "anclas API": sum(p["anchors"] for p in sub),
                  "avisos de eliminación": sum(page_class(p) == "aviso de eliminación" for p in sub)}))
print_table("distribución del inglés por página (3.13)",
            ["páginas", "tokens", "anclas API", "avisos de eliminación"],
            rows, smoke=SMOKE_TEST, name_header="franja")

classes = collections.defaultdict(list)
for p in P13:
    classes[page_class(p)].append(p)
notices, pointers, docs = (classes["aviso de eliminación"], classes["puntero"],
                           classes["documentación"])
docs_english = [p for p in docs if p["english_ratio"] > ENGLISH_CAP]
kept = notices + [p for p in docs if p["english_ratio"] <= ENGLISH_CAP]
dropped = pointers + docs_english


def agg(label, sub, effect):
    return (label, {"páginas": len(sub), "tokens": sum(p["tokens"] for p in sub),
                    "anclas API": sum(p["anchors"] for p in sub), "efecto": effect})


print_table("qué cuesta cada cláusula de la regla del subconjunto",
            ["páginas", "tokens", "anclas API", "efecto"],
            [agg("1 · aviso de eliminación", notices, "se queda, status=deprecated"),
             agg("2 · puntero", pointers, "descartado"),
             agg("3 · documentación en español", [p for p in docs if p["english_ratio"] <= ENGLISH_CAP],
                 "se queda"),
             agg("3 · documentación en inglés", docs_english, "descartada"),
             agg("TOTAL descartado", dropped, f"queda {len(kept)} de {len(P13)}")],
            smoke=SMOKE_TEST, name_header="clase")

print(f"\ncorpus final: {len(kept)} documentos, {sum(p['tokens'] for p in kept):,} tokens, "
      f"{sum(p['anchors'] for p in kept):,} de {sum(p['anchors'] for p in P13):,} anclas de API "
      f"({100 * sum(p['anchors'] for p in kept) / sum(p['anchors'] for p in P13):.1f} %)")
print("avisos de eliminación que se quedan (material de abstención):",
      sorted(p["rel"].split("/")[-1].replace(".html", "") for p in notices))
print("\npunteros descartados:", sorted(p["rel"] for p in pointers)[:8])
print("\nlas páginas de documentación más caras que la regla descarta:",
      [f"{p['rel']} ({p['english_ratio']:.0%}, {p['anchors']} anclas)"
       for p in sorted(docs_english, key=lambda p: -p["anchors"])[:5]])
print("\nEl costo con nombre propio: `builtins/` entero -- la referencia de las funciones "
      "integradas, que es la página más central de toda la documentación -- está sin traducir "
      "y la regla la descarta. Es el techo del corpus, no un defecto de la regla.")
print("CORRECCIÓN de la Etapa 1: esta celda mira solo 3.13, y de ahí habíamos concluido que "
      "'ninguna pregunta puede ser sobre print o len'. Es falso. 3.13 movió esas páginas a "
      "`builtins/` dejando un stub en `library/`, y 3.12 todavía las tiene en `library/` "
      "traducidas: se recuperan 398 anclas por la puerta de 3.12. El costo real es más fino y "
      "más interesante -- el gold de esas preguntas es una sola página con status=superseded, y "
      "un filtro de versión a 3.13 las vuelve incontestables. Ver 01_documents y corpus_card.md.")
print("páginas que Sphinx nunca mandó a traducir (100 % sin marcar):",
      sum(p["unmarked_ratio"] > 0.99 for p in P13))
print(verdict(sum(p["anchors"] for p in dropped) / max(sum(p["anchors"] for p in P13), 1) < 0.15,
              "la regla cuesta pocas anclas: el material de las preguntas de identificador sobrevive",
              "la regla se lleva una parte grande de las anclas: hay que elegir entre idioma y "
              "material de evaluación, y decirlo en el informe"))

# %% [markdown]
r"""
## Vecinos confundibles

La octava pregunta, y la única que no es una propiedad del corpus sino un hecho sobre él: *¿cuán
parecidos son los documentos entre sí?* Los vecinos confundibles son lo que hace difícil la
recuperación -- y lo que hace que una métrica a nivel de documento mienta.

La auditoría resuelta cuenta el solapamiento **exacto de párrafos entre versiones** (85 %).
Eso deja dos huecos, y los dos importan:

- **casi-duplicados, no idénticos.** Un párrafo con una palabra cambiada no cuenta como idéntico
  y es igual de confundible. MinHash sobre shingles de 5 palabras (128 permutaciones, semilla
  fija) estima el Jaccard por página.
- **vecinos dentro de una misma versión.** Son los peores, porque no hay ningún metadato que los
  separe: `os.path` y `pathlib` documentan la misma tarea, y ninguna etiqueta de versión ayuda a
  elegir. Para estos el Jaccard de 5-gramas es la medida **equivocada**: dos páginas que explican
  lo mismo con otras palabras no comparten casi ningún 5-grama. Lo que las hace confundibles es
  compartir vocabulario, así que acá va **contención de vocabulario**
  (`|A ∩ B| / min(|A|, |B|)`) -- y no Jaccard, porque las páginas tienen tamaños muy distintos y
  el Jaccard penaliza a la grande por ser grande.
"""

# %%
def jaccard_matrix(rows_: list[dict], field: str) -> np.ndarray:
    sig = np.array([r[field] for r in rows_], dtype=np.int64)
    return (sig[:, None, :] == sig[None, :, :]).mean(axis=2)


# --- vecinos confundibles dentro de 3.13: contención EXACTA de vocabulario --------------
jac_grams = jaccard_matrix(P13, "minhash_grams")
bits13 = vocab_bits[rows13]
sizes = np.bitwise_count(bits13).sum(axis=1, dtype=np.int64)   # popcount = |vocabulario|
inter = np.zeros((len(P13), len(P13)), dtype=np.int64)
for start in range(0, len(P13), 8):                            # en bloques: la matriz entera no entra
    block = bits13[start:start + 8]
    inter[start:start + 8] = np.bitwise_count(
        block[:, None, :] & bits13[None, :, :]).sum(axis=2, dtype=np.int64)
containment = inter / np.maximum(np.minimum(sizes[:, None], sizes[None, :]), 1)
np.fill_diagonal(containment, 0.0)
assert containment.max() <= 1.0 + 1e-9, "la contención no puede pasar de 1: hay un bug"
iu = np.triu_indices(len(P13), k=1)
pairs = sorted(zip(containment[iu], *iu), reverse=True)[:12]
rows = [(f"{P13[i]['rel']}  ~  {P13[j]['rel']}",
         {"contención vocab": round(float(c), 3),
          "Jaccard 5-gramas": round(float(jac_grams[i, j]), 3)})
        for c, i, j in pairs]
print_table("los pares más confundibles DENTRO de 3.13",
            ["contención vocab", "Jaccard 5-gramas"], rows, smoke=SMOKE_TEST, name_header="par")
print("\nLas dos columnas no coinciden, y eso es el punto: comparten vocabulario (se confunden "
      "al recuperar) sin ser copias (el 5-grama no las ve).")

# --- casi-duplicados entre versiones: Jaccard de 5-gramas -------------------------------
by_rel12 = {p["rel"]: p for p in P12}
twins = [(p, by_rel12[p["rel"]]) for p in P13 if p["rel"] in by_rel12]
tw = np.array([(np.array(a["minhash_grams"]) == np.array(b["minhash_grams"])).mean()
               for a, b in twins]) if twins else np.array([])
if twins:
    print(f"\npáginas presentes en 3.13 y 3.12: {len(twins)}")
    print(f"Jaccard de 5-gramas, 3.13 vs su copia en 3.12: p50 {np.median(tw):.3f}, "
          f"p10 {np.percentile(tw, 10):.3f}, p90 {np.percentile(tw, 90):.3f}")
    print(f"páginas con Jaccard > 0.8 (la copia es casi el mismo documento): "
          f"{int((tw > 0.8).sum())} de {len(twins)}")
    print(f"\nCHECK -- con {int((tw > 0.8).sum())} páginas casi idénticas entre versiones, un "
          "hit@k a nivel de documento que trate la copia de 3.12 como fallo castiga recuperación "
          "correcta. Dos consecuencias para la Etapa 2: la etiqueta gold tiene que listar las DOS "
          "páginas cuando la respuesta no depende de la versión, y la pregunta tiene que decir de "
          "qué versión habla cuando sí depende.")

# --- deriva de ids entre versiones: el techo de "¿sobrevive el id?" ---------------------
A13 = {a for p in P13 for a in p["anchor_ids"]}
A12 = {a for p in P12 for a in p["anchor_ids"]}
anchor_drift = {"only_3_13": len(A13 - A12), "only_3_12": len(A12 - A13),
                "shared": len(A13 & A12)}
print(f"\nanclas de API solo en 3.13: {anchor_drift['only_3_13']:,}  "
      f"(p. ej. {sorted(A13 - A12)[:3]})")
print(f"anclas de API solo en 3.12: {anchor_drift['only_3_12']:,}  "
      f"(p. ej. {sorted(A12 - A13)[:3]})")
print(f"anclas compartidas: {anchor_drift['shared']:,}")
print("Las que están solo en 3.13 son material de abstención para 3.12, y las que están solo en "
      "3.12 son material de abstención para 3.13: la Etapa 2 las usa en los dos sentidos.")

# %% [markdown]
r"""
## La tarea de datos sucios, dimensionada

`corpora.md` describe los defectos de este corpus en prosa. El paso 4 de la Etapa 0 pide su
tamaño: cuántos documentos toca cada defecto, para decidir cuáles arregla la Etapa 1 y cuáles se
declaran y se dejan.
"""

# %%
stubs = corpus_html.redirect_stubs("3.13", zip_path=zips["3.13"]["path"])
rows = [
    ("stubs de redirección (cuerpo vacío)",
     {"documentos": len(stubs), "qué hace la Etapa 1": "los saltea; el parser ya devuelve None"}),
    ("marcas ¶ de ancla pegadas al texto",
     {"documentos": sum(p["pilcrows"] > 0 for p in P13),
      "qué hace la Etapa 1": f"las borra ({sum(p['pilcrows'] for p in P13):,} ocurrencias)"}),
    ("páginas con párrafos en inglés (>10 %)",
     {"documentos": sum(p["english_ratio"] > 0.10 for p in P13),
      "qué hace la Etapa 1": "los deja y los marca; no se traduce nada"}),
    ("páginas que la regla del subconjunto descarta",
     {"documentos": len(dropped),
      "qué hace la Etapa 1": f"{len(docs_english)} por inglés, {len(pointers)} punteros"}),
    ("etiquetas de nota de versión en inglés",
     {"documentos": sum(p["english_label_notes"] > 0 for p in P13),
      "qué hace la Etapa 1": "normaliza la etiqueta, conserva el número"}),
    ("páginas sin sección con id",
     {"documentos": len(no_sections), "qué hace la Etapa 1": "una sola sección '_body'; sin gold de sección"}),
]
print_table("defectos conocidos, y quién los paga", ["documentos", "qué hace la Etapa 1"], rows,
            smoke=SMOKE_TEST, name_header="defecto")
print("\nstubs:", stubs[:10] or "ninguno")

# %% [markdown]
r"""
## Diez documentos para leer a mano

El último ítem del paso 2 -- *"diez documentos crudos leídos por una persona, de principio a
fin"* -- es el único que no se puede automatizar, y es el que encuentra los defectos que ninguna
métrica de arriba nombra. La celda **no** elige diez al azar: elige diez que cubran los extremos
que las tablas anteriores encontraron, los escribe a disco en texto plano, y deja la lectura
para una persona. Las notas de esa lectura van al informe y a `corpus_card.md`.
"""

# %%
def pick(rows_, key, reverse=True, skip=()):
    for p in sorted(rows_, key=key, reverse=reverse):
        if p["rel"] not in skip:
            return p
    return None


criteria = [
    ("la más larga", lambda p: p["tokens"], True),
    ("la más corta", lambda p: p["tokens"], False),
    ("la más en inglés", lambda p: p["english_ratio"], True),
    ("la más traducida y grande", lambda p: (p["english_ratio"] < 0.05, p["tokens"]), True),
    ("la de más anclas de API", lambda p: p["anchors"], True),
    ("la de más notas de versión", lambda p: p["notes"], True),
    ("la de estructura más profunda", lambda p: p["max_depth"], True),
    ("la de sección hoja más grande", lambda p: max(p["leaf_tokens"] or [0]), True),
    ("sin sección con id", lambda p: (p["section_ids_real"] == 0, -p["tokens"]), True),
    ("vecina confundible del par más parecido", lambda p: p["rel"] == P13[pairs[0][1]]["rel"], True),
]
read_dir = ART / "stage0_read_me"
read_dir.mkdir(parents=True, exist_ok=True)
chosen, rows = [], []
only = SMOKE_PAGES if SMOKE_TEST else None
texts = {p.rel: p for p in corpus_html.iter_pages("3.13", zip_path=zips["3.13"]["path"], only=only)}
for label, key, rev in criteria:
    p = pick(P13, key, rev, skip={c["rel"] for c in chosen})
    if p is None:
        continue
    chosen.append(p)
    page = texts.get(p["rel"])
    if page is not None:
        (read_dir / (p["rel"].replace("/", "__") + ".txt")).write_text(
            f"# {page.title}\n# {page.url}\n# {p['tokens']:,} tokens · "
            f"{p['english_ratio']:.0%} inglés · {p['sections']} secciones · "
            f"{p['anchors']} anclas\n\n{page.text}\n", encoding="utf-8")
    rows.append((label, {"página": p["rel"], "tokens": p["tokens"],
                         "inglés %": round(100 * p["english_ratio"], 1),
                         "secciones": p["sections"], "anclas": p["anchors"]}))
print_table("los diez documentos crudos, elegidos por extremo",
            ["página", "tokens", "inglés %", "secciones", "anclas"], rows,
            smoke=SMOKE_TEST, name_header="por qué este")
print(f"\nescritos en texto plano en: {read_dir}")
print("PENDIENTE (una persona): leerlos de principio a fin y anotar en corpus_card.md lo que "
      "ninguna tabla de arriba nombró.")

# %% [markdown]
r"""
## El chequeo a libro cerrado

El tercer número del paso 4. La documentación de Python está en la web pública, así que es casi
seguro que el generador la leyó: nuestro RAG no aporta conocimiento que al modelo le falte, sino
**la versión correcta** de ese conocimiento, **con una cita**, y la capacidad de decir "no está
documentado". Este chequeo mide qué tan alta está esa vara.

Las cinco preguntas están ancladas en hechos que medimos arriba, y cada una tiene una forma de
fallar distinta: una API que solo existe en 3.13, una que no existe en absoluto, una que existe
desde una versión intermedia, un parámetro concreto, y un módulo que 3.12 tiene y 3.13 ya no.

Sin servidor de Ollama la celda se saltea: por el reparto híbrido (`decision_log.md`) esto corre
en la máquina de escritorio o en Colab, no en la laptop donde se escribió el código.
"""

# %%
CLOSED_BOOK = [
    "¿En qué versión de Python se agregó el método PurePath.full_match de pathlib?",
    "¿Existe Path.copy en pathlib en Python 3.13?",
    "¿Desde qué versión existe itertools.batched?",
    "¿Qué parámetro de Path.resolve hace que falle si la ruta no existe?",
    "¿Todavía se puede importar el módulo aifc en Python 3.13?",
]
closed_book = []
if llm.server_alive() and llm.model_digest(GENERATOR):
    print(llm.placement(GENERATOR))
    for q in CLOSED_BOOK:
        r = llm.ask(q, system="Respondé en una o dos oraciones.", model=GENERATOR,
                    num_ctx=NUM_CTX, num_predict=160, cache=False)
        closed_book.append({"question": q, "answer": r.text, "error": r.error})
        print(f"Q: {q}\nA: {clip(r.text, 300)}\n")
    print("PENDIENTE (una persona): marcar cada respuesta como correcta / incorrecta / vaga "
          "contra las anclas medidas arriba, y anotar el conteo en el informe.")
else:
    print(f"(salteado: no hay servidor de Ollama con {GENERATOR} -- "
          f"instalá Ollama y corré `ollama pull {GENERATOR}`)")

# %% [markdown]
r"""
## La auditoría, en una tabla

Las siete propiedades y la similitud, con lo medido y **qué etapa posterior paga por cada "no"**.
Esta es la tabla que el paso 3 pide, y es el esqueleto de la sección de corpus del informe.
"""

# %%
PINNED_BY_WORKED_AUDIT = {   # los sha256 que 00_corpus_audit fijó el 2026-09-27
    "3.13": "73566c509e4d26acadf79410423e801775268e097a36efcc4c7b583e5c376ef7",
    "3.12": "0b4ec3162616218de068c2c53c9faca8b18fec6eefa3bcb3157d0b1265c03bd3",
}
still_pinned = [v for v in VERSIONS if zips[v]["sha256"] == PINNED_BY_WORKED_AUDIT[v]]

audit_rows = [
    ("1 ¿está la respuesta?",
     {"medido": f"{len(P13)} páginas, {all13['total']:,} tokens; "
                f"{sum(p['english_ratio'] > 0.10 for p in P13)} páginas con >10 % de párrafos en inglés",
      "qué etapa paga": "Etapa 3: consulta en español vs pasaje en inglés = cross-lingual"}),
    ("2 ¿está limpio?",
     {"medido": f"{sum(p['pilcrows'] for p in P13):,} marcas ¶, {len(stubs)} stubs de redirección, "
                f"etiquetas de versión en dos idiomas",
      "qué etapa paga": "Etapa 1: cada regla de limpieza, con el antes/después medido"}),
    ("3 ¿hay estructura para cortar?",
     {"medido": f"{sum(p['sections'] for p in P13):,} secciones "
                f"({sum(p['section_ids_real'] for p in P13):,} con id), "
                f"{sum(p['anchors'] for p in P13):,} anclas de API; "
                f"{len(no_sections)} páginas sin sección",
      "qué etapa paga": "Etapa 3: las hojas grandes necesitan fallback por largo"}),
    ("4 ¿el id sobrevive?",
     {"medido": ("idéntico en dos descargas consecutivas" if redownload["same_bytes"]
                 else "el zip se recompiló entre nuestras dos descargas: ver tabla de ids") +
                f"; contra 3.12 se mueven {anchor_drift['only_3_13'] + anchor_drift['only_3_12']:,} anclas",
      "qué etapa paga": "Etapa 2: el gold se congela contra NUESTRO sha256, no contra la URL"}),
    ("5 ¿hay metadatos?",
     {"medido": f"versión 100 % (por construcción), fecha 0 %, "
                f"nota de versión {100 * len(with_notes) / len(P13):.0f} %",
      "qué etapa paga": "Etapa 6: sin fechas del documento, los timestamps son simulados"}),
    ("6 ¿cambia?",
     {"medido": "el zip se recompila en el mismo lugar; de los sha256 fijados el 2026-09-27 "
                + (f"siguen coincidiendo {still_pinned}" if still_pinned else "ya no coincide ninguno"),
      "qué etapa paga": "Etapa 1: pin por sha256 propio + fecha, y on_mismatch='raise' después"}),
    ("7 ¿se puede evaluar?",
     {"medido": f"{sum(p['anchors'] for p in P13):,} anclas y "
                f"{sum(p['notes'] for p in P13):,} notas de versión: identificador y temporales salen solas",
      "qué etapa paga": "Etapa 2: salen DEMASIADO solas -- fuga léxica; chequeo BM25 obligatorio"}),
    ("8 la similitud",
     {"medido": (f"Jaccard mediano {np.median(tw):.2f} entre una página de 3.13 y su copia en 3.12"
                 if twins else "—") +
                f"; contención de vocabulario del par más confundible dentro de 3.13: {pairs[0][0]:.2f}",
      "qué etapa paga": "Etapa 3: el hit@k por documento no ve el efecto; gold con las dos versiones"}),
]
print_table("python-docs-es: la auditoría, nuestra parte",
            ["medido", "qué etapa paga"], audit_rows, smoke=SMOKE_TEST)

# %%
summary = {
    "profile": PROFILE,
    "route": "html",
    "generator": GENERATOR,
    "num_ctx": NUM_CTX,
    "num_predict": NUM_PREDICT,
    "zips": {v: zips[v]["sha256"] for v in VERSIONS},
    "redownload": redownload,
    "subset_rule": {"directories": list(corpus_html.SUBSET), "english_cap": ENGLISH_CAP,
                    "min_tokens": MIN_TOKENS, "stub_max_tokens": STUB_MAX_TOKENS,
                    "deprecation_kinds": sorted(DEPRECATION_KINDS),
                    "pages_dropped": len(dropped), "pages_kept": len(kept),
                    "kept_as_notice": [p["rel"] for p in notices],
                    "dropped_as_pointer": [p["rel"] for p in pointers],
                    "dropped_as_english": [p["rel"] for p in docs_english],
                    "tokens_kept": sum(p["tokens"] for p in kept),
                    "tokens_dropped": sum(p["tokens"] for p in dropped),
                    "anchors_kept": sum(p["anchors"] for p in kept),
                    "anchors_dropped": sum(p["anchors"] for p in dropped)},
    "pages": {"3.13": len(P13), "3.12": len(P12)},
    "tokens_3_13": all13,
    "structure": {"sections": sum(p["sections"] for p in P13),
                  "sections_with_id": sum(p["section_ids_real"] for p in P13),
                  "anchors": sum(p["anchors"] for p in P13),
                  "pages_without_section": [p["rel"] for p in no_sections],
                  "leaf_tokens": leaf,
                  "leaves_over_cap": sum(t > PARENT_CAP for t in leaf_tokens)},
    "metadata": {"with_notes": len(with_notes), "with_deprecation": len(with_dep),
                 "mixed_label_pages": len(mixed_labels), "with_date": 0},
    "similarity": {"twin_pages": len(twins),
                   "twin_jaccard_p50": float(np.median(tw)) if twins else None,
                   "twin_jaccard_over_0_8": int((tw > 0.8).sum()) if twins else 0,
                   "anchor_drift_3_13_vs_3_12": anchor_drift,
                   "top_confusable_in_3_13": [[P13[i]["rel"], P13[j]["rel"], round(float(c), 3)]
                                              for c, i, j in pairs]},
    "dirty": {"redirect_stubs": stubs, "pilcrows": sum(p["pilcrows"] for p in P13)},
    "read_by_hand": [c["rel"] for c in chosen],
    "closed_book": closed_book,
}
store.write_json(ART / "stage0_ours.json", summary)
print(f"escrito: {ART / 'stage0_ours.json'}")

run.note("pages_3_13", len(P13))
run.note("tokens_3_13", all13["total"])
run.note("pages_dropped_by_subset_rule", len(dropped))
run.finish()
