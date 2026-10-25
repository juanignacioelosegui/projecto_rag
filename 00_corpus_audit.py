# %% [markdown]
r"""
# Etapa 0 · El corpus como decisión

**Este notebook es el ejemplo resuelto.** Audita uno de los corpus del menú --
**python-docs-es**, la traducción al español de la documentación de Python -- de la forma en
que la Etapa 0 te pide que audites el tuyo. Nada de esto construye el pipeline; mide qué *es*
el corpus antes de que decidas qué hacer con él.

Por qué molestarse: haga lo que haga tu pipeline de recuperación, no puede recuperar algo que
los documentos no contienen. El corpus es el techo. Así que antes de elegir tamaños de chunk o
rerankers, respondé las preguntas que responde esta auditoría -- con números, no con impresiones:

1. **¿Está la respuesta ahí?** (¿y en qué idioma?)
2. **¿El texto está limpio?** ¿Qué dejaría pasar un loader ingenuo sin ningún error?
3. **¿Tiene estructura** por la que se pueda cortar?
4. **¿Cada documento tiene un identificador estable** -- uno que sobreviva a una nueva descarga?
5. **¿Hay metadatos** -- una fecha, una versión, un estado -- y son correctos?
6. **¿Cambia,** y cómo te enterarías?
7. **¿Se puede armar un conjunto de evaluación** sobre él?
8. Y una que no es una propiedad sino un hecho: **¿cuán parecidos son los documentos entre
   sí?** Los vecinos confundibles son lo que hace difícil la recuperación.

Si elegiste python-docs-es, esta auditoría es **donde tu Etapa 0 empieza, no donde termina**:
deja abierta a propósito la decisión que más importa -- por qué ruta entrás al corpus -- y
varios de sus números merecen una segunda mirada.
"""

# %%
import environment  # noqa: F401  (efectos secundarios: guarda de JAX, caché de HF, bundle TLS, UTF-8)

environment.summary()

from console import StageRun, clip, print_table, verdict

import os

# True: unas pocas páginas, minutos en una laptop. False: el subconjunto entero.
# Lo leemos del entorno porque el reparto híbrido (ver decision_log.md) corre esta misma etapa
# en dos máquinas: `SMOKE_TEST=0 python 00_corpus_audit.py` en la de escritorio, el default acá.
SMOKE_TEST = os.environ.get("SMOKE_TEST", "1").lower() not in ("0", "false", "no")
SEED = 1337
ART = environment.artifacts_dir(SMOKE_TEST)
PROFILE = "smoke" if SMOKE_TEST else "full"
run = StageRun.start("00_corpus_audit", ART, profile=PROFILE, config={"seed": SEED})

# %% [markdown]
r"""
## Dos rutas hacia el mismo corpus

La documentación de Python en español existe en dos formas, y *no* son el mismo corpus:

- **El repositorio de traducción** (`python/python-docs-es`): un archivo `.po` por página, cada
  uno una lista de pares *(original en inglés, traducción al español)*. Esta es la fuente que
  editan los traductores. Está versionado en git, una rama por versión de Python.
- **La documentación compilada** (`docs.python.org/es/<version>/`): el HTML que produce la
  traducción una vez que Sphinx la combina con las fuentes en inglés. Se puede descargar como un
  zip por versión.

Fijamos las dos. El repositorio por **commit**: el nombre de una rama se mueve, un commit no. El
archivo compilado no tiene commit -- se recompila en el mismo lugar, a menudo a diario -- así que
fijamos su **sha256 y la fecha en que lo descargamos**, y una discrepancia se reporta en lugar de
esconderse. Esa diferencia ya es en sí una respuesta a la pregunta 6.
"""

# %%
import store

PO_REPO = "https://github.com/python/python-docs-es"
PO_COMMITS = {"3.13": "507e3375a87dcd76d657493632e354957818a503"}  # rama 3.13, 2026-02-25
ARCHIVE_URL = "https://docs.python.org/es/{v}/archives/python-{v}-docs-html.zip"
# sha256 de los archivos tal como se descargaron el 2026-09-27. Se recompilan en el mismo lugar:
# un hash distinto significa que el corpus cambió bajo la misma URL, no que la descarga falló.
ARCHIVE_SHA256 = {
    "3.13": "73566c509e4d26acadf79410423e801775268e097a36efcc4c7b583e5c376ef7",
    "3.12": "0b4ec3162616218de068c2c53c9faca8b18fec6eefa3bcb3157d0b1265c03bd3",
}

corpora = environment.corpora_dir()
po_root = corpora / "python-docs-es-3.13"
store.fetch_git(PO_REPO, po_root, commit=PO_COMMITS["3.13"], branch="3.13")

archives = {}
for v in ("3.13", "3.12"):
    path, digest = store.fetch_url(ARCHIVE_URL.format(v=v), corpora / f"python-{v}-docs-html-es.zip",
                                   sha256=ARCHIVE_SHA256[v], on_mismatch="warn")
    archives[v] = {"path": path, "sha256": digest, "pinned": digest == ARCHIVE_SHA256[v]}
    print(f"   {v}: {verdict(archives[v]['pinned'], 'matches the pinned snapshot', 'NOT the pinned snapshot: the archive was rebuilt since')}")

# %% [markdown]
r"""
## Ruta A · los archivos de traducción

Una entrada `.po` es un par: `msgid` (el párrafo fuente en inglés) y `msgstr` (su traducción).
Importan tres estados:

- **traducida** (*translated*) -- `msgstr` está completo;
- **sin traducir** (*untranslated*) -- `msgstr` está vacío, y la documentación compilada muestra
  el inglés;
- **fuzzy** -- *existe* una traducción, pero está marcada como posiblemente desactualizada,
  porque el inglés cambió después de que se escribió. Sphinx no usa las entradas fuzzy (la
  configuración de compilación del proyecto no cambia ese valor por defecto), así que la página
  compilada muestra el inglés para ellas. Un loader que lee `msgstr` de forma ingenua las usa
  igual.
"""

# %%
import collections

import polib

po_files = sorted(p for p in po_root.rglob("*.po") if ".git" not in p.parts)
if SMOKE_TEST:
    po_files = [p for p in po_files if p.parts[len(po_root.parts)] in ("tutorial",)] + \
               [po_root / "library" / f"{m}.po" for m in ("pathlib", "json", "itertools", "os.path")]

by_dir = collections.defaultdict(collections.Counter)
merged, fuzzy_examples = 0, []
for path in po_files:
    rel = path.relative_to(po_root)
    top = rel.parts[0] if len(rel.parts) > 1 else "(root)"
    for entry in polib.pofile(str(path)):
        if entry.obsolete:
            continue
        merged += len(entry.occurrences) > 1
        if "fuzzy" in entry.flags:
            state = "fuzzy"
            if top == "library" and 20 < len(entry.msgid) < 90 and not entry.msgid.startswith(":"):
                fuzzy_examples.append((rel.as_posix(), entry.msgid, entry.msgstr))
        else:
            state = "translated" if entry.msgstr.strip() else "untranslated"
        by_dir[top][state] += 1

rows = []
for top, c in sorted(by_dir.items(), key=lambda kv: -sum(kv[1].values())):
    n = sum(c.values())
    rows.append((top, {"entries": n, "translated %": 100 * c["translated"] / n,
                       "fuzzy %": 100 * c["fuzzy"] / n, "untranslated %": 100 * c["untranslated"] / n}))
print_table(f".po entries by directory ({len(po_files)} files)",
            ["entries", "translated %", "fuzzy %", "untranslated %"], rows, smoke=SMOKE_TEST)
print(f"\nentries shared by more than one source location (gettext merges identical text): {merged}")
print("\nfuzzy entries -- translations the project itself flags as possibly wrong:")
for rel, en, es in fuzzy_examples[:: max(1, len(fuzzy_examples) // 4)][:4]:
    print(f"   {rel}\n     EN: {clip(en, 80)}\n     ES: {clip(es, 80)}")

# %% [markdown]
r"""
Lo que los archivos de traducción **no** tienen es igual de importante. La firma de la API al
principio de la entrada de cada función (`Path.resolve(strict=False)`) y las directivas de
versión (`.. versionadded:: 3.4`) viven en las fuentes `.rst` en inglés, no en el `.po`: solo el
*cuerpo* de una directiva es traducible. Verificalo en lugar de creerlo:
"""

# %%
import re

pathlib_po = "\n".join(e.msgid + "\n" + e.msgstr for e in polib.pofile(str(po_root / "library" / "pathlib.po")))
has_signature = "resolve(strict" in pathlib_po
has_versions = bool(re.search(r"\b3\.1[0-3]\b", pathlib_po))
print("pathlib.po contains the signature `resolve(strict=...)`:", has_signature)
print("pathlib.po contains a version number such as 3.12:     ", has_versions)
print(verdict(has_signature and has_versions,
              "signatures and versions are recoverable from the .po alone",
              "the .po route loses API signatures and version numbers -- recovering them means "
              "joining each entry back to the English .rst at the matching commit"))
print("\ndirectories with translations for pages Python no longer ships:",
      [d for d in ("distutils", "install") if d in by_dir] or "none in this sample")

# %% [markdown]
r"""
## Ruta B · la documentación compilada

El HTML compilado tiene lo que le falta al `.po`: cada objeto de la API es un `<dt>` con un `id`
estable (`pathlib.Path.resolve`) seguido de su firma; cada encabezado es una `<section>` con un
`id`; cada nota de versión es un elemento con una clase (`versionadded`, `versionchanged`,
`deprecated`). Esos ids son los candidatos obvios para **etiquetas gold** y **citas**: son
estables, legibles para una persona, y cada uno se abre como una URL.

También tiene lo que el `.po` esconde: en cada párrafo sin traducir o fuzzy, la página muestra
el **original en inglés**. Dos formas de medir cuánto:

- las propias marcas de la compilación: Sphinx etiqueta cada párrafo traducible como `translated`
  o `untranslated`. Un párrafo **sin ninguna** de las dos marcas nunca pasó por la traducción;
- un termómetro deliberadamente tosco: por párrafo, contar palabras funcionales comunes del
  español vs. del inglés y decidir por mayoría. No es un detector de idioma -- revisá unos
  cuantos párrafos a ojo antes de confiar en él -- pero no depende de cómo se compiló la página.

Cuando los dos no coinciden, averiguá por qué.
"""

# %%
import zipfile

from bs4 import BeautifulSoup

ES_WORDS = set("de la que el en y los se del las un por con no una su para es al lo como más pero sus le "
               "si sin sobre este ya entre cuando todo esta ser son también puede todos así nos ni tiene "
               "donde mismo ese ahora cada otro otros esa eso hace otra ella".split())
EN_WORDS = set("the of and to in is that for it as with was on be by this are or an at from which not "
               "have has can if but when you will its their they should use returns return used also "
               "may all one these more than only such other into each".split())
SUBSET = ("library", "builtins", "tutorial", "howto", "reference", "faq")  # candidatos para indexar
SMOKE_PAGES = {"library/pathlib.html", "library/json.html", "library/itertools.html",
               "library/os.path.html", "builtins/functions.html", "library/functions.html"}


def paragraph_language(text: str) -> str | None:
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    es, en = sum(w in ES_WORDS for w in words), sum(w in EN_WORDS for w in words)
    return None if es + en < 4 else ("en" if en > es else "es")


def audit_archive(version: str) -> dict:
    zf = zipfile.ZipFile(archives[version]["path"])
    prefix = f"python-{version}-docs-html/"
    per_dir = collections.defaultdict(collections.Counter)
    anchors, labels, texts = set(), collections.Counter(), {}
    for name in zf.namelist():
        if not name.startswith(prefix):
            continue
        rel = name[len(prefix):]
        top = rel.split("/")[0] if "/" in rel else "(root)"
        if not name.endswith(".html") or top not in SUBSET + ("whatsnew",):
            continue
        if SMOKE_TEST and rel not in SMOKE_PAGES and top != "tutorial":
            continue
        soup = BeautifulSoup(zf.read(name), "lxml")
        if soup.find("meta", attrs={"http-equiv": "refresh"}) is not None:
            per_dir[top]["redirect stubs"] += 1  # una página que solo redirige a otra
            continue
        body = soup.select_one("div.body")
        if body is None:
            continue
        c = per_dir[top]
        c["pages"] += 1
        c["sections"] += len(body.find_all("section"))
        page_anchors = [dt["id"] for dt in body.select("dt[id]")]
        c["api anchors"] += len(page_anchors)
        anchors.update(page_anchors)
        for note in body.select(".versionadded, .versionchanged, .deprecated, .deprecated-removed"):
            head = note.get_text(" ", strip=True)
            m = re.match(r"(Added in version|Changed in version|Deprecated since version|Nuevo en la versión"
                         r"|Distinto en la versión|Obsoleto desde la versión)", head)
            labels[m.group(1) if m else clip(head, 28)] += 1
        p_nodes = body.find_all("p")
        for p in p_nodes:
            classes = p.get("class") or []
            c["marked translated" if "translated" in classes else
              "marked untranslated" if "untranslated" in classes else "unmarked"] += 1
        paragraphs = [p.get_text(" ", strip=True) for p in p_nodes]
        langs = [paragraph_language(p) for p in paragraphs]
        c["paragraphs"] += sum(l is not None for l in langs)
        c["english paragraphs"] += sum(l == "en" for l in langs)
        text = body.get_text(" ", strip=True)
        c["characters"] += len(text)
        texts[rel] = paragraphs
    return {"per_dir": per_dir, "anchors": anchors, "labels": labels, "texts": texts}


audit = {v: audit_archive(v) for v in ("3.13", "3.12")}
for v, a in audit.items():
    if not any(c["pages"] for c in a["per_dir"].values()):
        raise RuntimeError(f"{v}: no page matched under 'python-{v}-docs-html/' -- the archive layout "
                           "changed, and every number below would be zero")
    rows = []
    for top, c in sorted(a["per_dir"].items()):
        marked = c["marked translated"] + c["marked untranslated"] + c["unmarked"]
        rows.append((top, {"pages": c["pages"], "redirects": c["redirect stubs"], "sections": c["sections"],
                           "api anchors": c["api anchors"], "characters": c["characters"],
                           "unmarked %": 100 * c["unmarked"] / max(marked, 1),
                           "English % (words)": 100 * c["english paragraphs"] / max(c["paragraphs"], 1)}))
    print_table(f"built docs {v} (es)", ["pages", "redirects", "sections", "api anchors", "characters",
                                         "unmarked %", "English % (words)"], rows, smoke=SMOKE_TEST)
    print("   version-note labels:", dict(a["labels"].most_common()))

# %% [markdown]
r"""
Dos cosas para leer en esas tablas antes de seguir.

**Las etiquetas.** "Added in version" y "Distinto en la versión" son el *mismo tipo de nota* en
dos idiomas, según si esa cadena de etiqueta se tradujo o no. Una pregunta en español --
"¿desde qué versión existe…?" -- no comparte ninguna palabra con una etiqueta en inglés. Decidí
ahora si tus documentos conservan la etiqueta tal cual, la normalizan, o pasan la versión a los
metadatos.

**La proporción en inglés.** No es uniforme. Fijate cuál es el peor directorio, qué contiene, y
qué dicen de él las propias marcas de la compilación -- después compará las mismas páginas en
las dos versiones. Cualquier cosa que un usuario hispanohablante pregunte sobre esas páginas se
va a responder, si es que se responde, a partir de texto en inglés.

**Las redirecciones.** Algunas páginas existen solo para mandar al navegador a otro lado. Un
loader que indexa todos los archivos `.html` las indexa como documentos vacíos.
"""

# %% [markdown]
r"""
## Versiones: el mismo documento, dos veces

El corpus tiene una copia de cada página *por versión de Python*, y las versiones consecutivas
son mayormente idénticas. Esa es la propiedad 8 hecha concreta: el vecino más confundible de un
pasaje es el mismo pasaje en la versión anterior. Medí cuánto difieren realmente.
"""

# %%
new_in_313 = sorted(audit["3.13"]["anchors"] - audit["3.12"]["anchors"])
gone_in_313 = sorted(audit["3.12"]["anchors"] - audit["3.13"]["anchors"])
shared_pages = sorted(set(audit["3.13"]["texts"]) & set(audit["3.12"]["texts"]))
same = total = 0
for rel in shared_pages:
    old = set(audit["3.12"]["texts"][rel])
    for p in audit["3.13"]["texts"][rel]:
        total += 1
        same += p in old
print(f"API anchors only in 3.13 : {len(new_in_313)}   e.g. {new_in_313[:4]}")
print(f"API anchors only in 3.12 : {len(gone_in_313)}   e.g. {gone_in_313[:4]}")
print(f"pages present in both    : {len(shared_pages)}")
print(f"paragraphs of 3.13 identical in 3.12: {same}/{total} = {100 * same / max(total, 1):.0f}%")
print(verdict(same / max(total, 1) < 0.5,
              "versions differ enough to be told apart by content",
              "most paragraphs are word-for-word identical across versions -- a document-level "
              "metric that treats the 3.12 copy as a miss will punish correct retrieval"))

# %% [markdown]
r"""
## ¿Hace falta siquiera la recuperación?

La línea de base honesta para cualquier sistema de RAG es: *si entra, pegalo*. Entonces: ¿qué
tan grande es este corpus en los tokens del modelo que lo va a leer -- no en caracteres, no en
palabras, y no en los tokens del embedder (cada modelo divide el mismo texto de forma distinta)?
"""

# %%
import llm

GENERATOR = "qwen3:1.7b" if SMOKE_TEST else "qwen3:8b"
NUM_CTX = 8192
total_chars = sum(c["characters"] for top, c in audit["3.13"]["per_dir"].items() if top in SUBSET)
sample = " ".join(p for paras in list(audit["3.13"]["texts"].values())[:40] for p in paras)[:400_000]
tokens_per_char = llm.count_tokens(sample, GENERATOR) / max(len(sample), 1)
est_tokens = int(total_chars * tokens_per_char)
print(f"subset {SUBSET} of 3.13: {total_chars:,} characters")
print(f"generator tokens per character (measured on a {len(sample):,}-char sample): {tokens_per_char:.3f}")
print(f"estimated size: ~{est_tokens:,} generator tokens = {est_tokens / NUM_CTX:,.0f} windows of {NUM_CTX}")
print(verdict(est_tokens > 10 * NUM_CTX, "far larger than one window: retrieval is not optional",
              "close to one window: 'paste it all' is a real competitor -- measure it"))

# %% [markdown]
r"""
## ¿El generador ya lo leyó?

La documentación de Python está en la web pública, y por lo tanto, casi con seguridad, en los
datos de entrenamiento del generador. Eso cambia *para qué* sirve acá un sistema de RAG: no para
aportar conocimiento que al modelo le falta, sino para aportar la **versión correcta** de ese
conocimiento, con una **cita**, y para decir "no está documentado" cuando no lo está. Un chequeo
puntual a libro cerrado -- el modelo sin ninguna recuperación -- te dice qué tan alta está esa
vara. Leé las respuestas contra las anclas medidas arriba: `PurePath.full_match` existe solo
desde 3.13; `Path.copy` no existe en absoluto en 3.13.
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
if environment.on_colab():
    llm.setup(GENERATOR, num_ctx=NUM_CTX)  # instala, levanta el servidor, descarga (~5 GB) y prueba el modelo
if llm.server_alive() and llm.model_digest(GENERATOR):
    for q in CLOSED_BOOK:
        r = llm.ask(q, system="Respondé en una o dos oraciones.", model=GENERATOR, num_ctx=NUM_CTX,
                    num_predict=160, cache=False)
        closed_book.append({"question": q, "answer": r.text, "error": r.error})
        print(f"Q: {q}\nA: {clip(r.text, 300)}\n")
else:
    print(f"(skipped: no Ollama server with {GENERATOR} pulled -- locally: install Ollama and "
          f"`ollama pull {GENERATOR}`)")

# %% [markdown]
r"""
## La auditoría, en una tabla

Las ocho preguntas del principio, respondidas para este corpus con los números de arriba. Tu
Etapa 0 completa la misma tabla para el tuyo -- y agrega una columna que esta deja en blanco a
propósito: **qué etapa posterior paga por cada "no".**
"""

# %%
import matplotlib.pyplot as plt

d313 = audit["3.13"]["per_dir"]
eng = {top: 100 * c["english paragraphs"] / max(c["paragraphs"], 1) for top, c in d313.items()}
worst = max(eng, key=eng.get) if eng else "?"
properties = [
    ("1 answer in it", f"API reference + tutorial; English-dominant up to {eng.get(worst, 0):.0f}% ({worst})"),
    ("2 clean text", "fuzzy entries used by a naive .po loader; mixed-language version labels"),
    ("3 structure", f"{sum(c['sections'] for c in d313.values())} sections, "
                    f"{len(audit['3.13']['anchors'])} API anchors (HTML route); flat pairs (.po route)"),
    ("4 stable ids", "HTML: section and API ids from the source (NOT measured here: compare two "
                     "downloads); .po: file path + source line"),
    ("5 metadata", "version = the branch/archive; per-API added/changed notes in the HTML only"),
    ("6 changes", "repo pinned by commit; archive rebuilt in place (pinned by sha256 + date)"),
    ("7 eval set", "auto-derivable from API anchors and version notes -- beware lexical leakage"),
    ("8 similarity", f"{100 * same / max(total, 1):.0f}% of paragraphs identical across 3.12/3.13"),
]
print_table("python-docs-es: the audit", ["measured"], [(p, {"measured": m}) for p, m in properties],
            smoke=SMOKE_TEST)

plt.figure(figsize=(7, 3))
plt.bar(list(eng), list(eng.values()))
plt.ylabel("English-dominant paragraphs (%)")
plt.title(f"python-docs-es 3.13 built docs: how Spanish is it? ({PROFILE})")
plt.tight_layout()
plt.savefig(ART / "stage0_english_by_dir.png", dpi=100)
plt.show()

store.write_json(ART / "stage0_audit.json", {
    "profile": PROFILE,
    "po": {"commit": PO_COMMITS["3.13"], "files": len(po_files),
           "by_dir": {k: dict(v) for k, v in by_dir.items()}, "merged_entries": merged,
           "has_signatures": has_signature, "has_versions": has_versions},
    "archives": {v: {"sha256": a["sha256"], "pinned": a["pinned"]} for v, a in archives.items()},
    "html": {v: {"per_dir": {k: dict(c) for k, c in a["per_dir"].items()}, "anchors": len(a["anchors"]),
                 "labels": dict(a["labels"])} for v, a in audit.items()},
    "versions": {"new_in_313": len(new_in_313), "gone_in_313": len(gone_in_313),
                 "identical_paragraph_share": same / max(total, 1)},
    "size": {"characters": total_chars, "tokens_per_char": tokens_per_char, "est_tokens": est_tokens},
    "closed_book": closed_book,
})
run.note("est_tokens", est_tokens)
run.finish()

# %% [markdown]
r"""
## Para tu informe

- **Qué ruta de entrada al corpus, y por qué.** Esta auditoría midió las dos y no eligió
  ninguna. La ruta `.po` tiene las marcas de calidad de los propios traductores y un historial de
  git; la ruta HTML tiene firmas, notas de versión e ids estables, y esconde la división
  fuzzy/sin traducir detrás de texto en inglés. Tomes la que tomes, nombrá lo que te cuesta más
  adelante.
- **¿Por qué el directorio con más inglés tiene tanto inglés?** Las marcas de la compilación y la
  comparación de versiones de arriba apuntan a una causa que no tiene nada que ver con el
  esfuerzo de los traductores. ¿Qué significa para un usuario que pregunta sobre esas páginas en
  cada versión?
- **¿Qué propiedades de las siete no cumple tu corpus -- y cuánto te cuesta la octava, la
  similitud -- y en qué etapa vas a pagar por cada una?** Una oración por cada "no".
- **¿Entra en una ventana?** Ahora tenés el número. Si entrara, ¿para qué construir
  recuperación?
- **¿Cuánto ya leyó el generador?** Mirá las respuestas a libro cerrado. ¿En cuáles confiaría un
  usuario -- y cuáles están mal de una forma que solo los documentos podrían corregir?
- **¿Qué hace que dos de tus documentos se puedan confundir,** y es una propiedad que querés
  conservar en tu evaluación o eliminar con ingeniería?

Siguiente: **Etapa 1** (`01_documents`) -- convertir la fuente en documentos: un registro por
documento, ids estables, secciones, metadatos que verificaste contra el texto.
"""
