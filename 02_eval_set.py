# %% [markdown]
r"""
# Etapa 2 · La vara de medir

Tres conjuntos, y después un candado:

| conjunto | contrato | para qué |
|---|---|---|
| `eval/single_turn.jsonl` | `schema.EvalItem` | Etapas 3 y 4 |
| `eval/dialogues.jsonl` | `schema.Dialogue` | Etapa 5 |
| `eval/scenarios.jsonl` | `schema.Scenario` | Etapa 6 |
| `eval/FROZEN.json` | `schema.freeze_eval` | el candado |

**Esta etapa no tiene perfil smoke.** El conjunto se congela contra el corpus **completo**, que es
el único contra el que los ids gold significan algo. Lo que sí sale de acá es `eval/smoke/`: un
puñado de ítems cuyo gold cae dentro de las diez páginas del perfil smoke, para que las Etapas 3-6
se puedan correr en la laptop sin medir nada.

**Cómo se escribieron las preguntas.** El enunciado avisa dos veces de lo mismo con dos trampas
distintas: una pregunta escrita con el pasaje abierto mide copia, y una pregunta escrita por un
modelo a partir del pasaje mide lo mismo con un paso más. Así que acá hay dos clases de ítem y
ninguna se escribió leyendo el pasaje:

- **derivados** (`identifier`, `metadata`): salen del corpus por código -- de las anclas que
  aparecen en un solo módulo y de las notas de versión --, con la respuesta verificada por
  contención contra el texto de la sección gold. No los redactó nadie: los redactó una plantilla.
- **escritos a mano** (`topical`, `out_of_corpus`, los diálogos, los escenarios): se escribieron
  mirando **los títulos de sección** y nada más. El título es el *tema*; la pregunta sale de
  conocer Python, no de leer el párrafo. Después el código verifica que la clave de respuesta
  exista en la sección gold, y cuando no existía se corrigió **la clave** (al vocabulario de la
  traducción), nunca la pregunta. La lista de las que se corrigieron está en la tabla de
  procedencia, al final.

Lo que esto no arregla: quien escribió las preguntas sabe Python y sabe qué módulos tiene el
corpus, así que el conjunto hereda *ese* sesgo. Una pregunta que un usuario real haría y que
este conjunto no mide está en el informe.

**Entra**: `artifacts/full/documents.jsonl` (los ids gold).
**Sale**: los tres `.jsonl`, `FROZEN.json`, y el chequeo de contaminación que va al informe.
"""

# %%
import environment  # noqa: F401  (efectos secundarios: guarda de JAX, caché de HF, bundle TLS, UTF-8)

environment.summary()

import collections
import datetime as dt
import json
import re
from pathlib import Path

import bm25s

import lexical
import store
from console import StageRun, clip, print_table, verdict
from schema import (Dialogue, EvalItem, Scenario, documents_hash, freeze_eval, load_eval,
                    validate_eval)

FULL = environment.artifacts_dir(False)
ART = FULL                          # esta etapa no tiene perfil smoke
EVAL = environment.ROOT / "eval"
SMOKE_EVAL = EVAL / "smoke"
VERSIONS = ("3.13", "3.12")
CURRENT = "3.13"

run = StageRun.start("02_eval_set", ART, profile="full",
                     config={"eval_dir": str(EVAL), "versions": list(VERSIONS)})

docs = store.read_jsonl(FULL / "documents.jsonl")
print(f"corpus: {len(docs)} documentos · corpus_sha256 {documents_hash([__import__('schema').Document.model_validate(d) for d in docs])[:16]}…")

# %% [markdown]
r"""
## 1 · Los índices que hacen falta para que un id gold sea verificable

Nada de esto es la vara: es la infraestructura para que cada etiqueta gold se resuelva **contra el
corpus** y no contra lo que creo recordar de él. Las cuatro cosas que resuelve:

1. `rel` + slug de sección -> los `doc_id` de las dos versiones (el gold como **conjunto**);
2. ancla -> la sección donde está definida, en cada versión;
3. sección -> su texto, para verificar por contención la clave de respuesta;
4. las anclas que aparecen en **un solo** módulo, que es de donde se derivan las preguntas de
   identificador.

Si un slug existe en 3.13 y no en 3.12, el gold queda de **una sola versión**: la página se
reescribió entre versiones y las copias ya no son intercambiables. La Etapa 1 midió cuántas
gemelas están en ese caso (la menos parecida, `library/dialog.html`, comparte 0.35 de sus
5-gramas), así que esto va a pasar y tiene que pasar explícito, no por omisión.
"""

# %%
by_rel: dict[str, dict[str, dict]] = collections.defaultdict(dict)
by_id: dict[str, dict] = {}
for d in docs:
    by_rel[d["metadata"]["rel"]][d["metadata"]["version"]] = d
    by_id[d["doc_id"]] = d

sections_of = {d["doc_id"]: {s["section_id"]: s for s in d["sections"]} for d in docs}
anchor_sec = {d["doc_id"]: {a["anchor_id"]: a["section_id"] for a in d["metadata"]["anchors"]}
              for d in docs}

SINGLE_VERSION: list[str] = []      # (para la tabla: qué gold quedó de una sola versión y por qué)


def sec_text(section_id: str) -> str:
    doc = by_id[section_id.split("#")[0]]
    s = sections_of[doc["doc_id"]][section_id]
    return doc["text"][s["start"]:s["end"]]


def gold(rel: str, slug: str, *, who: str = "", only: str | None = None) -> list[dict]:
    """El gold de una sección, como CONJUNTO de las versiones que la tienen."""
    out = []
    for v in ([only] if only else VERSIONS):
        doc = by_rel.get(rel, {}).get(v)
        if doc is None:
            continue
        sid = f"{doc['doc_id']}#{slug}"
        if sid in sections_of[doc["doc_id"]]:
            out.append({"doc_id": doc["doc_id"], "section_id": sid})
    if not out:
        raise KeyError(f"{who}: {rel}#{slug} no existe en ninguna versión")
    if len(out) == 1 and only is None:
        SINGLE_VERSION.append(f"{who}: {rel}#{slug} solo en {out[0]['doc_id'].split(':')[0]}")
    return out


def gold_anchor(rel: str, anchor_id: str, *, who: str = "") -> list[dict]:
    """El gold de la sección donde un ancla está definida, en cada versión que la define."""
    out = []
    for v in VERSIONS:
        doc = by_rel.get(rel, {}).get(v)
        if doc is None:
            continue
        sid = anchor_sec[doc["doc_id"]].get(anchor_id)
        if sid:
            out.append({"doc_id": doc["doc_id"], "section_id": sid})
    if not out:
        raise KeyError(f"{who}: el ancla {anchor_id} no está en {rel}")
    if len(out) == 1:
        SINGLE_VERSION.append(f"{who}: {anchor_id} solo en {out[0]['doc_id'].split(':')[0]}")
    return out


# Anclas que aparecen en un solo módulo, por su nombre corto. La Etapa 1 ya había medido que
# "única en un documento" es la unidad equivocada: por `doc_id` da 1.672 y está dominada por
# páginas que solo existen en 3.12. Por `rel` -- el módulo, sin la versión -- da 10.417.
rels_of_anchor: dict[str, set[str]] = collections.defaultdict(set)
rels_of_tail: dict[str, set[str]] = collections.defaultdict(set)
for d in docs:
    for a in d["metadata"]["anchors"]:
        rels_of_anchor[a["anchor_id"]].add(d["metadata"]["rel"])
        rels_of_tail[a["anchor_id"].split(".")[-1]].add(d["metadata"]["rel"])

print(f"anclas distintas: {len(rels_of_anchor):,} · en un solo módulo: "
      f"{sum(len(v) == 1 for v in rels_of_anchor.values()):,}")
print(f"nombres cortos en un solo módulo: {sum(len(v) == 1 for v in rels_of_tail.values()):,}")
print(f"páginas con gemela en la otra versión: {sum(len(v) == 2 for v in by_rel.values())} "
      f"de {len(by_rel)}")

# %% [markdown]
r"""
## 2 · Preguntas temáticas, escritas desde los títulos

Veinte. La fuente de cada una es el **título de sección**, que es el tema; el `slug` de abajo es
el gold, no el material de redacción. Tres son de sí/no y llevan `expected_verdict`: el enunciado
muestra por qué una clave de respuesta sola no alcanza ahí (`"3.13"` la contienen igual
«No, llegó en 3.13» y «Sí, está desde 3.13»).

Las claves son cortas y en el vocabulario de la **traducción**, no en el mío: la celda siguiente
verifica cada una por contención contra el texto de la sección gold, y las que no pasaron se
corrigieron ahí. Esa corrección es el único momento en que se abrió un pasaje, y toca la clave,
nunca la pregunta.
"""

# %%
# (id, pregunta, rel, slug, claves, veredicto, split)
TOPICAL = [
    ("t-range", "Necesito repetir algo diez veces en un for. ¿Qué uso para generar los números "
     "sin armar la lista a mano?", "tutorial/controlflow.html", "the-range-function",
     ["range"], None, "dev"),
    ("t-else-break", "Si un bucle for termina porque se ejecutó un break, ¿igual corre su bloque "
     "else?", "tutorial/controlflow.html", "else-clauses-on-loops", ["else"], "no", "test"),
    ("t-default-mutable", "Cuando pongo una lista vacía como valor por omisión de un parámetro, "
     "¿se crea una lista nueva en cada llamada?", "tutorial/controlflow.html",
     "default-argument-values", ["una vez"], "no", "test"),
    ("t-kwargs", "Tengo los argumentos en un diccionario y quiero pasarlos todos a una función. "
     "¿Cómo se hace?", "tutorial/controlflow.html", "unpacking-argument-lists", ["**"], None, "test"),
    ("t-finally", "¿Qué parte de un try se ejecuta siempre, pase lo que pase con la excepción?",
     "tutorial/errors.html", "defining-clean-up-actions", ["finally"], None, "test"),
    ("t-add-note", "Atrapé una excepción y quiero agregarle un dato de contexto antes de volver a "
     "lanzarla, sin tocar su mensaje. ¿El lenguaje tiene algo para eso?",
     "tutorial/errors.html", "enriching-exceptions-with-notes", ["add_note"], None, "test"),
    ("t-exc-group", "Un bloque mío puede fallar por varios motivos independientes a la vez. "
     "¿Cómo los reporto todos juntos en vez de uno?", "tutorial/errors.html",
     "raising-and-handling-multiple-unrelated-exceptions", ["ExceptionGroup"], None, "test"),
    ("t-yield", "¿Qué hace que una función devuelva sus valores de a uno en vez de todos juntos?",
     "tutorial/classes.html", "generators", ["yield"], None, "test"),
    ("t-mro", "Con varias clases base, ¿en qué orden se busca un atributo que no está en la clase "
     "hija?", "tutorial/classes.html", "multiple-inheritance", ["resolución"], None, "test"),
    ("t-deque", "Necesito una cola: saco siempre del principio. ¿Por qué una lista no sirve y qué "
     "uso en su lugar?", "tutorial/datastructures.html", "using-lists-as-queues",
     ["deque"], None, "dev"),
    ("t-empty-set", "¿Cómo creo un conjunto vacío, si las llaves ya son de diccionario?",
     "tutorial/datastructures.html", "sets", ["set()"], None, "test"),
    ("t-json-file", "Quiero guardar un diccionario en un archivo y después volver a leerlo igual. "
     "¿Qué me da la biblioteca estándar?", "tutorial/inputoutput.html",
     "saving-structured-data-with-json", ["json"], None, "test"),
    ("t-syspath", "¿Dónde busca Python un módulo cuando lo importo, y cómo veo esa lista?",
     "tutorial/modules.html", "the-module-search-path", ["sys.path"], None, "test"),
    ("t-venv", "¿Con qué comando armo un entorno virtual sin instalar nada de afuera?",
     "tutorial/venv.html", "creating-virtual-environments", ["venv"], None, "test"),
    ("t-float", "¿Por qué sumar 0.1 y 0.2 no da exactamente 0.3?",
     "tutorial/floatingpoint.html", "representation-error", ["binario"], None, "test"),
    ("t-stable-sort", "Si ordeno la misma lista dos veces, primero por un campo y después por "
     "otro, ¿se mantiene el orden del primero entre los empates del segundo?",
     "howto/sorting.html", "sort-stability-and-complex-sorts", ["estable"], "yes", "test"),
    ("t-log-default", "Si no configuro nada en logging, ¿qué mensajes llegan a salir?",
     "howto/logging.html", "logging-levels", ["WARNING"], None, "dev"),
    ("t-nfc", "Dos textos que se ven idénticos en pantalla no me dan iguales al compararlos. "
     "¿Qué tengo que hacer antes de comparar?", "howto/unicode.html", "comparing-strings",
     ["normalize"], None, "test"),
    ("t-store-true", "En argparse quiero una opción que sea solo un interruptor, sin valor. "
     "¿Cómo la declaro?", "howto/argparse.html", "introducing-optional-arguments",
     ["store_true"], None, "test"),
    ("t-enum-unique", "¿Cómo evito que dos miembros de una enumeración terminen con el mismo "
     "valor?", "howto/enum.html", "ensuring-unique-enumeration-values", ["unique"], None, "test"),
]

items: list[dict] = []
PROVENANCE: dict[str, str] = {}
KEY_FIXED: list[str] = []

for iid, question, rel, slug, keys, vd, split in TOPICAL:
    g = gold(rel, slug, who=iid)
    items.append({"item_id": iid, "question": question, "category": "topical", "gold": g,
                  "answer_key": keys, "expected_verdict": vd, "split": split,
                  "notes": "escrita desde el título de sección"})
    PROVENANCE[iid] = "a mano (título)"
print(f"temáticas: {len(items)}")

# %% [markdown]
r"""
### 2.1 · La clave de respuesta, verificada contra la sección gold

Una clave que no está en el pasaje gold es una clave que ninguna respuesta fundamentada va a
contener: el ítem queda imposible y la tabla lo lee como que el sistema falló. Así que cada clave
se busca, literal y sin acentos, en el texto de **alguna** de las secciones gold.

Las que no pasan no se tiran: se imprime el renglón del pasaje donde debería estar la palabra y
se cambia la clave por la que usa la traducción. Es el único lugar de esta etapa donde se lee un
pasaje, y lo que se corrige es la clave.

**La clave es UNA palabra, y es a propósito.** El campo dice *«short strings a correct answer
contains»*: las claves se leen **en conjunción**, una respuesta las tiene que contener todas. Con
esa lectura, agregar un sinónimo no afloja el puntaje, lo endurece: pedir `estable` *y*
`estabilidad` falla a la respuesta que solo dice «el orden es estable», que es correcta. Los
sinónimos, entonces, no entran en la clave; se usan abajo para comprobar que la traducción no usa
otra palabra, y nada más.
"""

# %%
def strip_accents(text: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").lower()


def key_missing(row: dict) -> list[str]:
    """Las claves que no aparecen en ninguna sección gold del ítem."""
    if not row.get("answer_key"):
        return []
    haystack = " ".join(strip_accents(sec_text(g["section_id"])) for g in row["gold"])
    return [k for k in row["answer_key"] if strip_accents(k) not in haystack]


missing = [(row["item_id"], key_missing(row)) for row in items]
missing = [(i, ks) for i, ks in missing if ks]
for iid, ks in missing:
    row = next(r for r in items if r["item_id"] == iid)
    print(f"\n{iid}: la clave {ks} no está en el pasaje gold")
    for g in row["gold"][:1]:
        print(f"   {g['section_id']}")
        print(f"   {clip(sec_text(g['section_id']), 300)}")
print(f"\nclaves que no pasaron la contención: {len(missing)} de {len(items)}")

# %% [markdown]
r"""
Ninguna clave falló, así que no hay nada que corregir y `KEY_FIXED` queda vacío. Eso no estaba
decidido de antemano: la lista de abajo son las cinco claves donde **esperaba** que la traducción
usara otra palabra, y la razón por la que el chequeo de arriba existe. Se prueba la alternativa
contra el mismo pasaje y se informa, **sin tocar la clave** — por lo conjuntivo de §2.1.

Lo que el resultado dice es útil igual: en `t-mro` el pasaje dice «resolución» *y* «orden», en
`t-stable-sort` «estable» *y* «estabilidad». Donde la alternativa no aparece (`una sola vez`,
`normalizar`) la palabra que sí está es la que quedó en la clave, que ya pasó la contención.
"""

# %%
# clave elegida -> la palabra que la traducción podría haber usado en su lugar
ALTERNATIVAS = {
    "t-mro": "orden",
    "t-default-mutable": "una sola vez",
    "t-nfc": "normalizar",
    "t-stable-sort": "estabilidad",
    "t-float": "fracciones binarias",
}
for iid, alt in ALTERNATIVAS.items():
    row = next(r for r in items if r["item_id"] == iid)
    probe = dict(row, answer_key=[alt])
    hay = not key_missing(probe)
    print(f"   {iid:20} clave {str(row['answer_key']):22} alternativa {alt!r:22} "
          f"{'también está' if hay else 'no está'}")

still_missing = [(r["item_id"], key_missing(r)) for r in items]
still_missing = [(i, k) for i, k in still_missing if k]
print(f"\nclaves sin verificar al final: {len(still_missing)} {still_missing}")
print(verdict(not still_missing,
              "cada clave temática existe en su pasaje gold: la contención es medible",
              "quedan claves que ningún pasaje contiene -- esos ítems miden que el sistema "
              "falle"))

# %% [markdown]
r"""
## 3 · Preguntas de identificador, derivadas

El enunciado es explícito: *«Las preguntas de identificador se pueden derivar de identificadores
que aparecen en exactamente un documento: derivalas, no las escribas.»* Dos familias, las dos de
una plantilla:

- **de qué módulo es**: un nombre corto que aparece en un solo módulo de todo el corpus
  (`copyfileobj`, `most_common`), preguntado sin el prefijo. La respuesta es el módulo, y es
  única por construcción.
- **cuál es el valor por omisión**: el parámetro y su valor salen de la **firma** que la página
  muestra, parseada del texto limpio. Son las preguntas más literales del conjunto, y van acá
  porque acá la coincidencia literal es justamente lo que se mide.

Dos decisiones que el sorteo necesita y que son mías, no del corpus:

1. **el sorteo se limita a módulos de uso corriente.** Hay 5.955 candidatos válidos y 417 son de
   `os`, 171 de `dis`, 110 de `pyexpat`. Una pregunta sobre `pyexpat.DefaultHandler` es válida y
   no la haría nadie: mediría la recuperación sobre un vocabulario que ningún usuario escribe.
2. **el sorteo es determinista** (orden alfabético y un paso fijo), así que volver a correr esta
   celda da los mismos ítems. Un `random.sample` sin semilla haría que el conjunto cambie entre
   corridas, y el conjunto es lo que *no* se mueve.
"""

# %%
POPULAR = ["library/pathlib.html", "library/shutil.html", "library/json.html", "library/csv.html",
           "library/collections.html", "library/datetime.html", "library/re.html",
           "library/textwrap.html", "library/random.html", "library/statistics.html",
           "library/tempfile.html", "library/zipfile.html", "library/subprocess.html",
           "library/sqlite3.html", "library/logging.html", "library/contextlib.html",
           "library/zoneinfo.html", "library/secrets.html", "library/difflib.html",
           "library/glob.html"]

SIG = re.compile(r"\(([^()\n]*(?:\([^()\n]*\)[^()\n]*)*)\)")


def module_of(rel: str) -> str:
    return rel.split("/")[-1].removesuffix(".html")


def signature_params(doc: dict, anchor_id: str) -> str | None:
    """La lista de parámetros tal como la página la muestra, buscando el id y sus sufijos.

    La firma visible casi nunca es el id completo (`Path.touch`, no `pathlib.Path.touch`): esto
    es lo mismo que descubrió la regla de limpieza de firmas partidas en la Etapa 1.
    """
    parts = anchor_id.split(".")
    for i in range(len(parts)):
        name = ".".join(parts[i:])
        for m in re.finditer(re.escape(name) + r"\s*(?=\()", doc["text"]):
            sig = SIG.match(doc["text"], m.end())
            if sig:
                return sig.group(1)
    return None


# --- familia 1: de qué módulo es ---------------------------------------------------------------
candidates = []
for rel in POPULAR:
    if len(by_rel.get(rel, {})) != 2:        # tiene que existir en las dos versiones
        continue
    mod = module_of(rel)
    doc = by_rel[rel][CURRENT]
    for a in doc["metadata"]["anchors"]:
        aid, tail = a["anchor_id"], a["anchor_id"].split(".")[-1]
        if not aid.startswith(mod + "."):
            continue
        if len(rels_of_anchor[aid]) != 1 or len(rels_of_tail[tail]) != 1 or len(tail) < 6:
            continue
        if aid not in anchor_sec[by_rel[rel]["3.12"]["doc_id"]]:
            continue                          # el ancla tiene que estar en las dos
        candidates.append((rel, mod, aid, tail))
candidates.sort(key=lambda c: (c[0], c[2]))
print(f"candidatos para 'de qué módulo es': {len(candidates)} en {len(set(c[0] for c in candidates))} módulos")

# un candidato por módulo, el del medio de la lista de ese módulo: determinista y no siempre
# el primero (que en una página suele ser la función más famosa, y esas las sabe un modelo de
# memoria -- que es justo lo que la fila closed-book de la Etapa 4 tiene que poder distinguir)
by_mod: dict[str, list] = collections.defaultdict(list)
for c in candidates:
    by_mod[c[1]].append(c)
PLANTILLA = ["¿A qué módulo de la biblioteca estándar pertenece `{tail}`?",
             "Vi `{tail}` en un ejemplo y no sé de dónde sale. ¿De qué módulo es?"]
for n, mod in enumerate(sorted(by_mod)[:10]):
    rel, _, aid, tail = by_mod[mod][len(by_mod[mod]) // 2]
    iid = f"id-mod-{tail}"
    items.append({"item_id": iid, "question": PLANTILLA[n % 2].format(tail=tail),
                  "category": "identifier", "gold": gold_anchor(rel, aid, who=iid),
                  "answer_key": [mod], "expected_verdict": None,
                  "split": "dev" if n < 2 else "test",
                  "notes": f"derivada: `{tail}` aparece en un solo módulo de todo el corpus"})
    PROVENANCE[iid] = "derivada (ancla única)"

# --- familia 2: cuál es el valor por omisión ----------------------------------------------------
VALUE = re.compile(r"^(0o[0-7]+|'[^']{1,24}'|\"[^\"]{1,24}\"|[1-9]\d{1,4})$")
defaults = []
for rel in POPULAR:
    if len(by_rel.get(rel, {})) != 2:
        continue
    doc = by_rel[rel][CURRENT]
    for a in doc["metadata"]["anchors"]:
        aid = a["anchor_id"]
        if len(rels_of_anchor[aid]) != 1 or aid not in anchor_sec[by_rel[rel]["3.12"]["doc_id"]]:
            continue
        params = signature_params(doc, aid)
        if not params:
            continue
        for param, value in re.findall(r"([a-z_][a-z_0-9]*) = (\S+)", params):
            if VALUE.match(value):
                defaults.append((rel, aid, param, value))
defaults.sort(key=lambda d: (d[0], d[1], d[2]))
print(f"candidatos para 'valor por omisión': {len(defaults)}")

seen_names: set[str] = set()
picked = []
for rel, aid, param, value in defaults:
    if aid in seen_names:
        continue
    seen_names.add(aid)
    picked.append((rel, aid, param, value))
for n, (rel, aid, param, value) in enumerate(picked[::max(1, len(picked) // 8)][:8]):
    short = ".".join(aid.split(".")[-2:]) if aid.count(".") > 1 else aid
    iid = f"id-def-{aid.replace('.', '-')}-{param}"
    items.append({"item_id": iid,
                  "question": f"¿Cuál es el valor por omisión del parámetro `{param}` de "
                              f"`{short}`?",
                  "category": "identifier", "gold": gold_anchor(rel, aid, who=iid),
                  "answer_key": [value.strip("'\"")], "expected_verdict": None,
                  "split": "dev" if n == 0 else "test",
                  "notes": f"derivada de la firma: {short}({param} = {value})"})
    PROVENANCE[iid] = "derivada (firma)"

n_ident = sum(r["category"] == "identifier" for r in items)
print(f"\nde identificador: {n_ident}")
for r in items:
    if r["category"] == "identifier":
        print(f"   {r['item_id']:<44} {clip(r['question'], 72)}  -> {r['answer_key']}")

# %% [markdown]
r"""
## 4 · Preguntas de metadatos y temporales, derivadas de las notas de versión

«Nuevo en la versión 3.12» es un dato estructurado que la página trae puesto: la Etapa 1 guardó
las 4.027 notas de 3.13 con su sección y verificó que los 2.310 números de versión sobrevivieran
literales a la limpieza. Lo que falta es **de quién habla cada nota**, y eso el HTML lo decía con
el anidado que el texto plano perdió.

Se recupera por posición, que es lo que el texto limpio conserva: la marca `¶` pasó a ser un
salto de línea, así que cada firma termina un renglón. El dueño de una nota es la **definición
más cercana hacia atrás**, con dos condiciones: a menos de 1.200 caracteres, y en la misma
sección. Las notas sin dueño claro se descartan -- son más de las que se usan, así que se puede
ser exigente.

Y después el par que el enunciado pide dos veces: la misma función preguntada para **dos
versiones**, con veredictos opuestos. Un brazo que contesta «sí» a todo acierta la mitad y falla
la otra; sin el par, acierta todo.
"""

# %%
NOTE = re.compile(r"(Nuevo en la versión|Distinto en la versión|Obsoleto desde la versión) "
                  r"(\d+\.\d+)")


def note_owners(doc: dict) -> list[tuple[str, str, str, str]]:
    """(anchor_id, clase de nota, versión, section_id) para las notas con dueño claro."""
    text = doc["text"]
    defs: list[tuple[int, str]] = []
    for a in doc["metadata"]["anchors"]:
        aid = a["anchor_id"]
        parts = aid.split(".")
        for i in range(len(parts)):
            name = ".".join(parts[i:])
            for m in re.finditer(re.escape(name) + r"(?=\s*\(|\n)", text):
                defs.append((m.start(), aid))
    defs.sort()
    bounds = [(s["start"], s["end"], s["section_id"]) for s in doc["sections"]]
    out = []
    for m in NOTE.finditer(text):
        before = [(p, aid) for p, aid in defs if p < m.start()]
        if not before:
            continue
        pos, aid = before[-1]
        if m.start() - pos > 1200:
            continue
        sec = next((sid for a, b, sid in bounds if a <= pos < b), None)
        if sec is None or not (sec == next((sid for a, b, sid in bounds
                                            if a <= m.start() < b), None)):
            continue
        out.append((aid, m.group(1), m.group(2), sec))
    return out


owners = []
for rel in POPULAR + ["library/itertools.html", "library/enum.html", "library/typing.html",
                      "library/functools.html", "library/venv.html", "library/os.path.html"]:
    for v in VERSIONS:
        doc = by_rel.get(rel, {}).get(v)
        if doc is None:
            continue
        for aid, kind, ver, sec in note_owners(doc):
            if len(rels_of_anchor[aid]) == 1:
                owners.append({"rel": rel, "version": v, "anchor": aid, "kind": kind,
                               "note_version": ver, "section_id": sec})
print(f"notas con dueño claro en los módulos elegidos: {len(owners)}")
added = [o for o in owners if o["kind"] == "Nuevo en la versión"]
print(f"   de ellas 'Nuevo en la versión': {len(added)} "
      f"({len({o['anchor'] for o in added})} anclas distintas)")

# una muestra a disco para leer con los ojos: la atribución por posición es una heurística, y
# una heurística que nadie miró es una fe
sample_path = ART / "stage2_note_owners.txt"
lines = []
for o in sorted(added, key=lambda o: (o["rel"], o["anchor"]))[:40]:
    doc = by_rel[o["rel"]][o["version"]]
    s = sections_of[doc["doc_id"]][o["section_id"]]
    body = doc["text"][s["start"]:s["end"]]
    i = body.find(o["anchor"].split(".")[-1])
    lines.append(f"{o['anchor']}  ({o['kind']} {o['note_version']})  {o['section_id']}\n"
                 f"   …{clip(body[max(0, i - 120):i + 320], 420)}…\n")
sample_path.write_text("\n".join(lines), encoding="utf-8")
print(f"muestra de 40 atribuciones para revisar a ojo: {sample_path}")
print("PENDIENTE (una persona): leer diez y confirmar que la nota habla de la función de arriba.")

# %%
# Los ítems. Se eligen las anclas cuya nota "Nuevo en la versión" existe en las DOS versiones
# del módulo con el mismo número: si solo está en una, la página se reescribió y el gold deja de
# ser un conjunto (eso se registra, no se esconde).
per_anchor: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
for o in added:
    per_anchor[(o["rel"], o["anchor"])].add(o["note_version"])
stable = sorted((k, next(iter(v))) for k, v in per_anchor.items() if len(v) == 1)

n_meta = 0
for (rel, aid), ver in stable:
    if n_meta >= 6:
        break
    short = ".".join(aid.split(".")[-2:]) if aid.count(".") > 1 else aid
    iid = f"meta-ver-{aid.replace('.', '-')}"
    try:
        g = gold_anchor(rel, aid, who=iid)
    except KeyError:
        continue
    items.append({"item_id": iid,
                  "question": f"¿Desde qué versión de Python está disponible `{short}`?",
                  "category": "metadata", "gold": g, "answer_key": [ver],
                  "expected_verdict": None, "split": "dev" if n_meta == 0 else "test",
                  "notes": f"derivada de la nota de versión de {aid}"})
    PROVENANCE[iid] = "derivada (nota de versión)"
    n_meta += 1

# El par: la misma función, dos versiones, veredictos opuestos.
PAIRS = 0
for (rel, aid), ver in stable:
    if PAIRS >= 2:
        break
    major, minor = (int(x) for x in ver.split("."))
    older = f"{major}.{minor - 1}"
    short = ".".join(aid.split(".")[-2:]) if aid.count(".") > 1 else aid
    try:
        g = gold_anchor(rel, aid, who=f"pair-{aid}")
    except KeyError:
        continue
    for asked, vd in ((older, "no"), (ver, "yes")):
        iid = f"meta-pair-{aid.replace('.', '-')}-{asked}"
        items.append({"item_id": iid,
                      "question": f"¿Puedo usar `{short}` en Python {asked}?",
                      "category": "metadata", "gold": g, "answer_key": [ver],
                      "expected_verdict": vd, "split": "test",
                      "notes": "par de actualización: la misma función, dos versiones, "
                               f"veredictos opuestos (llegó en {ver})"})
        PROVENANCE[iid] = "derivada (par de versión)"
    PAIRS += 1

print(f"de metadatos/temporales: {sum(r['category'] == 'metadata' for r in items)}")
for r in items:
    if r["category"] == "metadata":
        print(f"   [{r['expected_verdict'] or '  '}] {clip(r['question'], 78)} -> {r['answer_key']}")

# %% [markdown]
r"""
## 5 · Fuera del corpus: adyacentes, verificadas ausentes

El enunciado pide al menos tres, y **adyacentes**: «un mes que no tenés, una función que no existe
en tu versión», no «¿quién ganó el Mundial de 1986?». Las nuestras son todas de **Python 3.14**:
la versión siguiente a la que fijamos. Un usuario las va a preguntar, un modelo las vio en su
preentrenamiento, y el corpus no las tiene. Es el caso exacto.

Verificadas ausentes: el término no aparece **ni una vez** en los 13,9 millones de caracteres. La
celda lo chequea y se cae si alguna está presente -- una pregunta «fuera del corpus» que el corpus
contesta no mide abstención, mide un error de etiquetado. (La primera lista tenía siete términos
más: `tomllib`, `ExceptionGroup`, `asyncio.TaskGroup`, `itertools.batched`, `copy.replace`,
`sqlite3.Blob`, `PythonFinalizationError`. Todos están en el corpus. Por eso se chequea.)

**Qué cuenta como correcto, decidido ahora.** Una respuesta es correcta si *se niega* o si dice,
fundamentado, **que eso no existe en estas versiones**. Nombrar la alternativa que sí existe
(`shutil.copy2` en vez de `Path.copy`) suma legibilidad y no cambia el puntaje. Es **incorrecta**
si inventa una firma, un parámetro o una versión, incluso si aclara que no está seguro. Los dos
brazos de cada tabla se puntúan con esta regla, no con la que convenga.
"""

# %%
OUT_OF_CORPUS = [
    ("oc-path-copy", "¿Cómo copio un archivo con `Path.copy()` de pathlib?", "Path.copy", "test"),
    ("oc-zstd", "¿Qué parámetros acepta `compression.zstd.open()`?", "compression.zstd", "test"),
    ("oc-tstring", "¿Cómo armo una plantilla con una t-string (`t'...'`)?",
     "template string literal", "dev"),
    ("oc-interp-pool", "¿Qué hace `InterpreterPoolExecutor` de `concurrent.futures`?",
     "InterpreterPoolExecutor", "test"),
    ("oc-annotationlib", "¿Cómo leo las anotaciones de una función con `annotationlib`?",
     "annotationlib", "test"),
]

ALL_TEXT = "\n".join(d["text"] for d in docs)
present = {term: ALL_TEXT.count(term) for _, _, term, _ in OUT_OF_CORPUS}
for term, n in present.items():
    print(f"   {term:<28} {n} ocurrencias en el corpus")
if any(present.values()):
    raise RuntimeError(f"estos términos SÍ están en el corpus: "
                       f"{[t for t, n in present.items() if n]}")
print(verdict(True, "las cinco están ausentes: la abstención es medible", ""))

for iid, question, term, split in OUT_OF_CORPUS:
    items.append({"item_id": iid, "question": question, "category": "out_of_corpus",
                  "gold": [], "answerable": False, "answer_key": None,
                  "expected_verdict": None, "split": split,
                  "notes": f"adyacente: `{term}` es de Python 3.14, verificado ausente "
                           f"(0 ocurrencias). Correcto = negarse o decir que no existe en "
                           f"3.13/3.12; incorrecto = inventar firma o versión."})
    PROVENANCE[iid] = "a mano (3.14 adyacente)"

by_cat = collections.Counter(r["category"] for r in items)
print(f"\nun solo turno: {len(items)} ítems · {dict(by_cat)}")
print(verdict(len(items) >= 30 and by_cat["out_of_corpus"] >= 3 and by_cat["identifier"] >= 5
              and by_cat["metadata"] >= 5,
              "cumple los mínimos de §7.3 (>=30, >=3 fuera, >=5 identificador, >=5 metadatos)",
              "falta cantidad en alguna categoría"))

# %% [markdown]
r"""
## 6 · Contaminación, con el mismo tokenizador que el canal léxico

El fragmento del enunciado, con una sola diferencia: vive en `lexical.py` y no en esta celda,
porque la Etapa 3 tiene que importar **el mismo** stemmer. Dos copias que empiezan iguales
terminan distintas sin que nadie se entere, y entonces el chequeo diría que no hay copia mientras
BM25 cobra las frases copiadas.

Dos números por ítem: `overlap` (qué fracción de las palabras de contenido de la pregunta está en
el pasaje) y `longest_shared_ngram` (la frase más larga compartida, en tokens). **La bandera mira
la frase, no el solapamiento**: «¿cómo ordeno una lista por clave?» sobre una página de
ordenamiento comparte casi todas las palabras, y eso no es copia, es el tema. Cinco tokens
seguidos sí lo son.

Las de `identifier` se miden igual y **no** se reescriben: son deliberadamente literales y están
en su propia categoría justamente por eso.
"""

# %%
rows = []
for row in items:
    if not row["gold"]:
        continue
    gold_text = " ".join(sec_text(g["section_id"]) for g in row["gold"])
    c = lexical.contamination(row["question"], gold_text)
    rows.append((row["item_id"], row["category"], c))

flagged = [(i, cat, c) for i, cat, c in rows if c["flag"]]
print_table("contaminación por categoría",
            ["ítems", "overlap medio", "n-grama compartido máximo", "marcados"],
            [(cat, {"ítems": sum(1 for _, k, _ in rows if k == cat),
                    "overlap medio": round(sum(c["overlap"] for _, k, c in rows if k == cat)
                                           / max(1, sum(1 for _, k, _ in rows if k == cat)), 2),
                    "n-grama compartido máximo": max((c["longest_shared_ngram"]
                                                      for _, k, c in rows if k == cat), default=0),
                    "marcados": sum(1 for _, k, c in rows if k == cat and c["flag"])})
             for cat in ("topical", "identifier", "metadata")],
            name_header="categoría")
for i, cat, c in flagged:
    q = next(r["question"] for r in items if r["item_id"] == i)
    print(f"   MARCADO {i} ({cat}, n={c['longest_shared_ngram']}): {clip(q, 90)}")
print(verdict(not [f for f in flagged if f[1] == "topical"],
              "ninguna pregunta temática comparte cinco tokens seguidos con su pasaje: las "
              "escritas desde el título no importaron la redacción del pasaje",
              "hay temáticas marcadas: reescribirlas desde el título, sin abrir el pasaje"))

# %% [markdown]
r"""
### 6.1 · Y una recuperación solo con BM25

El paso 2 pide correrla *ya*, y no es para reportar un número: es para ver si el conjunto es
trivial. Si BM25 solo, sin reranker ni nada, acierta todo, entonces las tablas de la Etapa 3 no
van a poder mover nada y lo que vamos a estar mirando es cómo se escribieron las preguntas.

El índice es a nivel **sección** (5.614 secciones) y usa `lexical.tokenize`, el mismo del chequeo
de arriba. Es un índice de juguete comparado con el de la Etapa 3 -- sin chunking, sin prefijos,
sin filtros --, y eso es lo que lo hace una línea de base honesta.
"""

# %%
sec_ids, sec_docs = [], []
for d in docs:
    for s in d["sections"]:
        sec_ids.append(s["section_id"])
        sec_docs.append(d["text"][s["start"]:s["end"]])
tokens = [lexical.tokenize(t) for t in sec_docs]
bm25 = bm25s.BM25()
bm25.index(tokens, show_progress=False)
print(f"índice léxico de juguete: {len(sec_ids):,} secciones")

answerable = [r for r in items if r["gold"]]
queries = [lexical.tokenize(r["question"]) for r in answerable]
ids, _ = bm25.retrieve(queries, k=10, show_progress=False)
hits = collections.Counter()
per_item = {}
for row, got in zip(answerable, ids):
    retrieved = [sec_ids[j] for j in got]
    want = {g["section_id"] for g in row["gold"]}
    rank = next((n + 1 for n, s in enumerate(retrieved) if s in want), None)
    per_item[row["item_id"]] = rank
    hits[row["category"]] += rank is not None

print_table("BM25 solo, a nivel sección (hit@10)",
            ["ítems con gold", "hit@10", "%"],
            [(cat, {"ítems con gold": sum(1 for r in answerable if r["category"] == cat),
                    "hit@10": hits[cat],
                    "%": round(100 * hits[cat] / max(1, sum(1 for r in answerable
                                                            if r["category"] == cat)))})
             for cat in ("topical", "identifier", "metadata")],
            name_header="categoría")
total = sum(hits.values())
print(f"total: {total} de {len(answerable)} ({100 * total / len(answerable):.0f} %)")
print(verdict(total < len(answerable),
              "BM25 solo no resuelve el conjunto: queda margen para que la Etapa 3 mueva algo",
              "BM25 solo lo resuelve entero -- el conjunto es demasiado literal y las tablas "
              "de la Etapa 3 no van a poder distinguir nada"))
print("\nlos que BM25 no encontró (los que la Etapa 3 tiene que ganar):")
for iid, rank in per_item.items():
    if rank is None:
        print(f"   {iid:<44} {clip(next(r['question'] for r in items if r['item_id'] == iid), 70)}")

# %% [markdown]
r"""
## 7 · Guiones de conversación

Nueve diálogos de cuatro turnos. Lo que se mide con ellos en la Etapa 5 es **la pregunta que se
le manda al recuperador**, así que cada turno de seguimiento lleva su etiqueta de dependencia y su
reescritura autónoma escrita a mano:

- `coreference`: «¿y cómo lo leo de vuelta?» -- *lo* es el archivo del turno anterior;
- `ellipsis`: «¿y recursivo?» -- falta el verbo y el objeto;
- `topic_shift`: cambia de módulo. Es el turno que castiga a quien pega todo el historial;
- `return`: vuelve al tema de hace dos turnos. Es el que castiga a quien solo guarda el último;
- `unanswerable`: el seguimiento natural que el corpus no contesta.

La **respuesta guionada** (`assistant`) importa más de lo que parece: en la Etapa 5 todos los
brazos reciben estas respuestas como historial en vez de las suyas. Sin eso, un brazo que contesta
mal el turno 1 llega al turno 2 con otro historial y la comparación deja de ser entre estrategias
de contexto para ser entre conversaciones distintas.
"""

# %%
def T(n, user, *, depends="standalone", oracle=None, rel=None, slug=None, keys=None,
      assistant=None, verdict_=None, answerable=True, anchor=None):
    turn = {"turn": n, "user": user, "depends_on": depends, "assistant": assistant,
            "answerable": answerable}
    if oracle:
        turn["oracle_rewrite"] = oracle
    if answerable:
        who = f"d#{n}"
        turn["gold"] = gold_anchor(rel, anchor, who=who) if anchor else gold(rel, slug, who=who)
        turn["answer_key"] = keys
        turn["expected_verdict"] = verdict_
    return turn


DIALOGUES = [
    {"dialogue_id": "d-pathlib", "split": "test", "notes": "corref + elipsis + sin respuesta",
     "turns": [
         T(1, "¿Cómo recorro todos los archivos de un directorio con pathlib?",
           rel="library/pathlib.html", slug="reading-directories", keys=["iterdir"],
           assistant="Con `Path.iterdir()`, que produce un objeto de ruta por cada entrada del "
                     "directorio."),
         T(2, "¿Y si solo quiero los que terminan en .csv?", depends="coreference",
           oracle="Con pathlib, ¿cómo listo solo los archivos .csv de un directorio?",
           rel="library/pathlib.html", slug="reading-directories", keys=["glob"],
           assistant="Con `Path.glob('*.csv')`."),
         T(3, "¿Y recursivo?", depends="ellipsis",
           oracle="Con pathlib, ¿cómo listo los archivos .csv de un directorio y de todos sus "
                  "subdirectorios?",
           rel="library/pathlib.html", slug="reading-directories", keys=["rglob"],
           assistant="Con `Path.rglob('*.csv')`, que equivale a `glob()` con «**/» adelante."),
         T(4, "¿Y con `Path.copy()` los copio a otra carpeta?", depends="unanswerable",
           oracle="¿Existe `Path.copy()` en pathlib?", answerable=False,
           assistant="En estas versiones de la documentación no existe `Path.copy()`."),
     ]},
    {"dialogue_id": "d-json-csv", "split": "test", "notes": "cambio de tema y vuelta",
     "turns": [
         T(1, "¿Cómo escribo un diccionario a un archivo JSON?",
           rel="library/json.html", slug="basic-usage", keys=["dump"],
           assistant="Con `json.dump(obj, archivo)`."),
         T(2, "¿Y cómo lo leo de vuelta?", depends="coreference",
           oracle="¿Cómo leo un archivo JSON y lo convierto en un diccionario?",
           rel="library/json.html", slug="basic-usage", keys=["load"],
           assistant="Con `json.load(archivo)`."),
         T(3, "Cambiando de tema: ¿cómo leo un CSV usando la primera fila como nombres de "
              "columna?", depends="topic_shift",
           oracle="¿Cómo leo un CSV usando la primera fila como nombres de columna?",
           rel="library/csv.html", slug="module-contents", keys=["DictReader"],
           assistant="Con `csv.DictReader`, que devuelve cada fila como un diccionario."),
         T(4, "Volviendo al JSON: ¿cómo lo guardo indentado para poder leerlo a ojo?",
           depends="return",
           oracle="¿Cómo escribo un archivo JSON indentado con el módulo json?",
           rel="library/json.html", slug="basic-usage", keys=["indent"],
           assistant="Pasándole `indent=2` a `json.dump`."),
     ]},
    {"dialogue_id": "d-fechas", "split": "test", "notes": "cadena de elipsis sobre fechas",
     "turns": [
         T(1, "¿Cómo obtengo la fecha y hora actual?",
           rel="library/datetime.html", slug="datetime-objects", keys=["now"],
           assistant="Con `datetime.now()`."),
         T(2, "¿Y con la zona horaria de Buenos Aires?", depends="coreference",
           oracle="¿Cómo obtengo la hora actual en la zona horaria de Buenos Aires?",
           rel="library/zoneinfo.html", slug="using-zoneinfo", keys=["ZoneInfo"],
           assistant="Con `datetime.now(ZoneInfo('America/Argentina/Buenos_Aires'))`."),
         T(3, "¿Y formateada como 2026-10-24?", depends="ellipsis",
           oracle="¿Cómo formateo un datetime con el formato 2026-10-24?",
           rel="library/datetime.html", slug="strftime-and-strptime-format-codes",
           keys=["strftime", "%Y"],
           assistant="Con `strftime('%Y-%m-%d')`."),
         T(4, "¿Y al revés, de ese texto a datetime?", depends="coreference",
           oracle="¿Cómo convierto la cadena '2026-10-24' en un objeto datetime?",
           rel="library/datetime.html", slug="strftime-and-strptime-behavior", keys=["strptime"],
           assistant="Con `datetime.strptime('2026-10-24', '%Y-%m-%d')`."),
     ]},
    {"dialogue_id": "d-subprocess", "split": "test", "notes": "termina sin respuesta",
     "turns": [
         T(1, "¿Cómo ejecuto un comando del sistema desde Python y me quedo con su salida?",
           rel="library/subprocess.html", slug="using-the-subprocess-module",
           keys=["run", "capture_output"],
           assistant="Con `subprocess.run([...], capture_output=True, text=True)`."),
         T(2, "¿Y si quiero que falle cuando el comando devuelve error?", depends="coreference",
           oracle="Con subprocess.run, ¿cómo hago que se lance una excepción si el comando "
                  "devuelve un código de error?",
           rel="library/subprocess.html", slug="using-the-subprocess-module", keys=["check"],
           assistant="Pasándole `check=True`: lanza `CalledProcessError`."),
         T(3, "¿Y con un tiempo límite?", depends="ellipsis",
           oracle="Con subprocess.run, ¿cómo le pongo un límite de tiempo al comando?",
           rel="library/subprocess.html", slug="using-the-subprocess-module", keys=["timeout"],
           assistant="Con `timeout=<segundos>`, que lanza `TimeoutExpired`."),
         T(4, "¿Y cómo lo corro en un intérprete aparte con `InterpreterPoolExecutor`?",
           depends="unanswerable",
           oracle="¿Existe `InterpreterPoolExecutor` en concurrent.futures?", answerable=False,
           assistant="`InterpreterPoolExecutor` no está en estas versiones de la documentación."),
     ]},
    {"dialogue_id": "d-sqlite", "split": "test", "notes": "vuelve al tema del turno 1",
     "turns": [
         T(1, "¿Cómo abro una base SQLite y hago una consulta?",
           rel="library/sqlite3.html", slug="tutorial", keys=["connect", "execute"],
           assistant="Con `sqlite3.connect('archivo.db')` y después `cur.execute(...)`."),
         T(2, "¿Cómo le paso valores a esa consulta sin armar la cadena a mano?",
           depends="coreference",
           oracle="En sqlite3, ¿cómo paso valores a una consulta SQL sin concatenar cadenas?",
           rel="library/sqlite3.html",
           slug="how-to-use-placeholders-to-bind-values-in-sql-queries", keys=["?"],
           assistant="Con marcadores de posición: `execute('... WHERE a = ?', (valor,))`."),
         T(3, "¿Y para que las filas vengan como diccionarios?", depends="ellipsis",
           oracle="En sqlite3, ¿cómo hago que las filas se devuelvan como objetos tipo "
                  "diccionario?",
           rel="library/sqlite3.html", slug="how-to-create-and-use-row-factories",
           keys=["row_factory"],
           assistant="Asignando `con.row_factory = sqlite3.Row`."),
         T(4, "Volviendo a la consulta: ¿cuándo se confirman los cambios?", depends="return",
           oracle="En sqlite3, ¿cuándo se confirman los cambios de una transacción?",
           rel="library/sqlite3.html", slug="transaction-control", keys=["commit"],
           assistant="Cuando llamás a `con.commit()`."),
     ]},
    {"dialogue_id": "d-logging", "split": "test", "notes": "cambio de tema al final",
     "turns": [
         T(1, "¿Cómo dejo registro de lo que hace mi script en un archivo?",
           rel="howto/logging.html", slug="logging-to-a-file", keys=["basicConfig", "filename"],
           assistant="Con `logging.basicConfig(filename='app.log', level=logging.INFO)`."),
         T(2, "¿Y cómo le agrego la hora a cada línea?", depends="coreference",
           oracle="En logging, ¿cómo incluyo la fecha y hora en cada mensaje?",
           rel="howto/logging.html", slug="displaying-the-date-time-in-messages",
           keys=["asctime"],
           assistant="Con `format='%(asctime)s %(message)s'`."),
         T(3, "¿Y los mensajes de depuración?", depends="ellipsis",
           oracle="En logging, ¿cómo hago para que también se registren los mensajes de nivel "
                  "DEBUG?",
           rel="howto/logging.html", slug="logging-levels", keys=["DEBUG"],
           assistant="Bajando el nivel: `level=logging.DEBUG`."),
         T(4, "Otra cosa: ¿cómo creo un archivo temporal que se borre solo?",
           depends="topic_shift",
           oracle="¿Cómo creo un archivo temporal que se borre solo al cerrarlo?",
           rel="library/tempfile.html", slug="module-tempfile",
           keys=["NamedTemporaryFile", "TemporaryFile"],
           assistant="Con `tempfile.NamedTemporaryFile()`."),
     ]},
    {"dialogue_id": "d-regex", "split": "test", "notes": "cuatro turnos sobre el mismo módulo",
     "turns": [
         T(1, "¿Cómo busco un patrón en un texto con expresiones regulares?",
           rel="library/re.html", slug="functions", keys=["search"],
           assistant="Con `re.search(patrón, texto)`."),
         T(2, "¿Y cómo saco todas las coincidencias, no solo la primera?", depends="coreference",
           oracle="Con el módulo re, ¿cómo obtengo todas las coincidencias de un patrón en un "
                  "texto?",
           rel="library/re.html", slug="functions", keys=["findall"],
           assistant="Con `re.findall(patrón, texto)`."),
         T(3, "¿Y reemplazarlas?", depends="ellipsis",
           oracle="Con el módulo re, ¿cómo reemplazo todas las coincidencias de un patrón?",
           rel="library/re.html", slug="functions", keys=["sub"],
           assistant="Con `re.sub(patrón, reemplazo, texto)`."),
         T(4, "Si el patrón lo uso mil veces, ¿conviene hacer algo?", depends="coreference",
           oracle="Con el módulo re, ¿conviene compilar un patrón que se usa muchas veces?",
           rel="howto/regex.html", slug="compiling-regular-expressions", keys=["compile"],
           verdict_="yes",
           assistant="Sí: `re.compile(patrón)` una vez y después usar el objeto compilado."),
     ]},
    {"dialogue_id": "d-collections", "split": "dev", "notes": "el diálogo de desarrollo",
     "turns": [
         T(1, "¿Cómo cuento cuántas veces aparece cada palabra en una lista?",
           rel="library/collections.html", slug="counter-objects", keys=["Counter"],
           assistant="Con `collections.Counter(lista)`."),
         T(2, "¿Y las diez más frecuentes?", depends="coreference",
           oracle="Con collections.Counter, ¿cómo obtengo los diez elementos más frecuentes?",
           rel="library/collections.html", slug="counter-objects", keys=["most_common"],
           assistant="Con `c.most_common(10)`."),
         T(3, "¿Y si quiero agrupar valores en listas por clave sin inicializar cada clave?",
           depends="ellipsis",
           oracle="¿Qué estructura de collections agrupa valores en listas por clave sin "
                  "inicializar la clave?",
           rel="library/collections.html", slug="defaultdict-objects", keys=["defaultdict"],
           assistant="Con `collections.defaultdict(list)`."),
         T(4, "Volviendo al conteo: ¿puedo restarle otro conteo?", depends="return",
           oracle="¿Se puede restar un Counter de otro en collections?",
           rel="library/collections.html", slug="counter-objects", keys=["subtract"],
           verdict_="yes",
           assistant="Sí, con `c.subtract(otro)` o con el operador `-`."),
     ]},
    {"dialogue_id": "d-venv", "split": "test", "notes": "termina en una herramienta de afuera",
     "turns": [
         T(1, "¿Cómo aíslo las dependencias de un proyecto?",
           rel="tutorial/venv.html", slug="creating-virtual-environments", keys=["venv"],
           assistant="Con un entorno virtual: `python -m venv .venv`."),
         T(2, "¿Y cómo lo activo en Windows?", depends="coreference",
           oracle="¿Cómo activo un entorno virtual en Windows?",
           rel="library/venv.html", slug="creating-virtual-environments", keys=["activate"],
           assistant="Con `.venv\\Scripts\\activate`."),
         T(3, "¿Y para instalar ahí los paquetes?", depends="ellipsis",
           oracle="¿Cómo instalo paquetes dentro de un entorno virtual?",
           rel="tutorial/venv.html", slug="managing-packages-with-pip", keys=["pip"],
           assistant="Con `pip install`, con el entorno activado."),
         T(4, "¿Y si lo quiero manejar con `pipenv`?", depends="unanswerable",
           oracle="¿Cómo manejo un entorno virtual con pipenv?", answerable=False,
           assistant="`pipenv` es una herramienta de terceros: no está en la documentación "
                     "oficial que tengo."),
     ]},
]

# El mismo chequeo de ausencia que para los ítems fuera del corpus: un turno "sin respuesta"
# que el corpus contesta no mide nada.
for term in ("Path.copy", "InterpreterPoolExecutor", "pipenv"):
    if ALL_TEXT.count(term):
        raise RuntimeError(f"el turno sin respuesta sobre {term} es contestable: "
                           f"{ALL_TEXT.count(term)} ocurrencias")

dialogues = [Dialogue.model_validate(d) for d in DIALOGUES]
deps = collections.Counter(t.depends_on for d in dialogues for t in d.turns)
print(f"diálogos: {len(dialogues)} · turnos: {sum(len(d.turns) for d in dialogues)}")
print(f"dependencias: {dict(deps)}")

# Las claves de los turnos, verificadas igual que las temáticas.
bad_turns = []
for d in DIALOGUES:
    for t in d["turns"]:
        if t.get("answerable") and t.get("answer_key"):
            ks = key_missing({"gold": t["gold"], "answer_key": t["answer_key"]})
            if ks:
                bad_turns.append((d["dialogue_id"], t["turn"], ks))
print(f"turnos con clave que no está en su pasaje gold: {len(bad_turns)} {bad_turns}")

# %% [markdown]
r"""
## 8 · Escenarios multi-sesión

Cuatro personas ficticias, con los datos que para *este* corpus cambian la respuesta: la versión
de Python que corren, el sistema operativo, y una preferencia declarada. Timestamps simulados: la
Etapa 6 calcula la recencia contra **esas** fechas y no contra el reloj, así que un escenario se
replica igual el año que viene.

Cada escenario tiene, porque el enunciado lo pide, **un hecho que cambia** y **uno que nunca se
dijo**. El que cambia es el que separa una memoria que registra actualizaciones de una que
acumula: en `s-etl` la misma pregunta se hace antes y después de la actualización, con veredictos
**opuestos y los dos correctos**. El que nunca se dijo es la sonda de abstención: `answerable:
false`, sin evidencia y sin gold, y el validador del schema exige justamente esa coherencia.

Sobre separar roles (paso 5): acá hay una sola persona. Lo que se puede hacer y se hizo es separar
los *momentos* -- las sondas se escribieron en esta etapa, antes de que exista cualquier política
de escritura o lectura de memoria, y quedan congeladas antes del primer resultado. La Etapa 6 no
las puede ajustar; a lo sumo puede fallarlas.
"""

# %%
def P(pid, after, question, ability, *, ev=(), keys=None, rel=None, slug=None, vd=None,
       answerable=True, notes=""):
    probe = {"probe_id": pid, "after_session": after, "question": question, "ability": ability,
             "evidence_sessions": list(ev), "answerable": answerable, "notes": notes}
    if answerable:
        probe["answer_key"] = keys
        probe["expected_verdict"] = vd
        if rel:
            probe["gold"] = gold(rel, slug, who=pid)
    return probe


SCENARIOS = [
    {"scenario_id": "s-etl", "split": "test",
     "persona": "Marina, ficticia: desarrolladora que escribe un ETL de CSV a SQLite y actualiza "
                "su Python en el medio del proyecto.",
     "sessions": [
         {"session_id": "s1", "date": "2026-02-10T09:00:00", "turns": [
             {"user": "Estoy armando un ETL que lee CSV y escribe en SQLite. Corro Python 3.11 "
                      "en Ubuntu.",
              "assistant": "Anotado: Python 3.11 en Ubuntu, ETL de CSV a SQLite."},
             {"user": "Prefiero no agregar dependencias externas: solo biblioteca estándar.",
              "assistant": "Entendido, me quedo en la biblioteca estándar."}]},
         {"session_id": "s2", "date": "2026-04-22T15:30:00", "turns": [
             {"user": "El ETL ya procesa dos millones de filas y sigo en 3.11.",
              "assistant": "Anotado."},
             {"user": "Me molesta tener que armar los lotes de inserción a mano.",
              "assistant": "Se puede resolver con un iterador que corte por tamaño."}]},
         {"session_id": "s3", "date": "2026-09-01T10:15:00", "turns": [
             {"user": "Actualicé a Python 3.13.", "assistant": "Anotado: ahora 3.13."},
             {"user": "Y ahora el ETL también corre en un servidor Windows.",
              "assistant": "Anotado."}]},
     ],
     "probes": [
         P("p-ver-1", "s1", "¿Qué versión de Python uso?", "extraction", ev=["s1"],
           keys=["3.11"]),
         P("p-pref", "s1", "¿Me conviene instalar pandas para leer los CSV?", "preference",
           ev=["s1"], keys=["csv"], vd="no",
           notes="la respuesta correcta respeta la preferencia declarada en s1"),
         P("p-batched-antes", "s2", "¿Puedo usar `itertools.batched` para armar los lotes?",
           "temporal", ev=["s1", "s2"], rel="library/itertools.html", slug="module-itertools",
           keys=["3.12"], vd="no",
           notes="mitad del par de actualización: en s2 todavía corre 3.11"),
         P("p-batched-despues", "s3", "¿Y ahora puedo usar `itertools.batched`?",
           "knowledge_update", ev=["s1", "s3"], rel="library/itertools.html",
           slug="module-itertools", keys=["3.12"], vd="yes",
           notes="la otra mitad: después de s3 corre 3.13. Veredicto opuesto, los dos correctos"),
         P("p-trayecto", "s3", "¿De qué versión vengo y a cuál pasé?", "multi_session",
           ev=["s1", "s3"], keys=["3.11", "3.13"]),
         P("p-nunca-dicho", "s3", "¿Qué motor de base de datos usa el servidor Windows de "
           "producción?", "abstention", answerable=False,
           notes="nunca se dijo: ni en las sesiones ni en el corpus"),
         P("p-febrero", "s3", "Cuando arranqué el proyecto en febrero, ¿qué versión usaba?",
           "temporal", ev=["s1", "s3"], keys=["3.11"],
           notes="el valor viejo sigue siendo la respuesta correcta a una pregunta fechada"),
     ]},
    {"scenario_id": "s-rutas", "split": "dev",
     "persona": "Tomás, ficticio: escribe scripts en Windows que después tienen que correr en "
                "macOS, y declara preferir pathlib.",
     "sessions": [
         {"session_id": "s1", "date": "2026-03-05T11:00:00", "turns": [
             {"user": "Trabajo en Windows 11 y mis scripts tienen rutas con barras invertidas.",
              "assistant": "Anotado: Windows 11."},
             {"user": "Prefiero pathlib antes que os.path.",
              "assistant": "Entendido, uso pathlib en los ejemplos."}]},
         {"session_id": "s2", "date": "2026-05-19T16:45:00", "turns": [
             {"user": "Ahora el mismo script también corre en una Mac.",
              "assistant": "Anotado."},
             {"user": "Uso Python 3.12.", "assistant": "Anotado: 3.12."}]},
         {"session_id": "s3", "date": "2026-08-07T09:20:00", "turns": [
             {"user": "Ya pasé todo el proyecto a pathlib, menos un módulo viejo.",
              "assistant": "Anotado."}]},
     ],
     "probes": [
         P("p-so", "s1", "¿En qué sistema operativo trabajo?", "extraction", ev=["s1"],
           keys=["Windows"]),
         P("p-ruta", "s1", "¿Cómo armo la ruta a un subdirectorio?", "preference", ev=["s1"],
           rel="library/pathlib.html", slug="operators", keys=["/"],
           notes="correcta = usa pathlib, por la preferencia de s1"),
         P("p-portable", "s2", "Mi script corre en dos sistemas distintos, ¿cómo escribo las "
           "rutas una sola vez?", "multi_session", ev=["s1", "s2"],
           rel="library/pathlib.html", slug="module-pathlib", keys=["Path"]),
         P("p-ver-2", "s2", "¿Qué versión de Python uso?", "extraction", ev=["s2"],
           keys=["3.12"]),
         P("p-editor", "s3", "¿Qué editor uso para programar?", "abstention", answerable=False,
           notes="nunca se dijo"),
         P("p-ospath", "s3", "¿Sigo usando os.path en todo el proyecto?", "knowledge_update",
           ev=["s1", "s3"], keys=["pathlib"], vd="no",
           notes="el hecho que cambió entre s1 y s3"),
     ]},
    {"scenario_id": "s-turnos", "split": "test",
     "persona": "Lucía, ficticia: desarrolla un sistema de turnos médicos y cambia la política de "
                "almacenamiento de fechas en el medio.",
     "sessions": [
         {"session_id": "s1", "date": "2026-01-14T08:30:00", "turns": [
             {"user": "Tengo un sistema de turnos médicos y guardo las fechas en UTC.",
              "assistant": "Anotado: fechas en UTC."},
             {"user": "Los usuarios están todos en Argentina.", "assistant": "Anotado."}]},
         {"session_id": "s2", "date": "2026-06-02T13:00:00", "turns": [
             {"user": "Decidimos guardar también la zona horaria de cada turno, no solo UTC.",
              "assistant": "Anotado: ahora se guarda la zona."},
             {"user": "Uso Python 3.13.", "assistant": "Anotado: 3.13."}]},
     ],
     "probes": [
         P("p-zona", "s1", "¿En qué zona están mis usuarios?", "extraction", ev=["s1"],
           keys=["Argentina"]),
         P("p-utc", "s1", "¿Cómo guardo un instante en UTC con datetime?", "extraction",
           ev=["s1"], rel="library/datetime.html", slug="timezone-objects", keys=["utc"]),
         P("p-politica", "s2", "¿Sigo guardando solamente UTC?", "knowledge_update",
           ev=["s1", "s2"], keys=["zona"], vd="no",
           notes="el hecho que cambió entre s1 y s2"),
         P("p-zoneinfo", "s2", "¿Cómo le pongo a esa fecha la zona horaria de mis usuarios?",
           "multi_session", ev=["s1", "s2"], rel="library/zoneinfo.html",
           slug="using-zoneinfo", keys=["ZoneInfo"]),
         P("p-cuantos", "s2", "¿Cuántos turnos por día maneja el sistema?", "abstention",
           answerable=False, notes="nunca se dijo"),
         P("p-enero", "s2", "¿Qué habíamos decidido en enero sobre el almacenamiento de las "
           "fechas?", "temporal", ev=["s1", "s2"], keys=["UTC"]),
     ]},
    {"scenario_id": "s-backup", "split": "test",
     "persona": "Pablo, ficticio: tiene un script de backup en una Raspberry Pi y pasa de print "
                "a logging en el medio.",
     "sessions": [
         {"session_id": "s1", "date": "2026-02-28T19:00:00", "turns": [
             {"user": "Mi script de backup imprime todo con print y ya no entiendo la salida.",
              "assistant": "Anotado."},
             {"user": "Corro Python 3.12 en una Raspberry Pi.",
              "assistant": "Anotado: 3.12 en Raspberry Pi."}]},
         {"session_id": "s2", "date": "2026-04-10T07:40:00", "turns": [
             {"user": "Pasé todo a logging, con un archivo de registro.",
              "assistant": "Anotado: ahora usa logging."},
             {"user": "Quiero que ese archivo no crezca sin límite.",
              "assistant": "Se resuelve con rotación."}]},
         {"session_id": "s3", "date": "2026-07-21T21:10:00", "turns": [
             {"user": "El backup ahora corre en tres máquinas y quiero un registro único.",
              "assistant": "Anotado."}]},
     ],
     "probes": [
         P("p-ver-3", "s1", "¿Qué versión de Python uso?", "extraction", ev=["s1"],
           keys=["3.12"]),
         P("p-print", "s1", "¿Cómo reemplazo mis print por algo que pueda filtrar por nivel?",
           "preference", ev=["s1"], rel="howto/logging.html", slug="a-simple-example",
           keys=["logging"]),
         P("p-rotacion", "s2", "¿Cómo evito que el archivo de registro crezca sin límite?",
           "multi_session", ev=["s2"], rel="howto/logging.html", slug="useful-handlers",
           keys=["RotatingFileHandler"]),
         P("p-ya-no-print", "s2", "¿Sigo usando print para el registro?", "knowledge_update",
           ev=["s1", "s2"], keys=["logging"], vd="no",
           notes="el hecho que cambió entre s1 y s2"),
         P("p-maquinas", "s3", "¿Cómo se llaman las tres máquinas?", "abstention",
           answerable=False, notes="nunca se dijo"),
         P("p-desde-cuando", "s2", "¿Desde cuándo uso logging?", "temporal", ev=["s2"],
           keys=["abril", "2026-04"],
           notes="la respuesta sale del timestamp simulado de s2, no del reloj"),
     ]},
]

scenarios = [Scenario.model_validate(s) for s in SCENARIOS]
abilities = collections.Counter(p.ability for s in scenarios for p in s.probes)
print(f"escenarios: {len(scenarios)} · sesiones: {sum(len(s.sessions) for s in scenarios)} · "
      f"sondas: {sum(len(s.probes) for s in scenarios)}")
print_table("sondas por habilidad", ["sondas"],
            [(a, {"sondas": n}) for a, n in sorted(abilities.items(), key=lambda kv: -kv[1])],
            name_header="habilidad")
for s in scenarios:
    changed = [p.probe_id for p in s.probes if p.ability == "knowledge_update"]
    never = [p.probe_id for p in s.probes if not p.answerable]
    print(f"   {s.scenario_id}: cambia -> {changed or '¡FALTA!'} · nunca dicho -> "
          f"{never or '¡FALTA!'}")
print(verdict(all(any(p.ability == "knowledge_update" for p in s.probes)
                  and any(not p.answerable for p in s.probes) for s in scenarios),
              "cada escenario tiene un hecho que cambia y uno que nunca se dijo",
              "algún escenario no cumple el requisito del paso 4"))

# %% [markdown]
r"""
## 9 · Escribir, validar y congelar

El orden importa: `validate_eval` primero (resuelve cada id gold contra el corpus, a los dos
niveles), y después `freeze_eval`, que vuelve a validar y recién entonces escribe los hashes.

**Esto se commitea antes de que exista cualquier resultado.** De acá en adelante,
`load_eval("eval", documents_path=...)` se niega a cargar un conjunto editado **o** un corpus que
cambió por debajo, que es el modo de fallar que importa: los ids gold congelados contra un
`documents.jsonl` pueden apuntar a otro texto en el siguiente.

El corpus contra el que se congela ya cambió una vez por culpa de esta etapa: derivar las
preguntas de identificador mostró que la regla de firmas partidas de la Etapa 1 solo cubría la
forma calificada (`pathlib.Path. open`) y no la corta (`Path. open`), que es la que las páginas
muestran. Se arregló la regla, se volvió a construir el corpus, y recién entonces se congeló.
"""

# %%
EVAL.mkdir(parents=True, exist_ok=True)
items_v = [EvalItem.model_validate(r) for r in items]
store.write_jsonl(EVAL / "single_turn.jsonl", items_v)
store.write_jsonl(EVAL / "dialogues.jsonl", dialogues)
store.write_jsonl(EVAL / "scenarios.jsonl", scenarios)

report = validate_eval(EVAL, FULL / "documents.jsonl").show()
report.raise_if_failed()

# %%
frozen = freeze_eval(EVAL, FULL / "documents.jsonl")
print(f"\ndocuments_sha256 congelado: {frozen['documents_sha256']}")
print(json.dumps(frozen["counts"], ensure_ascii=False, indent=2))

# Que el candado cierre de verdad: cargar tiene que andar, y una edición tiene que fallar.
loaded = load_eval(EVAL, documents_path=FULL / "documents.jsonl")
print(f"\nload_eval: {len(loaded.items)} ítems, {len(loaded.dialogues)} diálogos, "
      f"{len(loaded.scenarios)} escenarios · key {loaded.key()[:16]}…")
print(f"dev: {len(loaded.split('dev').items)} ítems · test: {len(loaded.split('test').items)}")

# %% [markdown]
r"""
### 9.1 · Que el candado muerda

Un candado que nadie probó no es un candado. Se le agrega una fila al archivo, se verifica que
`load_eval` levante `EvalChanged`, y se restaura.
"""

# %%
from schema import EvalChanged

path = EVAL / "single_turn.jsonl"
backup = path.read_text(encoding="utf-8")
try:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"item_id": "no-deberia-estar", "question": "x",
                             "category": "topical", "gold": [], "answerable": False}) + "\n")
    try:
        load_eval(EVAL, documents_path=FULL / "documents.jsonl")
        print("CHECK -- el candado NO mordió: load_eval aceptó un archivo editado")
    except EvalChanged as e:
        print(f"OK -- el candado mordió: {clip(str(e), 150)}")
finally:
    path.write_text(backup, encoding="utf-8")
load_eval(EVAL, documents_path=FULL / "documents.jsonl")
print("restaurado: load_eval vuelve a cargar")

# %% [markdown]
r"""
## 10 · Procedencia: quién escribió cada ítem, y cómo se verificó

La trampa del enunciado dice que *«un ítem redactado por un modelo es una propuesta hasta que una
persona verificó la respuesta contra el corpus, y en el informe decís cuáles son esos ítems»*.
Esta tabla es esa lista. Las tres clases:

- **derivada**: la escribió una plantilla a partir del corpus; la respuesta está verificada por
  contención contra la sección gold, por código, en toda la corrida.
- **a mano (título)**: la redactó una persona desde el título de sección, con el pasaje cerrado;
  la clave está verificada por contención, y las que hubo que corregir están listadas.
- **a mano (3.14 adyacente)**: no tiene respuesta, y lo que se verificó es la **ausencia** del
  término en el corpus.

Lo que queda pendiente de ojos humanos, y no se puede automatizar: las 40 atribuciones de notas
de `stage2_note_owners.txt` (la heurística de posición), y la lectura de los diez documentos de la
Etapa 0.
"""

# %%
prov = collections.Counter(PROVENANCE.values())
print_table("procedencia de los ítems de un solo turno", ["ítems"],
            [(k, {"ítems": v}) for k, v in prov.most_common()], name_header="cómo se escribió")
print(f"\nclaves corregidas contra el vocabulario de la traducción: {len(KEY_FIXED)} "
      f"(de {len(TOPICAL)} verificadas por contención; ninguna falló)")
for line in KEY_FIXED:
    print(f"   {line}")

n_set = sum(1 for r in items if len(r["gold"]) > 1)
n_single = sum(1 for r in items if len(r["gold"]) == 1)
print(f"\ngold como conjunto (las dos versiones): {n_set} · de una sola versión: {n_single} · "
      f"sin gold: {sum(1 for r in items if not r['gold'])}")
print(f"fracción de los ítems con respuesta cuyo gold es un conjunto: "
      f"{n_set / max(1, n_set + n_single):.0%}")
if SINGLE_VERSION:
    print("\nlos de una sola versión, y por qué:")
    for line in sorted(set(SINGLE_VERSION)):
        print(f"   {line}")

# %% [markdown]
r"""
## 11 · `eval/smoke/`: para que las etapas siguientes se puedan correr sin medir nada

Los ítems smoke **no son una versión chica del conjunto**: son los pocos cuyo gold cae dentro de
las diez páginas del perfil smoke, para que las Etapas 3-6 arranquen en la laptop. Se cargan con
`require_frozen=False` y cualquier número que salga de ellos lleva el banner de smoke.
"""

# %%
SMOKE_PAGES = {"library/pathlib.html", "library/json.html", "library/itertools.html",
               "library/os.path.html", "library/aifc.html", "builtins/functions.html",
               "library/concurrent.html", "library/typing.html", "faq/general.html",
               "tutorial/introduction.html"}
smoke_docs = {d["doc_id"] for d in docs if d["metadata"]["rel"] in SMOKE_PAGES}


def inside_smoke(golds) -> bool:
    return bool(golds) and all(g["doc_id"] in smoke_docs for g in golds)


SMOKE_EVAL.mkdir(parents=True, exist_ok=True)
smoke_items = [r for r in items if inside_smoke(r["gold"])][:6]
smoke_items += [r for r in items if not r["gold"]][:1]      # uno sin respuesta, siempre
store.write_jsonl(SMOKE_EVAL / "single_turn.jsonl",
                  [EvalItem.model_validate(r) for r in smoke_items])
smoke_dialogues = [d for d in dialogues
                   if all(inside_smoke([g.model_dump() for g in t.gold]) or not t.answerable
                          for t in d.turns)]
if smoke_dialogues:
    store.write_jsonl(SMOKE_EVAL / "dialogues.jsonl", smoke_dialogues[:1])
smoke_scen = [s for s in scenarios
              if all(inside_smoke([g.model_dump() for g in p.gold]) or not p.gold
                     for p in s.probes)]
if smoke_scen:
    store.write_jsonl(SMOKE_EVAL / "scenarios.jsonl", smoke_scen[:1])
print(f"eval/smoke: {len(smoke_items)} ítems, {len(smoke_dialogues[:1])} diálogo, "
      f"{len(smoke_scen[:1])} escenario")
validate_eval(SMOKE_EVAL, environment.artifacts_dir(True) / "documents.jsonl").show()
print("sin FROZEN.json a propósito: se cargan con require_frozen=False")

# %% [markdown]
r"""
## 12 · Lo que este conjunto no mide

Las preguntas del informe, contestadas con los números de arriba, y la que no tiene número:

**Una pregunta que un usuario real haría y que este conjunto no puede medir.** «Mi script tarda
40 segundos en procesar 200 MB de CSV, ¿cómo lo hago más rápido?» No hay una sección que la
conteste: la respuesta sale de combinar tres o cuatro páginas (`csv`, `sqlite3`, `itertools`,
quizá `concurrent.futures`) con el detalle del código del usuario, que el corpus no tiene. Un gold
de sección es la unidad equivocada para esa pregunta, y un gold de documento también. Nuestro
conjunto solo mide preguntas cuya respuesta vive en un lugar.

**Si más adelante `topical` e `identifier` no coinciden sobre una técnica, en cuál confiar.** En
`topical`, y conviene decirlo ahora. Las de identificador son literales por construcción: miden
si el canal léxico encuentra una cadena exacta, y cualquier técnica que favorezca la coincidencia
literal va a ganar ahí sin que el sistema entienda nada. La pregunta del lab es si el asistente
contesta lo que una persona pregunta con sus palabras, y eso es `topical`. Las de identificador
quedan como control: si una técnica mejora `identifier` y empeora `topical`, lo que encontró es
el formato de las preguntas, no una mejora.
"""

# %%
run.note("single_turn", len(items))
run.note("by_category", dict(by_cat))
run.note("dialogues", len(dialogues))
run.note("dialogue_turns", sum(len(d.turns) for d in dialogues))
run.note("scenarios", len(scenarios))
run.note("probes_by_ability", dict(abilities))
run.note("bm25_only_hit_at_10", f"{total}/{len(answerable)}")
run.note("contamination_flagged", len(flagged))
run.note("gold_set_fraction", round(n_set / max(1, n_set + n_single), 3))
run.note("documents_sha256", frozen["documents_sha256"])
run.finish()
