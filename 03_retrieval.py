# %% [markdown]
r"""
# Etapa 3 · Recuperación sobre nuestro corpus

Siete filas, y ninguna más. La Entrada 1 del registro ya decidió por qué: *«una grilla completa
de la Etapa 3 y nada después»* es el recorte que el enunciado más discutiría, así que acá se
corren **dos líneas de base**, **tres filas acumulativas** y **dos ramas etiquetadas**, y el
tiempo que sobra se va a la Etapa 5.

| fila | qué cambia respecto de la anterior | carácter |
|---|---|---|
| solo BM25 | — | línea de base |
| solo denso | — (no es un escalón: cambia *dos* flags) | línea de base |
| + prefijo | `prefix` | acumulativa |
| + léxico (RRF) | `lexical` | acumulativa |
| + rerank | `rerank` | acumulativa |
| rama: + padre | `parent`, desde «+ rerank» | **rama**, no escalón |
| rama: + filtro de metadatos | `metadata_filter`, desde «+ rerank» | **rama**, no escalón |

**Por qué `+ padre` y `+ filtro` no son escalones.** Ninguna de las dos mejora el *ranking*. El
padre responde «con tokens igualados, ¿muchos hijos o pocos padres?», que es una pregunta sobre
**entrega**; y el hit a nivel documento no puede ni verla, porque el padre vive en el mismo
documento que el hijo (es la trampa que el enunciado nombra). El filtro de metadatos solo tiene
sentido sobre las preguntas de versión: promediarlo sobre 45 ítems lo diluye hasta que no se ve.
Las dos se miden contra «+ rerank» a **presupuesto de tokens igualado**.

**Entra**: `documents.jsonl` y el conjunto **congelado** — `load_eval` con `documents_path`, que
aborta si el corpus no es aquel contra el que se congeló.
**Sale**: `artifacts/<perfil>/index/` (hijos, padres, BM25 y manifiesto), los embeddings
cacheados en `artifacts/<perfil>/embeddings/`, y una fila por (brazo, ítem) en
`artifacts/<perfil>/results/03_retrieval.jsonl`.

**Dónde vive el recuperador.** En `retrieval.py`, no acá: lo usan también la Etapa 4 (le pone un
generador encima), la Etapa 5 (lo llama cuatro veces por turno) y la Etapa 6 (le pasa un filtro
de memoria). Tres copias de `retrieve` divergirían en el primer arreglo y esta tabla dejaría de
describir lo que corren las otras etapas. Esta etapa aporta las **filas**, el **piso de ruido** y
los **diagnósticos** del corte.

**Dónde corre.** El perfil `full` es para la 3050: `bge-m3` en fp16 (~2,3 GB) y el reranker
(~0,5 GB) entran sobrados porque acá no hay generador compitiendo por la placa. El perfil
`smoke` corre en CPU con un embebedor chico **y sus números no significan nada** — están para
que el código se ejercite, no para leerlos.
"""

# %%
import environment  # noqa: F401  (efectos secundarios: guarda de JAX, caché de HF, bundle TLS, UTF-8)

environment.summary()

import os
import time
from dataclasses import asdict

import numpy as np

import noise
import retrieval as rt
import store
from console import StageRun, clip, print_table, verdict
from retrieval import RetrievalConfig, answer_delivered
from schema import load_eval

SMOKE_TEST = os.environ.get("SMOKE_TEST", "1").lower() not in ("0", "false", "no")
ART = environment.artifacts_dir(SMOKE_TEST)
PROFILE = "smoke" if SMOKE_TEST else "full"
EVAL = environment.ROOT / ("eval/smoke" if SMOKE_TEST else "eval")
MODELS = rt.models(SMOKE_TEST)

BUDGET_TOLERANCE = 0.10   # dos filas que difieran más que esto no están comparando lo mismo
BOOTSTRAP = 2000
SEED = 0

environment.seed_everything(SEED)

run = StageRun.start("03_retrieval", ART, profile=PROFILE, config={
    **MODELS, "child_tokens": rt.CHILD_TOKENS, "child_overlap": rt.CHILD_OVERLAP,
    "parent_tokens": rt.PARENT_TOKENS, "candidates": rt.CANDIDATES, "rrf_k": rt.RRF_K,
    "budget_tokens": rt.BUDGET_TOKENS, "seed": SEED,
})

DOCS_PATH = ART / "documents.jsonl"

# El candado de la Etapa 2, del lado que lo usa: si el corpus no es el que se congeló, los ids
# gold apuntan a otro texto y toda esta tabla mide otra cosa. En smoke no hay candado (el
# conjunto de `eval/smoke` se escribió sin congelar, a propósito).
ev = load_eval(EVAL, require_frozen=not SMOKE_TEST,
               documents_path=None if SMOKE_TEST else DOCS_PATH)
ITEMS = ev.items
TEST = [i for i in ITEMS if i.split == "test"]
print(f"conjunto: {len(ITEMS)} ítems ({len(TEST)} de test, {len(ITEMS) - len(TEST)} de dev)")
print(f"embebedor {MODELS['dense']} · reranker {MODELS['rerank']} · "
      f"generador {MODELS['generator']}")

# %% [markdown]
r"""
## 1 · El corte, y qué hay que mirarle

`rt.load` hace tres cosas: el **texto propio** de cada sección, los **hijos y padres**, y el
**índice con su manifiesto**. Las tres están en el módulo; lo que esta etapa tiene que hacer es
mirarles los números, porque cada una tiene una forma silenciosa de salir mal.

- **Texto propio**: las secciones son un árbol y sus spans se *anidan* (el de la raíz cubre el
  documento entero). Si el corte usara los spans tal cual, el mismo párrafo entraría en cinco
  chunks de cinco niveles y la tabla de abajo mediría cuántas copias del texto llegaron. El
  chequeo es de conteo: los caracteres indexables no pueden pasar los del corpus.
- **Hijo** (350 tokens **del reranker**, prefijo incluido, 15 % de solapamiento) y **padre**
  (1.200 tokens **del generador**). Cada uno medido en el tokenizer que lo va a leer: el hijo
  porque el cross-encoder trunca en 512 **sin avisar**, el padre porque es el que paga la
  ventana. Un hijo que ya solo pasa 512 es un pasaje mutilado, y la fila «+ rerank» mediría eso.
- **Prefijo contextual**, de los metadatos: título del documento · ruta de títulos · versión. Es
  parte del *texto* del pasaje (`bge-m3` no usa prefijos de instrucción), y por eso la fila
  «+ prefijo» necesita una **matriz nueva** — `store.cached_matrix` la genera sola porque los
  textos entran en la clave de caché.
"""

# %%
R = rt.load(smoke=SMOKE_TEST, art=ART, docs_path=DOCS_PATH)
children, parents, S = R.children, R.parents, R.stats

print(f"\ncaracteres indexables {S['indexable_chars']:,} vs {S['corpus_chars']:,} del corpus")
print(verdict(S["indexable_chars"] <= S["corpus_chars"] + 2 * S["units"],
              "el texto propio no duplica nada: cada carácter del corpus está en una sola unidad",
              "los tramos se solapan -- hay párrafos indexados dos veces"))
print(f"ejemplo de prefijo: {clip(children[0]['prefix'], 120)}")

print_table("el índice, en unidades", ["cuántos", "tokens: mediana", "tokens: máx", "contados en"],
            [("hijos", {"cuántos": f"{len(children):,}",
                        "tokens: mediana": int(np.median(S["child_rr"])),
                        "tokens: máx": max(S["child_rr"]),
                        "contados en": "reranker, con prefijo"}),
             ("padres", {"cuántos": f"{len(parents):,}",
                         "tokens: mediana": int(np.median([p["n_gen"] for p in parents.values()])),
                         "tokens: máx": max(p["n_gen"] for p in parents.values()),
                         "contados en": "generador"})],
            smoke=SMOKE_TEST, name_header="unidad")
print(f"hijos que ya solos pasan los {rt.RERANK_MAX_LEN} tokens del reranker: {S['over_window']} "
      f"({S['over_window'] / len(children):.1%}) -- §9 los cuenta otra vez con la consulta adelante")

# %% [markdown]
r"""
## 2 · El índice y su manifiesto

El canal léxico usa **`lexical.tokenize`**, el mismo del chequeo de contaminación de la Etapa 2.
No es prolijidad: un chequeo que tokeniza distinto que BM25 mide un solapamiento distinto del
que infla a BM25, y entonces el chequeo dice que no hay contaminación mientras BM25 cobra las
frases copiadas.

El manifiesto del índice guarda el hash del corpus, los parámetros del chunker y el embebedor
(id, revisión, `max_seq_length`). Cualquier etapa que lo cargue sabe contra qué corpus se
construyó, que es lo que hace verificable el traspaso 3050 → Colab. El embebedor va ahí como
dato **descriptivo** y no en la clave de frescura: los hijos, los padres y BM25 no dependen de
él, y si estuviera en la clave, correr lo mismo en la otra máquina (fp16 vs fp32) re-cortaría el
corpus entero para llegar a los mismos hijos. De la matriz, que sí depende del embebedor, se
ocupa `store.cached_matrix`, que además hashea los textos.
"""

# %%
manifest = store.read_manifest(ART / "index")
print(f"índice {R.index_key[:16]} en {ART / 'index'} · construido {manifest['created_at']}")
print(f"   corpus      {manifest['spec']['corpus_sha256'][:16]}")
print(f"   chunker     {manifest['spec']['chunker']}")
print(f"   embebedor   {manifest['embedder']}")
print(f"   unidades    {manifest['n_children']:,} hijos · {manifest['n_parents']:,} padres")

# %% [markdown]
r"""
## 3 · Las métricas, y qué mide cada una

| métrica | qué contesta | por qué no alcanza sola |
|---|---|---|
| hit@entregados (documento) | ¿llegó **algún** pasaje del documento gold? | la trampa: llega el encabezado, la respuesta está diez secciones más adelante |
| MRR@20 (documento) | ¿en qué puesto llegó? | mismo problema, con decimales |
| hit@entregados (sección) | ¿llegó la **sección** donde está la respuesta? | no dice si la frase entró en el presupuesto |
| respuesta entregada | ¿la **clave** está en el texto que se le va a dar al generador? | es la única que la Etapa 4 puede alcanzar |
| tokens entregados | ¿las filas se comparan a presupuesto igual? | — |

El acierto de sección se mide **por contención de spans**, no por igualdad de `section_id`: si el
gold es una sección y el hijo salió de una subsección suya, el hijo está adentro del gold y
cuenta. Con igualdad de ids ese caso sería un fallo por una decisión de corte, no por
recuperación.

La **entrega es por presupuesto de tokens**, no por k: «el mismo k no es el mismo presupuesto»,
y dos filas con el mismo k pueden entregar un 40 % más de texto una que otra. El chequeo de abajo
compara los tokens entregados de todas las filas contra una tolerancia del 10 %.
"""


# %%
def measure(item, cfg: RetrievalConfig) -> dict:
    trace: dict = {}
    t = time.perf_counter()
    delivered = R.retrieve(item.question, cfg, trace=trace)
    dt = time.perf_counter() - t
    return {"item_id": item.item_id, "category": item.category,
            "doc_hit": bool({g.doc_id for g in item.gold} & {u["doc_id"] for u in delivered}),
            "sec_hit": R.section_hit(delivered, item),
            "mrr": R.mrr_doc(trace["ranked"], item),
            "delivered_key": answer_delivered(delivered, item),
            "tokens": sum(u["n_gen"] for u in delivered), "units": len(delivered),
            "latency_s": round(dt, 3), "top_score": trace.get("top_score", {}),
            "ranked": trace["ranked"][:10],
            "channels": {k: v[:10] for k, v in trace.get("channels", {}).items()},
            "filter": trace.get("filter")}


# %% [markdown]
r"""
## 4 · Las filas

Dos líneas de base y tres escalones. `rt.diff_rows` imprime, para cada escalón, qué variable
cambió — y avisa si cambió más de una, que es lo que convertiría la escalera en una grilla mal
leída. El otro lado de lo mismo está en `retrieve`: **cada flag que pide algo que no está
cargado levanta una excepción**, porque una pieza que avisa y sigue produce una fila idéntica a
la anterior y uno lee «sin efecto» en una tabla que nunca corrió la pieza.
"""

# %%
ARMS: list[tuple[str, RetrievalConfig, str]] = [
    ("solo BM25", RetrievalConfig(dense=False, lexical=True), "base"),
    ("solo denso", RetrievalConfig(dense=True), "base"),
    ("+ prefijo", RetrievalConfig(dense=True, prefix=True), "escalón"),
    ("+ léxico (RRF)", RetrievalConfig(dense=True, prefix=True, lexical=True), "escalón"),
    ("+ rerank", RetrievalConfig(dense=True, prefix=True, lexical=True, rerank=True), "escalón"),
    ("rama: + padre", RetrievalConfig(dense=True, prefix=True, lexical=True, rerank=True,
                                      parent=True), "rama"),
    ("rama: + filtro de metadatos", RetrievalConfig(dense=True, prefix=True, lexical=True,
                                                    rerank=True, metadata_filter=True), "rama"),
]
BEST = "+ rerank"       # la punta de la escalera: las dos ramas se comparan contra ella

CFG = {name: cfg for name, cfg, _ in ARMS}

print("qué cambia cada fila")
for (pn, _, _), (n, c, kind) in zip(ARMS, ARMS[1:]):
    base = BEST if kind == "rama" else pn
    print(f"   {n:28} vs {base:16} -> {rt.diff_rows(CFG[base], c)}")

# %%
log = store.ResultsLog(ART / "results" / "03_retrieval.jsonl",
                       run_spec={"stage": "03_retrieval", "eval": ev.key(),
                                 "index": R.index_key,
                                 "budget_tokens": rt.BUDGET_TOKENS,
                                 "candidates": rt.CANDIDATES})

for name, cfg, kind in ARMS:
    t0 = time.perf_counter()
    todo = [it for it in ITEMS if not log.done(name, it.item_id)]
    for it in todo:
        row = measure(it, cfg)
        log.write({"arm": name, "config": asdict(cfg), **row})
    print(f"{name:28} {len(todo):3} ítems en {time.perf_counter() - t0:6.1f} s")

ROWS = {name: {r["item_id"]: r for r in log.rows(name)} for name, _, _ in ARMS}
TEST_IDS = [it.item_id for it in TEST]
WITH_GOLD = [it.item_id for it in TEST if it.gold]
WITH_KEY = [it.item_id for it in TEST if it.answer_key and it.gold]


def frac(name: str, ids: list[str], field: str) -> float:
    vals = [ROWS[name][i][field] for i in ids if i in ROWS[name]]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


table = []
for name, cfg, kind in ARMS:
    rows = [ROWS[name][i] for i in TEST_IDS if i in ROWS[name]]
    table.append((name, {
        "carácter": kind,
        "hit doc": f"{frac(name, WITH_GOLD, 'doc_hit'):.0%}",
        "MRR@20": f"{np.mean([ROWS[name][i]['mrr'] for i in WITH_GOLD]):.3f}",
        "hit sec": f"{frac(name, WITH_GOLD, 'sec_hit'):.0%}",
        "resp. entregada": f"{frac(name, WITH_KEY, 'delivered_key'):.0%}",
        "tokens": int(np.mean([r["tokens"] for r in rows])),
        "unid.": round(float(np.mean([r["units"] for r in rows])), 1),
        "seg/consulta": round(float(np.mean([r["latency_s"] for r in rows])), 2),
    }))
print_table(f"las filas · {len(WITH_GOLD)} ítems de test con gold, presupuesto "
            f"{rt.BUDGET_TOKENS} tokens",
            ["carácter", "hit doc", "MRR@20", "hit sec", "resp. entregada", "tokens", "unid.",
             "seg/consulta"], table, smoke=SMOKE_TEST, name_header="fila")

tokens = {name: np.mean([ROWS[name][i]["tokens"] for i in TEST_IDS if i in ROWS[name]])
          for name, _, _ in ARMS}
lo, hi = min(tokens.values()), max(tokens.values())
print(verdict(hi - lo <= BUDGET_TOLERANCE * hi,
              f"todas las filas entregaron entre {lo:.0f} y {hi:.0f} tokens: el presupuesto está igualado",
              f"las filas difieren en {(hi - lo) / hi:.0%} de tokens entregados (tolerancia "
              f"{BUDGET_TOLERANCE:.0%}): parte de la diferencia es volumen, no técnica"))

# %%
cats = sorted({it.category for it in TEST if it.gold})
print_table("hit a nivel sección, por categoría", cats + ["total"],
            [(name, {**{c: f"{frac(name, [i.item_id for i in TEST if i.category == c and i.gold], 'sec_hit'):.0%}"
                        for c in cats},
                     "total": f"{frac(name, WITH_GOLD, 'sec_hit'):.0%}"})
             for name, _, _ in ARMS],
            smoke=SMOKE_TEST, name_header="fila")

gap_doc, gap_sec, gap_key = (frac(BEST, WITH_GOLD, "doc_hit"), frac(BEST, WITH_GOLD, "sec_hit"),
                             frac(BEST, WITH_KEY, "delivered_key"))
print(f"\nla brecha de la trampa, en «{BEST}»: documento {gap_doc:.0%} -> sección {gap_sec:.0%} "
      f"-> respuesta entregada {gap_key:.0%}")
print(f"lo que predice para la Etapa 4: el techo de la generación es {gap_key:.0%}, no {gap_doc:.0%}. "
      f"Las respuestas «no está en el contexto» que pasen de ese {1 - gap_key:.0%} son de generación; "
      "las de abajo ya estaban perdidas acá.")

# %% [markdown]
r"""
## 5 · El piso de ruido, antes de afirmar nada

Con ~45 resultados binarios cerca del 85 %, el intervalo del 95 % de una sola fila abarca unos
±10 puntos. Así que una fila contra otra se lee **pareada**: solo los ítems **discordantes**
—los que cambiaron de resultado— llevan información, y el test de signos bilateral sobre ellos
da `p = 2 × 0.5ⁿ` cuando todos se mueven para el mismo lado. Un 5-0 da 0,06; un 6-0, 0,03.
"""


# %%
def paired(a: str, b: str, ids: list[str], field: str) -> dict:
    return noise.paired([ROWS[a][i][field] for i in ids], [ROWS[b][i][field] for i in ids])


def boot_ci(values: list[int]) -> tuple[float, float]:
    return noise.boot_ci(values, reps=BOOTSTRAP, seed=SEED)


for field, ids, label in (("sec_hit", WITH_GOLD, "hit a nivel sección"),
                          ("delivered_key", WITH_KEY, "respuesta entregada")):
    rows = []
    for (pn, _, _), (n, _, kind) in zip(ARMS, ARMS[1:]):
        base = BEST if kind == "rama" else pn
        p = paired(base, n, ids, field)
        ci = boot_ci([int(bool(ROWS[n][i][field])) for i in ids if ROWS[n][i][field] is not None])
        rows.append((f"{n} vs {base}", {**p, "IC95 de la fila": f"{ci[0]:.0%}-{ci[1]:.0%}",
                                        "¿sobrevive?": "sí" if p["p"] < 0.05 else "no"}))
    print_table(f"pareado · {label} · n = {len(ids)}",
                ["gana", "pierde", "empata", "p", "IC95 de la fila", "¿sobrevive?"],
                rows, smoke=SMOKE_TEST, name_header="comparación")

survivors = [n for (pn, _, _), (n, _, kind) in zip(ARMS, ARMS[1:])
             if paired(BEST if kind == "rama" else pn, n, WITH_GOLD, "sec_hit")["p"] < 0.05]
print(f"\npiezas que se ganaron su costo por encima del piso de ruido: "
      f"{survivors or 'ninguna'}")
print(verdict(True,
              "un resultado débil que podemos acotar vale más que uno fuerte que no: las filas "
              "que no sobreviven al pareado se informan como lo que son, no como mejoras",
              ""))

# %% [markdown]
r"""
## 6 · Abrir la caja en el movimiento más grande

RRF **supone que los canales fallan de forma independiente**. Cuando una consulta engaña a los
dos de la misma manera, el acuerdo rankea más alto los pasajes equivocados y el consenso entierra
el correcto. Así que antes de acreditarle (o culparle) la fila a la pieza, se miran los rankings
**por canal** de las preguntas que cambiaron de resultado.
"""

# %%
moves = []
for (pn, _, _), (n, _, kind) in zip(ARMS, ARMS[1:]):
    if kind == "base":
        continue        # base contra base cambia dos flags: no hay una pieza a la que abrirle la caja
    base = BEST if kind == "rama" else pn
    p = paired(base, n, WITH_GOLD, "sec_hit")
    moves.append((abs(p["gana"] - p["pierde"]), n, base, p))
_, big, big_base, big_p = max(moves)
print(f"el movimiento más grande: {big} vs {big_base} ({big_p['gana']} ganadas, "
      f"{big_p['pierde']} perdidas)")

flipped = [i for i in WITH_GOLD
           if bool(ROWS[big][i]["sec_hit"]) != bool(ROWS[big_base][i]["sec_hit"])]
item_by_id = {it.item_id: it for it in ITEMS}
for iid in flipped[:6]:
    it = item_by_id[iid]
    won = ROWS[big][iid]["sec_hit"]
    print(f"\n{'GANÓ ' if won else 'PERDIÓ'} {iid}: {clip(it.question, 90)}")
    print(f"   gold: {[g.section_id for g in it.gold][:2]}")
    for ch, ranking in ROWS[big][iid].get("channels", {}).items():
        marks = ["*" if R.in_gold_section(R.child(c)["section_id"], it) else "."
                 for c in ranking[:10] if c in R.by_chunk]
        print(f"   {ch:6} {''.join(marks):10} top1: {clip(R.child(ranking[0])['section_id'], 60)}")
if not flipped:
    print("   ninguna pregunta cambió de resultado: la pieza no movió nada que podamos abrir")

# %% [markdown]
r"""
## 7 · La ventana del reranker

El cross-encoder lee `max_length` tokens de (consulta + hijo) y **descarta el resto en silencio**.
Si una parte grande de los pares la pasa, el reranker está puntuando pasajes mutilados y la fila
«+ rerank» mide eso. Se cuenta en el tokenizer **del propio reranker**, no en el del generador.
"""

# %%
pairs_over, pairs_total, longest = 0, 0, 0
for it in TEST:
    cand = ROWS[BEST][it.item_id]["ranked"][:10]
    texts = [R.child_text(c) for c in cand if c in R.by_chunk]
    enc = R.rerank_tokenizer([it.question] * len(texts), texts,
                             add_special_tokens=True)["input_ids"]
    lens = [len(e) for e in enc]
    pairs_total += len(lens)
    pairs_over += sum(n > rt.RERANK_MAX_LEN for n in lens)
    longest = max([longest] + lens)
print(f"pares (consulta + hijo) que pasan los {rt.RERANK_MAX_LEN} tokens: {pairs_over} de "
      f"{pairs_total} ({pairs_over / max(pairs_total, 1):.1%}) · el más largo: {longest}")
print(verdict(pairs_over / max(pairs_total, 1) < 0.05,
              "casi ningún par se trunca: el reranker está leyendo el hijo entero",
              "una parte de los pares se trunca en silencio -- la fila «+ rerank» puntúa "
              "pasajes cortados, y bajar CHILD_TOKENS es la palanca"))

# %% [markdown]
r"""
## 8 · ¿Puede la recuperación sola decir que la respuesta no está?

Los 5 ítems fuera del corpus no tienen gold, así que no entran en ninguna fila. Lo que sí
permiten es mirar la **distribución del puntaje máximo**: si las preguntas sin respuesta
puntuaran claramente más bajo que las que sí la tienen, un umbral sería un detector de
abstención gratis, antes de gastar una llamada al generador.
"""

# %%
OUT = [it.item_id for it in ITEMS if it.category == "out_of_corpus"]
IN = [it.item_id for it in ITEMS if it.gold]
for ch in ("dense", "rerank"):
    a = [ROWS[BEST][i]["top_score"].get(ch) for i in IN
         if ROWS[BEST][i]["top_score"].get(ch) is not None]
    b = [ROWS[BEST][i]["top_score"].get(ch) for i in OUT
         if ROWS[BEST][i]["top_score"].get(ch) is not None]
    if not a or not b:
        continue
    overlap = sum(1 for x in b if x >= np.percentile(a, 25)) / len(b)
    print(f"{ch:7} con respuesta: mediana {np.median(a):7.3f} · fuera del corpus: "
          f"mediana {np.median(b):7.3f} · fuera que superan el P25 de las de adentro: {overlap:.0%}")
print(verdict(False, "", "con 5 ítems fuera del corpus esto es una mirada, no una medición: la "
                        "lista de arriba va al informe como observación y la abstención se mide "
                        "en la Etapa 4, donde el generador puede decir que no sabe"))

# %% [markdown]
r"""
## 9 · Las respuestas del informe

**¿Qué piezas se ganaron su costo?** Las que sobreviven al pareado están arriba, en la lista de
`survivors`; las demás se informan como lo que dieron, no como mejoras. El criterio es el del
enunciado: con ~45 ítems, «26/30 → 28/30» no es evidencia de nada.

**¿Qué mostró abrir la caja?** §6 imprime, para el movimiento más grande, los rankings por canal
de cada pregunta que cambió. Lo que hay que mirar ahí es si el canal denso y BM25 fallaron
*juntos* (RRF no puede arreglar eso, y su acuerdo lo empeora) o si fallaron distinto (que es la
hipótesis bajo la que RRF funciona).

**¿Qué tan separados están el documento y la sección?** La brecha está impresa al final de §4, y
es la predicción para la Etapa 4: el techo que la generación puede alcanzar es el de *respuesta
entregada*, no el de documento.

**¿Muchos hijos o pocos padres?** Es la rama `+ padre`, a tokens igualados: misma tabla, misma
columna de tokens. El hit a nivel documento **no puede** contestarlo —el padre está en el mismo
documento que el hijo—, así que la columna que decide es *respuesta entregada*.

**¿Ayudó el filtro de metadatos?** Solo puede ayudar donde la pregunta nombra una versión; en el
resto no se dispara. El conteo de cuántas veces se disparó está en las filas de resultado
(`filter`), y la categoría donde podría verse es `metadata`.
"""

# %%
fired = [r for r in log.rows("rama: + filtro de metadatos")
         if (r.get("filter") or {}).get("version")]
print(f"el filtro de metadatos se disparó en {len(fired)} de {len(ITEMS)} preguntas "
      f"(las que nombran una versión que el corpus tiene)")
if fired:
    kept = np.mean([f["filter"]["kept"] / max(f["filter"]["of"], 1) for f in fired])
    print(f"   cuando se disparó dejó, en promedio, el {kept:.0%} de los candidatos")

store.write_json(ART / "stage3_rows.json", {
    "arms": [{"name": n, "config": asdict(c), "kind": k} for n, c, k in ARMS],
    "table": {n: v for n, v in table},
    "survivors": survivors,
    "best": BEST,
    "gap": {"doc": gap_doc, "section": gap_sec, "answer_delivered": gap_key},
    "reranker_truncation": {"over": pairs_over, "total": pairs_total},
    "index_key": R.index_key,
})

run.note("hijos", len(children))
run.note("padres", len(parents))
run.note("ítems de test con gold", len(WITH_GOLD))
run.note("mejor fila", BEST)
run.note("hit sección (mejor fila)", round(gap_sec, 3))
run.note("respuesta entregada (mejor fila)", round(gap_key, 3))
run.note("piezas que sobreviven al pareado", survivors)
run.note("pares truncados por el reranker", f"{pairs_over}/{pairs_total}")
run.finish()
