# %% [markdown]
r"""
# Etapa 5 · La conversación: una ventana, dos fuentes

La API **no tiene estado**: todo lo que el modelo «recuerda» dentro de una sesión es lo que este
código vuelve a poner en la ventana en cada turno. Y una vez que hay conversación, dos cosas
compiten por esa ventana —los pasajes del corpus y el historial—, así que cada turno tiene las dos
superficies de falla del RAG de un solo turno:

- **con qué busca el recuperador** (Parte A: cuatro consultas sobre los mismos guiones);
- **qué ve el generador** (Parte B: el mismo total de tokens, repartido 80/20, 50/50 y 30/70).

**Entra**: los guiones congelados de la Etapa 2 (`eval/dialogues.jsonl`), el recuperador de
`retrieval.py` y el ensamblador de `prompting.py` —los mismos que corrió la Etapa 4, para que las
filas se comparen con las de allá. **Sale**:
`artifacts/<perfil>/results/05_dialogue_{a,b}.jsonl`, `stage5_rows.json`, y dos archivos para leer
a mano: `stage5_rewrites.txt` y `stage5_summaries.txt`.

**Las cuatro decisiones de diseño que esta etapa no puede no tomar**, y por qué:

1. **El ensamblador decide qué entra, no el servidor.** Pasado `num_ctx`, Ollama descarta los
   mensajes más viejos y lo registra en nivel debug: un brazo de «historial completo» se vuelve en
   secreto «sistema + los últimos turnos», y la comparación completo/ventana/resumen queda
   confundida. Acá cada turno se arma dentro de un presupuesto verificado y `prompt_eval_count`
   lo confirma (§6).
2. **El turno más nuevo va PRIMERO en la consulta de conversación completa.** El embebedor trunca
   por la derecha en `DENSE_MAX_SEQ`; una concatenación cronológica pierde justo el turno que dice
   qué se está preguntando *ahora*. Eso reproduce el hallazgo de MTRAG por una razón que no tiene
   nada que ver con el hallazgo. §3 mide cuántos tokens mide cada consulta en el tokenizer del
   embebedor, y muestra en qué punto empieza a importar.
3. **Historiales guionados (teacher forcing).** Todos los brazos ven el *mismo* historial —las
   respuestas guionadas del asistente, de la Etapa 2— y solo se genera el turno actual. Si cada
   brazo se realimentara con sus propias respuestas, divergirían desde el turno 2 y toda
   comparación posterior mediría la divergencia.
4. **Lo que va al historial es la `answer`, no el `analysis`.** El campo de análisis es
   razonamiento, y el usuario nunca lo vio.

**El ruido se acota por guion, no por turno**: 36 turnos de 9 guiones no son 36 muestras.
"""

# %%
import environment  # noqa: F401  (efectos secundarios: guarda de JAX, caché de HF, bundle TLS, UTF-8)

environment.summary()

import collections
import os
import time

import numpy as np
from pydantic import BaseModel, Field

import llm
import noise
import prompting as pr
import retrieval as rt
import store
from console import StageRun, clip, print_table, verdict, wrap
from retrieval import RetrievalConfig
from schema import load_eval

SMOKE_TEST = os.environ.get("SMOKE_TEST", "1").lower() not in ("0", "false", "no")
ART = environment.artifacts_dir(SMOKE_TEST)
PROFILE = "smoke" if SMOKE_TEST else "full"
EVAL = environment.ROOT / ("eval/smoke" if SMOKE_TEST else "eval")
MODELS = rt.models(SMOKE_TEST)
MODEL = MODELS["generator"]

NUM_CTX = 8192          # fijo en TODOS los brazos: cada cambio recarga el modelo (y la latencia
NUM_PREDICT = 700       # de un brazo pasaría a contar recargas en vez de generación)
TOTAL_TOKENS = 1400     # pasajes + historial: dimensionado por NUESTRO historial, no por §7.3
SPLITS = {"80/20": 0.80, "50/50": 0.50, "30/70": 0.30}   # la fracción que se lleva PASAJES
STRATEGY_SPLIT = "80/20"   # la división donde se comparan las estrategias (ver §6)
WINDOW_TURNS = 2        # la ventana: los últimos N turnos (2 mensajes por turno). Ver §5.
SUMMARY_AT = 0.70       # el umbral de resumen de MemGPT, como fracción del presupuesto de historial
SUMMARY_KEEP = 4        # mensajes recientes que sobreviven a un resumen
REWRITE_TURNS = 2       # el «historial reciente» que ve el reescritor
TOPK = 10               # la k de hit@k, sobre la lista rankeada ANTES del presupuesto
SEED = 0

environment.seed_everything(SEED)
run = StageRun.start("05_dialogue", ART, profile=PROFILE, config={
    "model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "total": TOTAL_TOKENS,
    "splits": SPLITS, "window_turns": WINDOW_TURNS, "summary_at": SUMMARY_AT, "topk": TOPK,
    "seed": SEED,
})

ev = load_eval(EVAL, require_frozen=not SMOKE_TEST,
               documents_path=None if SMOKE_TEST else ART / "documents.jsonl")
DIALOGUES = [d for d in ev.dialogues if d.split == "test"]
DEV_DIALOGUES = [d for d in ev.dialogues if d.split == "dev"]
N_TURNS = sum(len(d.turns) for d in DIALOGUES)
DEPS = collections.Counter(t.depends_on for d in DIALOGUES for t in d.turns)
print(f"guiones de test: {len(DIALOGUES)} · {N_TURNS} turnos · {len(DEV_DIALOGUES)} guiones de dev")
print(f"dependencias: {dict(DEPS)}")
print(f"turnos sin respuesta en el corpus: "
      f"{sum(1 for d in DIALOGUES for t in d.turns if not t.answerable)}")

# %%
llm.use_cache(ART / "llm_cache" / f"{MODEL.replace(':', '-')}.jsonl")
info = llm.setup(MODEL, num_ctx=NUM_CTX)
run.note("sondas del servidor", {k: info.get(k) for k in ("server", "digest", "placement")})

R = rt.load(smoke=SMOKE_TEST, art=ART, docs_path=ART / "documents.jsonl")
BEST_CFG = RetrievalConfig(dense=True, prefix=True, lexical=True, rerank=True)
try:
    _s3 = store.read_json(ART / "stage3_rows.json")
    print(f"la Etapa 3 dejó como mejor fila: {_s3['best']}")
except FileNotFoundError:
    print("sin stage3_rows.json en este perfil: se usa «+ rerank» por defecto")


def count(text: str) -> int:
    """Tokens del GENERADOR, texto pelado. Los marcadores de rol no están acá: por eso el total
    que imponemos se verifica después contra el `prompt_eval_count` del servidor (§6)."""
    return llm.count_tokens(text, MODEL) or 0


def dense_count(text: str) -> int:
    """Tokens del EMBEBEDOR. Es el conteo que decide si una consulta se trunca por la derecha, y
    no se puede aproximar con el del generador: son vocabularios distintos."""
    return len(R.encoder.tokenizer(text, add_special_tokens=False)["input_ids"])


# El historial sirve para entender la pregunta; los hechos siguen saliendo solo de los pasajes.
SYSTEM = pr.SYSTEM + ("\n5. El historial de la conversación sirve para entender a qué se refiere "
                      "la pregunta (pronombres, elipsis). Los HECHOS salen solo de los pasajes.")

# %% [markdown]
r"""
## 1 · Los guiones, como historial

Cada turno tiene su propio gold, su `answer_key` y su etiqueta de dependencia, y arrastra el
historial **guionado** de los turnos anteriores: `user` y la respuesta `assistant` que escribió la
Etapa 2. La respuesta guionada es corta a propósito —es lo que un asistente habría contestado—, y
es lo que hace que los brazos de la Parte B sean comparables.

`REPLAY` es la lista de turnos puntuados **uno detrás de otro, como una sola conversación larga**:
el historial del turno *k* incluye los guiones anteriores completos. Eso es lo que hace que el lado
del historial de cada división sea una restricción real: un guion suelto son unos 100 tokens de
historial en su último turno, y contra cualquiera de los tres presupuestos de §4 las tres
estrategias conservarían todo y la tabla diría «el historial no importa» cuando lo que pasó es que
nunca se llenó. Concatenados llegan a unos 900, y entonces la división decide algo.

El gold y el `dialogue_id` viajan con cada turno: la conversación se concatena, el puntaje no.
"""


# %%
def dialogue_history(d, upto: int) -> list[dict]:
    """El historial DENTRO del guion, hasta (sin incluir) el turno `upto` (1-based)."""
    out = []
    for t in d.turns:
        if t.turn >= upto:
            break
        out.append({"role": "user", "content": t.user})
        if t.assistant:
            out.append({"role": "assistant", "content": t.assistant})
    return out


ORDER = sorted(DIALOGUES, key=lambda d: d.dialogue_id)   # un orden fijo: el replay es reproducible

REPLAY: list[tuple] = []        # (dialogue, turn, historial del replay, item_id)
_hist: list[dict] = []
for d in ORDER:
    for t in d.turns:
        REPLAY.append((d, t, list(_hist), f"{d.dialogue_id}#{t.turn}"))
        _hist.append({"role": "user", "content": t.user})
        if t.assistant:
            _hist.append({"role": "assistant", "content": t.assistant})

print(f"replay: {len(REPLAY)} turnos puntuados en una sola conversación de "
      f"{len(_hist)} mensajes ({count(' '.join(m['content'] for m in _hist)):,} tokens en total)")
print(f"historial del último turno: {count(' '.join(m['content'] for m in REPLAY[-1][2])):,} "
      f"tokens · del último turno de su guion, suelto: "
      f"{count(' '.join(m['content'] for m in dialogue_history(ORDER[-1], ORDER[-1].turns[-1].turn))):,}")

# %% [markdown]
r"""
## 2 · Parte A · Con qué buscar

Cuatro consultas sobre los **mismos** turnos y el **mismo** recuperador (`+ rerank`, la mejor fila
de la Etapa 3). Lo único que cambia es el texto que se le manda:

- **último turno**: `turn.user` tal cual. La línea de base, y la que gana en MTRAG.
- **conversación completa**: el turno actual **primero** y después los turnos del usuario en orden
  inverso. Decisión declarada: solo los turnos del *usuario* —meter las respuestas del asistente
  mete el vocabulario de la respuesta en la consulta, que es otro experimento (HyDE).
- **reescritura del modelo**: un LLM condensa el último turno a partir de los
  `REWRITE_TURNS` turnos anteriores, con un schema de **un solo campo**. Un solo campo es
  deliberado: sin campo de análisis no hay dónde razonar, y sin dónde razonar es más difícil que
  el reescritor se ponga a *responder* —que es la falla de Ye et al.
- **reescritura oráculo**: la escrita a mano en la Etapa 2 (`oracle_rewrite`). En los turnos
  `standalone` no hay oráculo porque no hace falta: ahí cae a `turn.user`, y por eso esa fila
  empata con «último turno» en esos turnos por construcción.

`hit@k` se mide sobre la lista **rankeada** (antes del presupuesto de entrega), que es lo que la
consulta controla; la columna «sección entregada» se mide sobre lo que de verdad se entregó, que
es lo que el generador de la Parte B va a ver.
"""


# %%
class Rewrite(BaseModel):
    """Un solo campo: nada con qué razonar, nada que responder."""

    query: str = Field(max_length=300)


REWRITE_SYSTEM = (
    "Reescribí la última pregunta del usuario para que se entienda SOLA, sin el historial.\n"
    "Reglas:\n"
    "1. NO la respondas. NO expliques. Devolvé solo la pregunta reescrita.\n"
    "2. Reemplazá los pronombres y completá lo que esté elidido usando el historial.\n"
    "3. No agregues datos que el usuario no dijo ni los que vos supongas que son la respuesta.\n"
    "4. Si ya se entiende sola, devolvela igual."
)


def rewrite_query(history: list[dict], question: str) -> tuple[str, int]:
    """La reescritura del modelo, y las llamadas al LLM que costó (1, o 0 si falló)."""
    recent = history[-2 * REWRITE_TURNS:]
    convo = "\n".join(f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
                      for m in recent)
    msgs = [{"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": (f"Historial:\n{convo}\n\n" if convo else "")
             + f"Última pregunta: {question}"}]
    out, _ = llm.structured(msgs, Rewrite, model=MODEL, num_ctx=2048, num_predict=200)
    if out is None or not out.query.strip():
        return question, 0        # un reescritor que falla no inventa una fila: cae al turno
    return out.query.strip(), 1


def conversation_query(history: list[dict], question: str) -> str:
    """El turno MÁS NUEVO primero: el embebedor trunca por la derecha (ver la trampa en §3)."""
    return " \n".join([question] + [m["content"] for m in reversed(history)
                                    if m["role"] == "user"])


QUERY_CONFIGS = {
    "último turno": lambda h, t: (t.user, 0),
    "conversación completa": lambda h, t: (conversation_query(h, t.user), 0),
    "reescritura del modelo": lambda h, t: rewrite_query(h, t.user),
    "reescritura oráculo": lambda h, t: (t.oracle_rewrite or t.user, 0),
}

# %%
log_a = store.ResultsLog(ART / "results" / "05_dialogue_a.jsonl",
                         run_spec={"stage": "05_dialogue_A", "eval": ev.key(), "model": MODEL,
                                   "index": R.index_key, "topk": TOPK,
                                   "rewrite_turns": REWRITE_TURNS,
                                   "cfg": {"dense": True, "prefix": True, "lexical": True,
                                           "rerank": True}})
QUERIES: dict[tuple[str, str], str] = {}     # (config, item_id) -> el texto de la consulta


def part_a_row(d, t, history: list[dict], make) -> dict:
    query, calls = make(history, t)
    tr: dict = {}
    delivered = R.retrieve(query, BEST_CFG, trace=tr)
    ranked = tr["ranked"][:TOPK]
    want_docs = {g.doc_id for g in t.gold}
    return {
        "dialogue_id": d.dialogue_id, "turn": t.turn, "depends_on": t.depends_on,
        "answerable": t.answerable, "query": query, "llm_calls": calls,
        "query_tokens_dense": dense_count(query),
        "truncated_by_embedder": dense_count(query) > rt.DENSE_MAX_SEQ,
        # sin gold no hay acierto que medir: un turno sin respuesta no entra en hit@k
        "hit_doc": (any(R.child(c)["doc_id"] in want_docs for c in ranked)
                    if want_docs else None),
        "hit_sec": (any(R.in_gold_section(R.child(c)["section_id"], t) for c in ranked)
                    if t.gold else None),
        "sec_delivered": R.section_hit(delivered, t) if t.gold else None,
        "n_delivered": len(delivered),
    }


for name, make in QUERY_CONFIGS.items():
    t0 = time.perf_counter()
    todo = 0
    for d in DIALOGUES:
        for t in d.turns:
            iid = f"{d.dialogue_id}#{t.turn}"
            if log_a.done(name, iid):
                continue
            todo += 1
            log_a.write({"arm": name, "item_id": iid,
                         **part_a_row(d, t, dialogue_history(d, t.turn), make)})
    print(f"{name:24} {todo:3} turnos nuevos en {time.perf_counter() - t0:6.1f} s")

ROWS_A = {name: {r["item_id"]: r for r in log_a.rows(name)} for name in QUERY_CONFIGS}
for name in QUERY_CONFIGS:
    for iid, r in ROWS_A[name].items():
        QUERIES[(name, iid)] = r["query"]

# %% [markdown]
r"""
## 3 · Parte A · Las tablas, y la trampa del embebedor

Primero las cuatro filas enteras; después abiertas **por etiqueta de dependencia** y **por número
de turno**, que es donde el ranking puede darse vuelta: un turno 1 `standalone` no tiene nada que
reescribir, y un turno 4 con elipsis no se sostiene solo.
"""

# %%
IDS_A = [f"{d.dialogue_id}#{t.turn}" for d in DIALOGUES for t in d.turns]
WITH_GOLD = [i for i in IDS_A if ROWS_A["último turno"][i]["hit_sec"] is not None]


def rate_a(name: str, ids: list[str], field: str) -> float:
    vals = [ROWS_A[name][i][field] for i in ids if i in ROWS_A[name]]
    vals = [v for v in vals if v is not None]
    return sum(bool(v) for v in vals) / len(vals) if vals else float("nan")


def dial_means_a(name: str, ids: list[str], field: str) -> list[float]:
    acc = collections.defaultdict(list)
    for i in ids:
        v = ROWS_A[name][i][field]
        if v is not None:
            acc[i.split("#")[0]].append(bool(v))
    return [sum(v) / len(v) for v in acc.values()]


table_a = []
for name in QUERY_CONFIGS:
    ci = noise.boot_ci(dial_means_a(name, WITH_GOLD, "hit_sec"), seed=SEED)
    table_a.append((name, {
        f"hit@{TOPK} doc": f"{rate_a(name, WITH_GOLD, 'hit_doc'):.0%}",
        f"hit@{TOPK} sección": f"{rate_a(name, WITH_GOLD, 'hit_sec'):.0%}",
        "IC95 (por guion)": f"{ci[0]:.0%}-{ci[1]:.0%}",
        "sección entregada": f"{rate_a(name, WITH_GOLD, 'sec_delivered'):.0%}",
        "tokens de consulta (embebedor)":
            int(np.mean([ROWS_A[name][i]["query_tokens_dense"] for i in IDS_A])),
        "truncadas": sum(ROWS_A[name][i]["truncated_by_embedder"] for i in IDS_A),
        "llamadas": sum(ROWS_A[name][i]["llm_calls"] for i in IDS_A),
    }))
print_table(f"Parte A · cuatro consultas · {len(WITH_GOLD)} turnos con gold de "
            f"{len(IDS_A)} puntuados", list(table_a[0][1]), table_a,
            smoke=SMOKE_TEST, name_header="consulta")

for a in list(QUERY_CONFIGS)[1:]:
    p = noise.paired([ROWS_A["último turno"][i]["hit_sec"] for i in WITH_GOLD],
                     [ROWS_A[a][i]["hit_sec"] for i in WITH_GOLD])
    print(f"{a:24} vs último turno -> gana {p['gana']}, pierde {p['pierde']}, "
          f"empata {p['empata']}, p = {p['p']}"
          f"{' (sobrevive)' if p['p'] < 0.05 else ' (no sobrevive al piso de ruido)'}")
print(f"\ncon {len(WITH_GOLD)} turnos hacen falta {noise.detectable()} discordantes todos para el "
      f"mismo lado para bajar de p < 0,05 -- y los turnos de un guion no son independientes, "
      f"así que el IC de la tabla es por guion")

# %%
by_dep = []
for dep in sorted(DEPS, key=lambda k: -DEPS[k]):
    ids = [i for i in WITH_GOLD if ROWS_A["último turno"][i]["depends_on"] == dep]
    if not ids:
        continue
    by_dep.append((f"{dep} (n={len(ids)})",
                   {name: f"{rate_a(name, ids, 'hit_sec'):.0%}" for name in QUERY_CONFIGS}))
print_table(f"hit@{TOPK} de sección por etiqueta de dependencia", list(QUERY_CONFIGS), by_dep,
            smoke=SMOKE_TEST, name_header="dependencia")

by_turn = []
for n in sorted({ROWS_A["último turno"][i]["turn"] for i in WITH_GOLD}):
    ids = [i for i in WITH_GOLD if ROWS_A["último turno"][i]["turn"] == n]
    by_turn.append((f"turno {n} (n={len(ids)})",
                    {name: f"{rate_a(name, ids, 'hit_sec'):.0%}" for name in QUERY_CONFIGS}))
print_table(f"hit@{TOPK} de sección por número de turno", list(QUERY_CONFIGS), by_turn,
            smoke=SMOKE_TEST, name_header="turno")

WINNER = max(QUERY_CONFIGS, key=lambda n: rate_a(n, WITH_GOLD, "hit_sec"))
print(f"\nla consulta con mejor hit@{TOPK} de sección: «{WINNER}» "
      f"({rate_a(WINNER, WITH_GOLD, 'hit_sec'):.0%}) -- es la que usa la Parte B, fija")

# %% [markdown]
r"""
### La trampa: una consulta con la conversación completa pierde su último turno

Los embebedores truncan **por la derecha** en el `max_seq_length` que fijamos (acá
`DENSE_MAX_SEQ`). Una conversación concatenada en orden cronológico conserva sus *primeros* turnos
y descarta en silencio el más nuevo, que es el único que dice qué se está preguntando ahora: el
resultado coincide con MTRAG («la conversación completa rinde menos que el último turno») por una
razón que no tiene nada que ver con MTRAG.

Dos cosas abajo. Primero, que con guiones de 4 turnos la consulta **no llega** al límite: la
columna «truncadas» de la tabla es 0 y el efecto que mide la tabla es el del texto, no el del
truncado. Segundo, que el mismo código sobre la conversación **concatenada** (el replay de la
Parte B) sí lo cruza, y entonces el orden pasa a decidir qué turno sobrevive — por eso
`conversation_query` pone el más nuevo primero.

Y una oración para el informe: BM25 **no** tiene ese límite. Pasado el truncado, los dos canales
ven consultas distintas (el denso ve un prefijo, BM25 ve todo), y eso no es una elección: es un
efecto de haber concatenado.
"""

# %%
tok_full = [ROWS_A["conversación completa"][i]["query_tokens_dense"] for i in IDS_A]
print(f"consulta de conversación completa, dentro de un guion: mediana {int(np.median(tok_full))} "
      f"tokens del embebedor, máximo {max(tok_full)} · límite {rt.DENSE_MAX_SEQ}")
print(verdict(max(tok_full) <= rt.DENSE_MAX_SEQ,
              "ninguna llegó al límite: a 4 turnos por guion la trampa no se activa, así que la "
              "tabla de arriba compara textos y no truncados",
              f"{sum(n > rt.DENSE_MAX_SEQ for n in tok_full)} consultas se truncaron: la fila "
              f"«conversación completa» está midiendo el truncado"))

last = REPLAY[-1]
chrono = " \n".join([m["content"] for m in last[2] if m["role"] == "user"] + [last[1].user])
newest = conversation_query(last[2], last[1].user)
n_chrono, n_newest = dense_count(chrono), dense_count(newest)
print(f"\nla misma consulta sobre el replay concatenado ({len(last[2])} mensajes de historial):")
print(f"   cronológica  {n_chrono:5} tokens -> el embebedor ve los primeros {rt.DENSE_MAX_SEQ} y "
      f"descarta {max(n_chrono - rt.DENSE_MAX_SEQ, 0)}: el turno actual queda afuera")
print(f"   más nuevo primero {n_newest:5} tokens -> el turno actual entra siempre")
cut_at = R.encoder.tokenizer.decode(
    R.encoder.tokenizer(chrono, add_special_tokens=False)["input_ids"][:rt.DENSE_MAX_SEQ])
print(verdict(last[1].user not in cut_at,
              "la pregunta del turno actual NO sobrevive al truncado cronológico: la elección de "
              "orden no es cosmética",
              "a este largo el truncado cronológico todavía conserva el turno actual"))

# %% [markdown]
r"""
### Las reescrituras, leídas a mano

El enunciado pide **leer 20 reescrituras**. El archivo `stage5_rewrites.txt` las deja ordenadas al
lado del turno original, del oráculo y de si cambiaron el acierto de sección; lo de abajo son
**proxies mecánicos**, no el resultado de la lectura: cuentan las tres formas de salirse de
reescribir que nombra Ye et al. —responder la pregunta, agregar detalles que el usuario no dijo,
razonar en voz alta—, y una reescritura que hace eso dirige la recuperación como HyDE, por razones
ajenas a reescribir.
"""

# %%
REASONING = ("primero", "entonces", "para eso", "paso 1", "la respuesta", "porque")
lines, n_answers, n_reasons, n_longer, flips = [], 0, 0, 0, {"ganó": [], "perdió": []}
for d in DIALOGUES:
    for t in d.turns:
        iid = f"{d.dialogue_id}#{t.turn}"
        rw = ROWS_A["reescritura del modelo"][iid]
        base = ROWS_A["último turno"][iid]
        orc = ROWS_A["reescritura oráculo"][iid]
        q = rw["query"]
        low = rt.strip_accents(q)
        answers = bool(t.answer_key) and all(rt.strip_accents(k) in low for k in t.answer_key)
        reasons = any(m in low for m in REASONING)
        longer = len(q) > 1.5 * len(t.oracle_rewrite or t.user)
        n_answers += answers
        n_reasons += reasons
        n_longer += longer
        if rw["hit_sec"] is not None and orc["hit_sec"] is not None:
            if rw["hit_sec"] and not orc["hit_sec"]:
                flips["ganó"].append(iid)
            elif orc["hit_sec"] and not rw["hit_sec"]:
                flips["perdió"].append(iid)
        lines += [
            f"=== {iid} · {t.depends_on} · gold {'sí' if t.gold else 'no (sin respuesta)'}",
            f"  usuario : {t.user}",
            f"  oráculo : {t.oracle_rewrite or '(ninguna: standalone)'}",
            f"  modelo  : {q}",
            f"  sección: último turno {base['hit_sec']} · oráculo {orc['hit_sec']} · "
            f"modelo {rw['hit_sec']}"
            + ("  [responde la pregunta]" if answers else "")
            + ("  [razona]" if reasons else "")
            + ("  [1,5x más larga que el oráculo]" if longer else ""), ""]
(ART / "stage5_rewrites.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"{len(IDS_A)} reescrituras en {ART / 'stage5_rewrites.txt'} (hay que leer 20 a mano)")
print(f"proxies mecánicos: {n_answers} contienen la clave de respuesta (responden en vez de "
      f"reescribir) · {n_reasons} traen marcadores de razonamiento · {n_longer} miden más de 1,5x "
      f"el oráculo")
print(f"el modelo le ganó al oráculo en {len(flips['ganó'])} turnos {flips['ganó'][:4]} y perdió "
      f"en {len(flips['perdió'])} {flips['perdió'][:4]}")
for iid in (flips["ganó"][:2] + flips["perdió"][:2]):
    print(f"\n{iid}: oráculo «{clip(ROWS_A['reescritura oráculo'][iid]['query'], 90)}»")
    print(f"{' ' * len(iid)}  modelo  «{clip(ROWS_A['reescritura del modelo'][iid]['query'], 90)}»")

# %% [markdown]
r"""
## 4 · Parte B · El presupuesto, impreso ANTES de la primera respuesta

`num_ctx` queda fijo en 8.192 y el total de **pasajes + historial** lo imponemos nosotros:
`TOTAL_TOKENS`. Se dimensiona por el historial, no por la ventana —un total de 7.000 dejaría al
historial sin restricción en todos los turnos, y la tabla diría «da igual» cuando lo que pasó es
que nunca se llenó.

**Por qué 1.400 y no los 2.000-3.000 que sugiere §7.3.** Esos números suponen unos pocos *miles*
de tokens de historial hacia el final del replay. Nuestras respuestas guionadas son de una línea
—la Etapa 2 las escribió así a propósito, para que el historial sea el de un asistente y no un
ensayo—, así que el replay entero son 858 tokens de historial. Con un total de 2.500 los
presupuestos de historial serían 500/1.250/1.750 y ninguno de los dos últimos restringiría nunca:
«50/50 vs 30/70» sería en realidad «menos pasajes, mismo historial», y el brazo de resumen nunca
cruzaría su umbral.

El otro borde es el **tamaño de corte**: los hijos miden 267 tokens de mediana y 300 en el
percentil 75, así que el presupuesto de pasajes se mueve a saltos de un hijo. Con 1.400 la
escalera queda en 1.120/700/420 ≈ 3/2/1 pasajes; con 1.200 serían 960/600/360, y 600 no alcanza
para dos hijos de 318 y 331 —un caso real de nuestro índice—, así que dos de las tres divisiones
entregarían lo mismo. Y por debajo de ~300 un brazo entrega **cero** pasajes: ahí la fila dejaría
de medir una división y mediría el chunker.

Lo que esta elección deja sobre la mesa, dicho antes de correr: con 280 tokens de historial el
lado del historial restringe en 21 de los 32 turnos; con 700, en 7; con 980, en ninguno. O sea que
mover la división mueve **mucho** los pasajes y **poco** el historial, salvo en 80/20. Por eso las
estrategias se comparan ahí (§6) y no en 50/50.

La tabla de abajo es el chequeo que el enunciado pide hacer *antes* de gastar una sola llamada:
los tokens de historial acumulados por turno contra el presupuesto de historial de cada división.
"""

# %%
BUDGETS = {name: {"pass": int(TOTAL_TOKENS * frac),
                  "hist": TOTAL_TOKENS - int(TOTAL_TOKENS * frac)} for name, frac in SPLITS.items()}
hist_tokens = [count(" ".join(m["content"] for m in h)) for _, _, h, _ in REPLAY]

rows_b = []
for name, b in BUDGETS.items():
    binds = [n > b["hist"] for n in hist_tokens]
    first = next((i + 1 for i, v in enumerate(binds) if v), None)
    rows_b.append((name, {
        "pasajes": b["pass"], "historial": b["hist"],
        "turnos donde el historial no entra": f"{sum(binds)}/{len(binds)}",
        "primer turno restringido": first if first else "ninguno",
    }))
print_table(f"las tres divisiones de {TOTAL_TOKENS} tokens, contra el historial del replay",
            list(rows_b[0][1]), rows_b, smoke=SMOKE_TEST, name_header="división")
print(f"historial acumulado: turno 1 = {hist_tokens[0]} tokens · mediana "
      f"{int(np.median(hist_tokens))} · último turno {hist_tokens[-1]}")
print(verdict(sum(n > min(b['hist'] for b in BUDGETS.values()) for n in hist_tokens)
              >= 0.5 * len(hist_tokens),
              f"el presupuesto de historial más chico ({min(b['hist'] for b in BUDGETS.values())}) "
              f"restringe en la mayoría de los turnos puntuados: la división puede mover algo",
              "el presupuesto de historial más chico no restringe en la mayoría de los turnos -- "
              "hay que bajar TOTAL_TOKENS antes de correr, o la tabla va a decir «el historial no "
              "importa» cuando lo que pasa es que nunca se llenó"))

CHILD_MED = int(np.median([c["n_gen"] for c in R.children]))
MIN_PASS = min(b["pass"] for b in BUDGETS.values())
print(f"\nel otro borde, el de los pasajes: un hijo mide {CHILD_MED} tokens de mediana, así que "
      f"el presupuesto más chico ({MIN_PASS}) entrega del orden de {MIN_PASS // CHILD_MED} "
      f"pasajes y el más grande ({max(b['pass'] for b in BUDGETS.values())}), unos "
      f"{max(b['pass'] for b in BUDGETS.values()) // CHILD_MED}")
print(verdict(MIN_PASS >= CHILD_MED,
              "el presupuesto de pasajes más chico alcanza para al menos un hijo: las tres "
              "divisiones siguen midiendo una división",
              f"el presupuesto más chico ({MIN_PASS}) no alcanza para un hijo de {CHILD_MED} "
              f"tokens: ese brazo entrega cero pasajes y pasa a ser libro cerrado, que es la "
              f"fila de la Etapa 4 y no una división"))

overhead = pr.overhead("x" * 40, MODEL, system=SYSTEM)
room = NUM_CTX - NUM_PREDICT - overhead
print(f"\nla ventana: {NUM_CTX} - {NUM_PREDICT} de reserva - {overhead} de prompt vacío = {room} "
      f"para {TOTAL_TOKENS} de contenido")
print(verdict(room > TOTAL_TOKENS * 1.3,
              "sobra lugar: el total que medimos lo impone el ensamblador, no la ventana (que es "
              "justamente el punto -- un experimento de presupuesto no se puede correr contra el "
              "borde de la ventana)",
              "el total está cerca del borde de la ventana: los marcadores de rol de cada mensaje "
              "pueden empujarlo afuera"))

# %% [markdown]
r"""
## 5 · Parte B · Las tres estrategias de historial

Las tres reciben el mismo historial y el mismo presupuesto, y las tres **cuentan** lo que
conservan: el que decide qué entra es este código, no el servidor.

- **completo**: todo el historial, *mientras entre de verdad*. Pasado el presupuesto descarta los
  mensajes más viejos — y eso se registra en la columna «recortó», porque un brazo «completo» que
  en secreto es una ventana deslizante es la trampa central de esta etapa.
- **ventana N**: los últimos `WINDOW_TURNS` turnos y nada más. Es la estrategia barata: cero
  llamadas extra, y pierde por construcción lo que se dijo antes. **N = 2, y la elección decide el
  experimento**: los guiones son de 4 turnos, así que una ventana de 3 contiene el guion entero y
  solo se diferenciaría de «completo» en el historial de los *otros* guiones —que es ruido—, y la
  comparación estaría estructuralmente incapacitada para mostrar algo. Con N = 2, el turno 4
  pierde el turno 1: es justo lo que los turnos `return` («volviendo al JSON…») apuntan a usar.
- **resumen (MemGPT)**: un resumen continuo más los turnos recientes. Cuando el historial cruza el
  `SUMMARY_AT` de su presupuesto (la línea de advertencia de MemGPT), los mensajes viejos se
  resumen en una sola llamada y quedan los `SUMMARY_KEEP` más recientes. El resumen cuesta
  llamadas y tokens: la tabla los cuenta.

Dos detalles que no son de estilo. El historial recortado **no arranca con una respuesta sin su
pregunta**: un `assistant` huérfano al principio es un turno a medias que el modelo lee como si
él mismo lo hubiera dicho sin que nadie preguntara. Y el resumen entra **adentro del presupuesto
de historial**, no aparte: si no, «resumen» sería «ventana + tokens gratis» y la comparación no
sería de estrategias sino de presupuestos.
"""


# %%
def keep_newest(history: list[dict], budget: int) -> tuple[list[dict], int]:
    """Los mensajes más nuevos que entran en `budget`, en orden, y lo que ocupan."""
    kept, used = [], 0
    for msg in reversed(history):
        n = count(msg["content"])
        if used + n > budget:
            break                 # acá descartar los MÁS VIEJOS es justamente la idea
        kept.insert(0, msg)
        used += n
    if kept and kept[0]["role"] == "assistant":    # no arrancar con una respuesta sin su pregunta
        used -= count(kept[0]["content"])
        kept.pop(0)
    return kept, used


class Summary(BaseModel):
    """Un solo campo, como el reescritor: no hay dónde razonar ni a quién responderle.

    El tope de 500 caracteres (~130 tokens) no es cosmético: el resumen se paga del presupuesto de
    historial, y un resumen de 250 tokens dentro de un presupuesto de 280 dejaría al brazo sin
    turnos recientes -- «resumen» sería «solo resumen», que es otra estrategia.
    """

    summary: str = Field(max_length=500)


SUMMARY_SYSTEM = (
    "Resumí una conversación entre un usuario y un asistente sobre documentación de Python.\n"
    "Reglas:\n"
    "1. Conservá los nombres EXACTOS que aparecieron (funciones, módulos, parámetros, versiones).\n"
    "2. Conservá qué preguntó el usuario y qué se le respondió, en orden.\n"
    "3. No agregues nada que no esté en el texto. Si algo no se dijo, no lo pongas.\n"
    "4. Español, hasta 5 viñetas de una línea. Máximo 500 caracteres en total."
)
SUMMARIES: list[dict] = []        # para la auditoría de §7


def summarize(previous: str, msgs: list[dict], *, arm: str, upto: str) -> tuple[str, int]:
    convo = "\n".join(f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}"
                      for m in msgs)
    body = (f"Resumen previo:\n{previous}\n\n" if previous else "") + f"Turnos nuevos:\n{convo}"
    out, res = llm.structured([{"role": "system", "content": SUMMARY_SYSTEM},
                               {"role": "user", "content": body}],
                              Summary, model=MODEL, num_ctx=4096, num_predict=500)
    if out is None:
        # un resumen que falla no se descarta en silencio: se conserva el anterior y se cuenta
        SUMMARIES.append({"arm": arm, "upto": upto, "error": res.error, "summary": previous,
                          "n_msgs": len(msgs)})
        return previous, 1
    SUMMARIES.append({"arm": arm, "upto": upto, "error": None, "summary": out.summary,
                      "n_msgs": len(msgs), "tokens": count(out.summary)})
    return out.summary, 1


def block_of(summary: str) -> str:
    return f"Resumen de la conversación previa:\n{summary}" if summary else ""


def history_for(strategy: str, history: list[dict], budget: int, state: dict,
                upto: str) -> dict:
    """Lo que esta estrategia conserva de `history` en `budget` tokens."""
    if strategy == "completo":
        kept, used = keep_newest(history, budget)
        return {"kept": kept, "memory": None, "used": used, "calls": 0,
                "trimmed": len(kept) < len(history)}
    if strategy.startswith("ventana"):
        window = history[-2 * WINDOW_TURNS:]
        kept, used = keep_newest(window, budget)
        return {"kept": kept, "memory": None, "used": used, "calls": 0,
                "trimmed": len(kept) < len(history)}

    calls = 0
    recent = history[state["covered"]:]
    block = block_of(state["summary"])
    while (count(block) + sum(count(m["content"]) for m in recent) > SUMMARY_AT * budget
           and len(recent) > SUMMARY_KEEP):
        old, recent = recent[:-SUMMARY_KEEP], recent[-SUMMARY_KEEP:]
        state["summary"], n = summarize(state["summary"], old, arm=state["arm"], upto=upto)
        state["covered"] = len(history) - len(recent)
        block = block_of(state["summary"])
        calls += n
    kept, used = keep_newest(recent, max(budget - count(block), 0))
    return {"kept": kept, "memory": block or None, "used": used + count(block), "calls": calls,
            "trimmed": len(kept) < len(recent), "summary": state["summary"]}


# %% [markdown]
r"""
## 6 · Parte B · El experimento: una variable por vez

**Primero la división**, con la estrategia fija en «completo»: 80/20, 50/50, 30/70 del mismo total.
**Después la estrategia**, con la división fija en **80/20**. El brazo «80/20 · completo» es el
mismo en los dos y se corre una sola vez: es la bisagra que hace que las dos mitades del
experimento estén en la misma escala.

La estrategia se compara en 80/20, no en 50/50, porque es la única división donde el historial
escasea: su presupuesto (280 tokens) no alcanza en 21 de los 32 turnos, y el umbral de resumen
(196) se cruza en el turno 9. En 50/50 el umbral no se cruza hasta el turno 19 y las tres
estrategias serían la misma en los primeros dos tercios del replay **por construcción** —una tabla
de tres filas iguales no dice «las estrategias dan igual», dice que no había nada que decidir.

Fijo en todos los brazos: los guiones y su orden, el recuperador, la consulta ganadora de la
Parte A, `num_ctx`, la reserva de salida y el total. La recuperación se hace **una vez por turno**
con el presupuesto de entrega más grande, y después cada brazo recorta a su propio presupuesto de
pasajes con el mismo ensamblador: así los brazos ven el mismo *ranking* y lo único que los separa
es cuánto de él entró.
"""

# %%
MAX_PASS = max(b["pass"] for b in BUDGETS.values())
PASSAGES: dict[str, list[dict]] = {}
t0 = time.perf_counter()
for d, t, _h, iid in REPLAY:
    # la consulta es la de la Parte A (del historial de SU guion: eso es lo que la Parte A midió);
    # el historial del generador, en cambio, es el del replay. Son dos cosas distintas a propósito.
    PASSAGES[iid] = R.retrieve(QUERIES[(WINNER, iid)],
                               RetrievalConfig(dense=True, prefix=True, lexical=True, rerank=True,
                                               budget_tokens=MAX_PASS))
print(f"recuperación: {len(PASSAGES)} turnos con «{WINNER}» en {time.perf_counter() - t0:.0f} s "
      f"(una vez por turno, compartida por los brazos)")

S = STRATEGY_SPLIT
ARMS = {"80/20 · completo": ("80/20", "completo"),
        "50/50 · completo": ("50/50", "completo"),
        "30/70 · completo": ("30/70", "completo"),
        f"{S} · ventana {WINDOW_TURNS}": (S, f"ventana {WINDOW_TURNS}"),
        f"{S} · resumen": (S, "resumen")}

log_b = store.ResultsLog(ART / "results" / "05_dialogue_b.jsonl",
                         run_spec={"stage": "05_dialogue_B", "eval": ev.key(), "model": MODEL,
                                   "index": R.index_key, "num_ctx": NUM_CTX,
                                   "num_predict": NUM_PREDICT, "total": TOTAL_TOKENS,
                                   "query": WINNER, "window": WINDOW_TURNS,
                                   "summary_at": SUMMARY_AT, "summary_keep": SUMMARY_KEEP,
                                   "system": store.sha256_text(SYSTEM)[:16]})

for arm, (split, strategy) in ARMS.items():
    b = BUDGETS[split]
    state = {"summary": "", "covered": 0, "arm": arm}
    t0, n_new = time.perf_counter(), 0
    for d, t, history, iid in REPLAY:
        # el historial se arma SIEMPRE, incluso si la fila ya está: el estado del resumen es
        # secuencial, y saltearlo dejaría al brazo con un resumen que no corresponde al turno.
        h = history_for(strategy, history, b["hist"], state, iid)
        if log_b.done(arm, iid):
            continue
        n_new += 1
        delivered, skipped = pr.assemble(PASSAGES[iid], b["pass"], count)
        msgs = pr.messages(t.user, delivered, history=h["kept"], memory=h["memory"],
                           system=SYSTEM)
        t1 = time.perf_counter()
        ans, res = pr.answer(msgs, model=MODEL, num_ctx=NUM_CTX, num_predict=NUM_PREDICT)
        row = pr.score(ans, res, t, delivered, in_gold=lambda sid: R.in_gold_section(sid, t))
        log_b.write({"arm": arm, "item_id": iid, **row,
                     "dialogue_id": d.dialogue_id, "turn": t.turn, "depends_on": t.depends_on,
                     "split": split, "strategy": strategy,
                     "hist_budget": b["hist"], "pass_budget": b["pass"],
                     "hist_tokens": h["used"], "hist_msgs": len(h["kept"]),
                     "hist_available": len(history), "hist_trimmed": h["trimmed"],
                     "extra_calls": h["calls"], "n_skipped": len(skipped),
                     "wall_s": round(time.perf_counter() - t1, 2),
                     "answer_text": (ans.answer if ans else "")})
    print(f"{arm:26} {n_new:3} turnos nuevos en {time.perf_counter() - t0:6.1f} s")

ROWS_B = {arm: {r["item_id"]: r for r in log_b.rows(arm)} for arm in ARMS}
IDS_B = [iid for _, _, _, iid in REPLAY]
TURN_BY_ID = {iid: t for _, t, _, iid in REPLAY}
ANSWERABLE_B = [i for i in IDS_B if TURN_BY_ID[i].answerable]
UNANSWERABLE_B = [i for i in IDS_B if not TURN_BY_ID[i].answerable]

# %%
# El brazo más ancho tiene el MISMO presupuesto de pasajes con el que se recuperó: si ahí se
# saltea algo, el ensamblador y el recuperador no están contando lo mismo. En los otros brazos
# saltear ES el experimento.
WIDEST = [a for a, (s, _) in ARMS.items() if BUDGETS[s]["pass"] == MAX_PASS]
print(verdict(all(r["n_skipped"] == 0 for a in WIDEST for r in ROWS_B[a].values()),
              f"{', '.join(WIDEST)} no salteó ningún pasaje: el ensamblador cuenta lo mismo que "
              f"el recuperador, así que lo que saltean los demás es su presupuesto y no un "
              f"desacuerdo de conteo",
              "el brazo más ancho salteó pasajes: el `n_gen` del índice y el conteo del "
              "ensamblador no coinciden, y eso hay que arreglarlo antes de leer la tabla"))

drift = []
for arm in ARMS:
    rows = list(ROWS_B[arm].values())
    pairs = [(r["expected_prompt_tokens"], r["prompt_tokens"]) for r in rows
             if r["expected_prompt_tokens"] and r["prompt_tokens"]]
    drift.append((arm, {
        "nuestra cuenta (media)": int(np.mean([a for a, _ in pairs])) if pairs else 0,
        "el servidor (media)": int(np.mean([b for _, b in pairs])) if pairs else 0,
        "máx. diferencia": max((abs(a - b) for a, b in pairs), default=0),
        "recortó el historial": sum(r["hist_trimmed"] for r in rows),
        "de": len(rows)}))
print_table("el total, tal como lo contó el servidor (la trampa de comparar brazos distintos)",
            list(drift[0][1]), drift, smoke=SMOKE_TEST, name_header="brazo")
print(verdict(max(d[1]["máx. diferencia"] for d in drift) <= 64,
              "nuestra cuenta y la del servidor coinciden en todos los brazos: nadie fue truncado "
              "en silencio, y los totales son comparables",
              "nuestra cuenta y la del servidor se separan: antes de leer cualquier fila hay que "
              "saber si Ollama descartó mensajes viejos por su cuenta"))

# %%
def rate_b(arm: str, ids: list[str], field: str) -> float:
    vals = [ROWS_B[arm][i][field] for i in ids if i in ROWS_B[arm]]
    vals = [v for v in vals if v is not None]
    return sum(bool(v) for v in vals) / len(vals) if vals else float("nan")


def dial_means(arm: str, ids: list[str], field: str) -> list[float]:
    """Un valor por GUION: 36 turnos de 9 guiones no son 36 muestras."""
    acc = collections.defaultdict(list)
    for i in ids:
        v = ROWS_B[arm].get(i, {}).get(field)
        if v is not None:
            acc[i.split("#")[0]].append(bool(v))
    return [sum(v) / len(v) for v in acc.values()]


def paired_dialogues(a: str, b: str, ids: list[str], field: str = "correct") -> dict:
    """El test de signos sobre GUIONES: un guion gana si su promedio sube."""
    acc_a = collections.defaultdict(list)
    acc_b = collections.defaultdict(list)
    for i in ids:
        va, vb = ROWS_B[a].get(i, {}).get(field), ROWS_B[b].get(i, {}).get(field)
        if va is not None and vb is not None:
            acc_a[i.split("#")[0]].append(bool(va))
            acc_b[i.split("#")[0]].append(bool(vb))
    w = l = t = 0
    for k in acc_a:
        ma, mb = sum(acc_a[k]) / len(acc_a[k]), sum(acc_b[k]) / len(acc_b[k])
        w += mb > ma
        l += ma > mb
        t += ma == mb
    return {"gana": w, "pierde": l, "empata": t, "p": round(noise.sign_test(w, l), 3),
            "guiones": w + l + t}


def table_for(arms: list[str], title: str) -> list:
    out = []
    for arm in arms:
        rows = [ROWS_B[arm][i] for i in IDS_B if i in ROWS_B[arm]]
        ci = noise.boot_ci(dial_means(arm, ANSWERABLE_B, "correct"), seed=SEED)
        out.append((arm, {
            "correctas": f"{rate_b(arm, ANSWERABLE_B, 'correct'):.0%}",
            "IC95 (por guion)": f"{ci[0]:.0%}-{ci[1]:.0%}",
            "clave en la respuesta": f"{rate_b(arm, ANSWERABLE_B, 'key_contained'):.0%}",
            f"abstuvo bien ({len(UNANSWERABLE_B)} s/resp.)":
                f"{sum(1 for i in UNANSWERABLE_B if ROWS_B[arm][i]['abstained'])}"
                f"/{len(UNANSWERABLE_B)}",
            "citas ⊆ entregadas": f"{rate_b(arm, IDS_B, 'citations_subset'):.0%}",
            "hist. tokens": int(np.mean([r["hist_tokens"] for r in rows])),
            "pasajes": round(float(np.mean([r["n_delivered"] for r in rows])), 1),
            "llamadas/turno": round(1 + float(np.mean([r["extra_calls"] for r in rows])), 2),
            "seg/turno": round(float(np.mean([r["wall_s"] for r in rows])), 1),
            "fallos": sum(1 for r in rows if r["category"] == "fallo"),
        }))
    print_table(title, list(out[0][1]), out, smoke=SMOKE_TEST, name_header="brazo")
    return out


SPLIT_ARMS = [a for a, (_, s) in ARMS.items() if s == "completo"]
STRAT_ARMS = [a for a, (sp, _) in ARMS.items() if sp == STRATEGY_SPLIT]
table_split = table_for(SPLIT_ARMS, f"una variable: la división de {TOTAL_TOKENS} tokens "
                                    f"(estrategia fija: completo)")
for a in SPLIT_ARMS[1:]:
    print(f"{a:26} vs {SPLIT_ARMS[0]:26} -> {paired_dialogues(SPLIT_ARMS[0], a, ANSWERABLE_B)}")

table_strat = table_for(STRAT_ARMS, f"una variable: la estrategia de historial (división fija: "
                                    f"{STRATEGY_SPLIT}, donde el historial escasea)")
for a in STRAT_ARMS[1:]:
    print(f"{a:26} vs {STRAT_ARMS[0]:26} -> {paired_dialogues(STRAT_ARMS[0], a, ANSWERABLE_B)}")
print(f"\ncon {len({i.split('#')[0] for i in ANSWERABLE_B})} guiones, el test de signos no puede "
      f"bajar de p < 0,05 con menos de {noise.detectable()} discordantes para el mismo lado: "
      f"un resultado nulo acá es «la prueba no podía mostrarlo», no «no hay diferencia»")

BEST_ARM = max(ARMS, key=lambda a: rate_b(a, ANSWERABLE_B, "correct"))
print(f"el brazo con más correctas: {BEST_ARM}")

# %%
by_dep_b = []
for dep in sorted(DEPS, key=lambda k: -DEPS[k]):
    ids = [i for i in ANSWERABLE_B if ROWS_B[BEST_ARM][i]["depends_on"] == dep]
    if not ids:
        continue
    by_dep_b.append((f"{dep} (n={len(ids)})",
                     {a: f"{rate_b(a, ids, 'correct'):.0%}" for a in ARMS}))
print_table("correctas por etiqueta de dependencia, brazo por brazo", list(ARMS), by_dep_b,
            smoke=SMOKE_TEST, name_header="dependencia")

by_turn_b = []
for n in sorted({t.turn for t in TURN_BY_ID.values()}):
    ids = [i for i in ANSWERABLE_B if ROWS_B[BEST_ARM][i]["turn"] == n]
    if not ids:
        continue
    by_turn_b.append((f"turno {n} (n={len(ids)})",
                      {a: f"{rate_b(a, ids, 'correct'):.0%}" for a in ARMS}))
print_table("correctas por número de turno", list(ARMS), by_turn_b, smoke=SMOKE_TEST,
            name_header="turno")

# %% [markdown]
r"""
## 7 · Poner a prueba el resumen

Los turnos etiquetados `return` preguntan por un detalle de los turnos 1-2 del mismo guion: son
los turnos donde la ventana de 2 ya tiró lo que hace falta y el resumen tiene que haberlo
conservado. Son **2** en el conjunto congelado, así que la tabla de abajo no puede sostener
ninguna conclusión por sí sola: `noise.detectable()` pide 6 discordantes para el mismo lado. Dos
cosas más, entonces: los dos turnos `return` **nombran su propio tema** («volviendo al JSON…»), lo
que los hace menos dependientes del historial de lo que su etiqueta sugiere; y el conjunto que de
verdad separa las estrategias es el de **todos** los turnos que dependen del historial
(`coreference`, `ellipsis`, `return`) **en el régimen donde el presupuesto restringió** —ahí
sí hay n, y es la segunda tabla.

Y después la auditoría de los resúmenes en las tres categorías de Wang et al. —**hechos
inventados**, **relaciones equivocadas**, **detalles faltantes**—, que no se puede automatizar: el
texto de cada resumen queda en `stage5_summaries.txt`, al lado de los mensajes que resumió. Las
cuentas se cargan a mano en `stage5_summary_audit.json`; si no está, se imprime lo que hay que
leer y se dice que falta, en vez de inventar un número.
"""

# %%
RETURN_IDS = [i for i in ANSWERABLE_B if ROWS_B[BEST_ARM][i]["depends_on"] == "return"]
if RETURN_IDS:
    print_table(f"los {len(RETURN_IDS)} turnos «return» (un detalle de los turnos 1-2)",
                ["correctas", "clave en la respuesta", "hist. tokens"],
                [(a, {"correctas": f"{rate_b(a, RETURN_IDS, 'correct'):.0%}",
                      "clave en la respuesta": f"{rate_b(a, RETURN_IDS, 'key_contained'):.0%}",
                      "hist. tokens": int(np.mean([ROWS_B[a][i]["hist_tokens"]
                                                   for i in RETURN_IDS]))}) for a in ARMS],
                smoke=SMOKE_TEST, name_header="brazo")
else:
    print("no hay turnos «return» en este perfil: la prueba del resumen no se puede correr")

# El conjunto con n: los turnos que dependen del historial, y los que además cayeron en el
# régimen restringido (el presupuesto de historial no alcanzaba para todo lo dicho).
HIST_DEP = [i for i in ANSWERABLE_B
            if ROWS_B[STRAT_ARMS[0]][i]["depends_on"] in ("coreference", "ellipsis", "return")]
TIGHT = [i for i in HIST_DEP
         if any(ROWS_B[a][i]["hist_trimmed"] or ROWS_B[a][i]["extra_calls"] for a in STRAT_ARMS)]
print_table(f"turnos que dependen del historial: {len(HIST_DEP)} en total, {len(TIGHT)} en el "
            f"régimen donde el presupuesto de {STRATEGY_SPLIT} restringió",
            ["correctas (dep. del historial)", f"correctas (n={len(TIGHT)}, restringidos)",
             "llamadas/turno"],
            [(a, {"correctas (dep. del historial)": f"{rate_b(a, HIST_DEP, 'correct'):.0%}",
                  f"correctas (n={len(TIGHT)}, restringidos)":
                      f"{rate_b(a, TIGHT, 'correct'):.0%}",
                  "llamadas/turno": round(1 + float(np.mean(
                      [ROWS_B[a][i]["extra_calls"] for i in HIST_DEP])), 2)})
             for a in STRAT_ARMS], smoke=SMOKE_TEST, name_header="brazo")
print(f"el test de signos por guion sobre los restringidos: "
      f"{ {a: paired_dialogues(STRAT_ARMS[0], a, TIGHT) for a in STRAT_ARMS[1:]} }")

lines = []
for s in SUMMARIES:
    lines += [f"=== {s['arm']} · hasta {s['upto']} · {s['n_msgs']} mensajes resumidos"
              + (f" · ERROR {s['error']}" if s["error"] else f" · {s.get('tokens')} tokens"),
              wrap(s["summary"] or "(vacío)", width=90, indent="  "), ""]
(ART / "stage5_summaries.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n{len(SUMMARIES)} resúmenes escritos ({sum(1 for s in SUMMARIES if s['error'])} con "
      f"error) en {ART / 'stage5_summaries.txt'}")
try:
    audit = store.read_json(ART / "stage5_summary_audit.json")
    print_table("auditoría de los resúmenes (lectura a mano, categorías de Wang et al.)",
                ["cuenta", "de"], [(k, {"cuenta": v, "de": audit.get("claims_read", "?")})
                                   for k, v in audit.items() if k != "claims_read"],
                smoke=SMOKE_TEST, name_header="categoría")
except FileNotFoundError:
    print("falta stage5_summary_audit.json: la auditoría de los resúmenes está PENDIENTE "
          "(hechos inventados / relaciones equivocadas / detalles faltantes, a mano sobre "
          f"{ART / 'stage5_summaries.txt'})")
if SUMMARIES:
    print(f"\nun resumen, para mirarlo acá:\n{wrap(SUMMARIES[-1]['summary'], max_lines=6)}")

# %% [markdown]
r"""
## 8 · Las respuestas del informe

**¿Qué configuración de consulta ganó en qué tipo de dependencia, y cambia el ranking con el número
de turno?** Las dos tablas de §3. Lo que hay que leer ahí no es el promedio: es si el orden de las
cuatro filas se da vuelta entre `standalone` y `coreference`/`ellipsis` (donde el último turno
solo no se sostiene) y entre el turno 1 y el 4.

**Con una ventana fija, ¿dónde quedó la mejor división pasajes/historial, y la diferencia está por
encima de la cota de ruido?** La primera tabla de §6, con su test de signos **por guion** y el
tamaño de prueba declarado. La tabla de §4 es la que dice si la pregunta era contestable: sin
turnos donde el presupuesto de historial restrinja, las tres divisiones solo mueven el presupuesto
de pasajes.

**¿Qué hicieron mal los resúmenes?** §7, en las tres categorías de Wang et al., contado a mano
sobre `stage5_summaries.txt`.

**¿Cuánto cuesta cada estrategia por turno, y vale la pena el ganador?** Las columnas
«llamadas/turno», «hist. tokens» y «seg/turno» de §6: el resumen paga una llamada extra por turno
desde que cruza su umbral, y solo las paga bien si gana en los turnos que dependen del historial y
cayeron en el régimen restringido (la segunda tabla de §7), no en el promedio de los 32.

**¿Dónde perdió la reescritura oráculo frente a la del modelo?** El final de §3, turno por turno,
con los proxies mecánicos de las tres formas de salirse de reescribir.
"""

# %%
store.write_json(ART / "stage5_rows.json", {
    "model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "total": TOTAL_TOKENS,
    "part_a": {"winner": WINNER, "table": {n: v for n, v in table_a},
               "by_dependency": {n: v for n, v in by_dep},
               "by_turn": {n: v for n, v in by_turn},
               "rewrite_flips": flips,
               "rewrite_proxies": {"responden": n_answers, "razonan": n_reasons,
                                   "mas_largas": n_longer},
               "full_conversation_query_tokens": {"median": int(np.median(tok_full)),
                                                  "max": max(tok_full),
                                                  "limit": rt.DENSE_MAX_SEQ,
                                                  "replay_chrono": n_chrono,
                                                  "replay_newest_first": n_newest}},
    "part_b": {"budgets": BUDGETS, "best_arm": BEST_ARM,
               "hist_tokens_per_turn": hist_tokens,
               "splits": {n: v for n, v in table_split},
               "strategies": {n: v for n, v in table_strat},
               "paired_splits": {a: paired_dialogues(SPLIT_ARMS[0], a, ANSWERABLE_B)
                                 for a in SPLIT_ARMS[1:]},
               "paired_strategies": {a: paired_dialogues(STRAT_ARMS[0], a, ANSWERABLE_B)
                                     for a in STRAT_ARMS[1:]},
               "return_turns": {a: rate_b(a, RETURN_IDS, "correct") for a in ARMS}
               if RETURN_IDS else {},
               "history_dependent": {"ids": HIST_DEP, "tight": TIGHT,
                                     "rates": {a: rate_b(a, HIST_DEP, "correct")
                                               for a in STRAT_ARMS},
                                     "rates_tight": {a: rate_b(a, TIGHT, "correct")
                                                     for a in STRAT_ARMS}},
               "n_summaries": len(SUMMARIES)},
})

run.note("Parte A · consulta ganadora", WINNER)
run.note("Parte A · hit@k de sección", {n: round(rate_a(n, WITH_GOLD, "hit_sec"), 3)
                                        for n in QUERY_CONFIGS})
run.note("Parte B · correctas", {a: round(rate_b(a, ANSWERABLE_B, "correct"), 3) for a in ARMS})
run.note("Parte B · mejor brazo", BEST_ARM)
run.note("Parte B · resúmenes escritos", len(SUMMARIES))
run.note("pendiente a mano", ["leer 20 reescrituras (stage5_rewrites.txt)",
                              "auditar los resúmenes (stage5_summaries.txt)"])
run.finish()
