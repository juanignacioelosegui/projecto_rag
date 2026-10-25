# %% [markdown]
r"""
# Etapa 4 · Respuestas fundamentadas, y qué se puede puntuar sin un juez

Tres filas sobre las **mismas** preguntas: libro cerrado (sin recuperación), la mejor
configuración de la Etapa 3, y oráculo (la sección gold entregada directamente). Las tres con el
mismo ensamblador, la misma reserva de salida y el mismo `num_ctx`, porque si el prompt se arma
distinto en cada fila la tabla mide el ensamblador y no la recuperación.

**Lo que hace esta etapa y la Etapa 3 no podía.** La Etapa 3 cerró con una brecha —documento →
sección → *respuesta entregada*— y una predicción: el techo de la generación es el de respuesta
entregada. Acá se mide cuánto de ese techo se alcanza, y las dos fallas que separan la tabla 2×2:
**fiel al pasaje equivocado** (recuperó mal y respondió lo que decía ese pasaje) y **lo tenía y
respondió otra cosa**.

**Entra**: el conjunto congelado, el recuperador de `retrieval.py`, el ensamblador de
`prompting.py`. **Sale**: `artifacts/<perfil>/results/04_generation.jsonl`, la caché de llamadas
en `artifacts/<perfil>/llm_cache/`, y `stage4_rows.json`.

**Cómo Ollama trata el prompt** (§6.2 del enunciado, y los hechos del encabezado de `llm.py`):
la API **no tiene estado**, `num_ctx` se manda en **cada** llamada (si no, una T4 usa 4.096), y
un prompt que no entra **no da error**: el servidor conserva los primeros tokens y la cola y
descarta el medio — o sea el sistema, la regla de abstención y los pasajes mejor rankeados. Por
eso cada llamada compara **nuestra cuenta** con el `prompt_eval_count` del servidor, y §5 lo hace
pasar una vez a propósito.
"""

# %%
import environment  # noqa: F401  (efectos secundarios: guarda de JAX, caché de HF, bundle TLS, UTF-8)

environment.summary()

import collections
import os
import time
from pathlib import Path  # noqa: F401

import numpy as np

import llm
import noise
import prompting as pr
import retrieval as rt
import store
from console import StageRun, clip, print_table, verdict, wrap
from retrieval import RetrievalConfig, answer_delivered
from schema import load_eval

SMOKE_TEST = os.environ.get("SMOKE_TEST", "1").lower() not in ("0", "false", "no")
ART = environment.artifacts_dir(SMOKE_TEST)
PROFILE = "smoke" if SMOKE_TEST else "full"
EVAL = environment.ROOT / ("eval/smoke" if SMOKE_TEST else "eval")
MODELS = rt.models(SMOKE_TEST)
MODEL = MODELS["generator"]

NUM_CTX = 8192            # explícito en CADA llamada: el default de una T4 son 4.096
NUM_PREDICT = 700         # la reserva de salida; el prompt se mantiene por debajo de ctx - esto
DELIVER_TOKENS = rt.BUDGET_TOKENS   # el mismo presupuesto de entrega que midió la Etapa 3
POSITION_TEST = os.environ.get("POSITION", "1").lower() not in ("0", "false", "no")
SEED = 0

environment.seed_everything(SEED)
run = StageRun.start("04_generation", ART, profile=PROFILE, config={
    "model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
    "deliver_tokens": DELIVER_TOKENS, "seed": SEED,
})

ev = load_eval(EVAL, require_frozen=not SMOKE_TEST,
               documents_path=None if SMOKE_TEST else ART / "documents.jsonl")
ITEMS = ev.items
TEST = [i for i in ITEMS if i.split == "test"]
DEV = [i for i in ITEMS if i.split == "dev"]
print(f"conjunto: {len(TEST)} ítems de test · {len(DEV)} de dev (el prompt se mira en dev)")

# %% [markdown]
r"""
## 1 · El servidor, medido antes de usarlo

`llm.setup` hace lo de siempre (servidor, pull, placement) y **tres sondas** que deciden cómo
hay que escribir el resto:

- `think=False` de verdad apaga el razonamiento en este tag (si no, `num_predict` se va en
  pensar y la respuesta llega cortada);
- en qué **orden escribe las claves** del JSON: por eso el schema declara `analysis` primero
  *y* con un nombre que también va primero alfabéticamente — así el modelo razona antes de
  comprometerse con una respuesta en los dos casos;
- la **calibración**: nuestra cuenta con la plantilla de chat menos la del servidor, y si
  `prompt_eval_count` incluye los tokens servidos desde la caché de prompt (que es lo que decide
  cómo se detecta la truncación silenciosa).
"""

# %%
llm.use_cache(ART / "llm_cache" / f"{MODEL.replace(':', '-')}.jsonl")
info = llm.setup(MODEL, num_ctx=NUM_CTX)
run.note("sondas del servidor", {k: info.get(k) for k in ("server", "digest", "placement")})
run.note("orden de claves", info.get("key_order", {}).get("order"))

# %% [markdown]
r"""
## 2 · El ensamblador, verificado contra el servidor

El ensamblador deja una reserva de salida igual a `num_predict`, cuenta cada pasaje con el
tokenizer del generador y **saltea** el que no entra en vez de cortar el loop: un `break` en el
primer pasaje que no entra descarta también todos los más chicos que vienen detrás.

Dos cuentas que no son la misma: `llm.count_tokens` cuenta **texto pelado** (no ve los
marcadores de la plantilla ni los corchetes de cada pasaje) y por eso el presupuesto de pasajes
se calcula contra `llm.count_prompt_tokens` del prompt **vacío** más un margen. La verificación
de abajo es contra el servidor, en un ítem de dev.
"""

# %%
R = rt.load(smoke=SMOKE_TEST, art=ART, docs_path=ART / "documents.jsonl")
BEST_CFG = RetrievalConfig(dense=True, prefix=True, lexical=True, rerank=True)
try:                       # la mejor fila de la Etapa 3, si ya corrió en este perfil
    _s3 = store.read_json(ART / "stage3_rows.json")
    print(f"la Etapa 3 dejó como mejor fila: {_s3['best']} · piezas que sobrevivieron: "
          f"{_s3['survivors']}")
except FileNotFoundError:
    print("sin stage3_rows.json en este perfil: se usa la configuración «+ rerank» por defecto")


def count(text: str) -> int:
    return llm.count_tokens(text, MODEL) or 0


def budget_for(item, **kw) -> int:
    """El presupuesto de pasajes: el de la Etapa 3, pero nunca más de lo que deja la ventana."""
    return min(DELIVER_TOKENS, pr.passage_budget(item.question, MODEL, num_ctx=NUM_CTX,
                                                 num_predict=NUM_PREDICT, **kw))


probe_item = DEV[0] if DEV else TEST[0]
probe_passages = R.retrieve(probe_item.question, BEST_CFG)
delivered, skipped = pr.assemble(probe_passages, budget_for(probe_item), count)
msgs = pr.messages(probe_item.question, delivered)
mine = llm.count_prompt_tokens(msgs, MODEL)
ans, r = pr.answer(msgs, model=MODEL, num_ctx=NUM_CTX, num_predict=NUM_PREDICT)
print(f"ítem {probe_item.item_id}: {len(delivered)} pasajes entregados, {len(skipped)} salteados "
      f"· presupuesto {budget_for(probe_item)} tokens")
print(f"nuestra cuenta (plantilla de chat) {mine} · el servidor (prompt_eval_count) "
      f"{r.prompt_tokens} · done_reason {r.done_reason} · error {r.error}")
print(verdict(r.error is None and mine is not None and r.prompt_tokens is not None
              and abs(mine - r.prompt_tokens) <= 32,
              "las dos cuentas coinciden: el prompt que armamos es el que el servidor evaluó",
              "nuestra cuenta y la del servidor no coinciden -- antes de leer cualquier fila hay "
              "que saber por qué"))
if ans:
    print(f"\nanalysis: {clip(ans.analysis, 200)}")
    print(f"answer:   {clip(ans.answer, 200)}")
    print(f"can_answer: {ans.can_answer} · citations: {ans.citations}")

# %% [markdown]
r"""
## 3 · Las tres filas

- **libro cerrado**: sin pasajes. Mide cuánto de nuestras preguntas ya está en los pesos — y es
  la fila que dice si el corpus aporta algo.
- **la mejor de la Etapa 3**: `+ rerank`, con el mismo presupuesto de entrega.
- **oráculo**: la sección gold, entregada directamente. Es el techo de este generador con este
  prompt: lo que falle acá no lo va a arreglar ningún recuperador.

El oráculo entrega **una sola versión** del gold. El gold de la Etapa 2 es un *conjunto* (las dos
versiones de Python son copias intercambiables); entregar las dos duplicaría el texto y le daría
al oráculo el doble de presupuesto que a las otras filas.

Un ítem **sin respuesta en el corpus** no tiene gold: su fila oráculo es, por construcción, sin
pasajes, y responder bien ahí significa **abstenerse**. Esos cinco ítems son los únicos donde
«abstenerse» es la respuesta correcta; en todos los demás, abstenerse con la respuesta en el
contexto es la falla que §4 cuenta aparte.
"""


# %%
def oracle_passages(item) -> list[dict]:
    want = [g.section_id for g in item.gold if g.section_id]
    if not want:
        return []
    doc = want[0].split("#")[0]        # una sola versión, no el conjunto entero
    ps = [p for p in R.parents.values() if p["section_id"] in want and p["doc_id"] == doc]
    return [{"id": p["parent_id"], "doc_id": p["doc_id"], "section_id": p["section_id"],
             "text": p["text"], "n_gen": p["n_gen"], "rank": i + 1}
            for i, p in enumerate(sorted(ps, key=lambda p: p["parent_id"]))]


ROW_PASSAGES = {
    "libro cerrado": lambda it: [],
    "mejor recuperación": lambda it: R.retrieve(it.question, BEST_CFG),
    "oráculo": oracle_passages,
}


def run_item(item, passages: list[dict], *, num_ctx: int = NUM_CTX,
             num_predict: int = NUM_PREDICT, strict_ctx: bool = True) -> dict:
    delivered, skipped = pr.assemble(passages, budget_for(item), count)
    msgs = pr.messages(item.question, delivered)
    t0 = time.perf_counter()
    ans, res = pr.answer(msgs, model=MODEL, num_ctx=num_ctx, num_predict=num_predict,
                         strict_ctx=strict_ctx)
    row = pr.score(ans, res, item, delivered,
                   in_gold=lambda sid: R.in_gold_section(sid, item))
    return {**row, "item_id": item.item_id, "category_item": item.category,
            "n_skipped": len(skipped), "wall_s": round(time.perf_counter() - t0, 2),
            # las dos columnas de recuperación, para la 2x2: ¿llegó la sección? ¿llegó la clave?
            "sec_hit": R.section_hit(delivered, item),
            "key_delivered": answer_delivered(delivered, item),
            "answer_text": (ans.answer if ans else ""),
            "analysis_text": (ans.analysis if ans else "")}


log = store.ResultsLog(ART / "results" / "04_generation.jsonl",
                       run_spec={"stage": "04_generation", "eval": ev.key(), "model": MODEL,
                                 "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
                                 "deliver_tokens": DELIVER_TOKENS, "index": R.index_key})

for name, make in ROW_PASSAGES.items():
    t0 = time.perf_counter()
    todo = [it for it in TEST if not log.done(name, it.item_id)]
    for it in todo:
        log.write({"arm": name, **run_item(it, make(it))})
    print(f"{name:20} {len(todo):3} ítems en {time.perf_counter() - t0:6.1f} s")

ROWS = {name: {r["item_id"]: r for r in log.rows(name)} for name in ROW_PASSAGES}
IDS = [it.item_id for it in TEST]
item_by_id = {it.item_id: it for it in ITEMS}

# %%
ANSWERABLE = [i for i in IDS if item_by_id[i].answerable]
UNANSWERABLE = [i for i in IDS if not item_by_id[i].answerable]


def rate(name: str, ids: list[str], field: str) -> float:
    vals = [ROWS[name][i][field] for i in ids if i in ROWS[name]]
    vals = [v for v in vals if v is not None]
    return sum(bool(v) for v in vals) / len(vals) if vals else float("nan")


def n_with(name: str, ids: list[str], field: str, value=True) -> int:
    return sum(1 for i in ids if i in ROWS[name] and ROWS[name][i][field] == value)


table = []
for name in ROW_PASSAGES:
    rows = [ROWS[name][i] for i in IDS if i in ROWS[name]]
    ci = noise.boot_ci([int(bool(ROWS[name][i]["correct"])) for i in ANSWERABLE
                        if ROWS[name][i]["correct"] is not None])
    table.append((name, {
        "correctas": f"{rate(name, ANSWERABLE, 'correct'):.0%}",
        "IC95": f"{ci[0]:.0%}-{ci[1]:.0%}",
        "abstuvo": f"{rate(name, ANSWERABLE, 'abstained'):.0%}",
        "abstuvo bien (5 s/resp.)": f"{n_with(name, UNANSWERABLE, 'abstained')}/{len(UNANSWERABLE)}",
        "citas ⊆ entregadas": f"{rate(name, IDS, 'citations_subset'):.0%}",
        "citó gold": f"{rate(name, ANSWERABLE, 'cited_gold'):.0%}",
        "fallos": n_with(name, IDS, "category", "fallo"),
        "tokens prompt": int(np.mean([r["prompt_tokens"] or 0 for r in rows])),
        "seg": round(float(np.mean([r["wall_s"] for r in rows])), 1),
    }))
print_table(f"las tres filas · {len(ANSWERABLE)} ítems con respuesta en el corpus, "
            f"{len(UNANSWERABLE)} sin ella", list(table[0][1]), table,
            smoke=SMOKE_TEST, name_header="fila")

for a, b in (("libro cerrado", "mejor recuperación"), ("mejor recuperación", "oráculo")):
    p = noise.paired([ROWS[a][i]["correct"] for i in ANSWERABLE],
                     [ROWS[b][i]["correct"] for i in ANSWERABLE])
    print(f"{b:20} vs {a:20} -> gana {p['gana']}, pierde {p['pierde']}, empata {p['empata']}, "
          f"p = {p['p']} {'(sobrevive)' if p['p'] < 0.05 else '(no sobrevive al piso de ruido)'}")
print(f"\nhacen falta {noise.detectable()} discordantes todos para el mismo lado para bajar de "
      f"p < 0,05 con este n")

# %% [markdown]
r"""
## 4 · La tabla 2×2, y las dos fallas que separa

Por pregunta: **recuperó bien** (llegó la sección gold) × **respondió bien**. Las dos casillas
que importan no son los aciertos:

- **fiel al pasaje equivocado**: recuperó mal, y respondió igual lo que decía ese pasaje. Es la
  falla que un sistema «bien portado» produce de todos modos: el modelo citó, no inventó, y la
  respuesta está mal.
- **lo tenía y respondió otra cosa**: recuperó bien, y respondió mal. Acá el recuperador no tiene
  nada más que dar, y es donde se ve el techo del generador.
"""

# %%
cell = collections.Counter()
examples: dict[tuple[bool, bool], list[str]] = collections.defaultdict(list)
for i in ANSWERABLE:
    row = ROWS["mejor recuperación"][i]
    if row["correct"] is None:
        continue
    k = (bool(row["sec_hit"]), bool(row["correct"]))
    cell[k] += 1
    examples[k].append(i)
print_table("la 2×2 de «mejor recuperación»", ["respondió bien", "respondió mal"],
            [("recuperó bien", {"respondió bien": cell[(True, True)],
                                "respondió mal": cell[(True, False)]}),
             ("recuperó mal", {"respondió bien": cell[(False, True)],
                               "respondió mal": cell[(False, False)]})],
            smoke=SMOKE_TEST, name_header="")
total = sum(cell.values()) or 1
print(f"fiel al pasaje equivocado (recuperó mal, respondió mal): {cell[(False, False)]} "
      f"({cell[(False, False)] / total:.0%})")
print(f"lo tenía y respondió otra cosa (recuperó bien, respondió mal): {cell[(True, False)]} "
      f"({cell[(True, False)] / total:.0%})")

for iid in examples[(True, False)][:3]:
    row, it = ROWS["mejor recuperación"][iid], item_by_id[iid]
    print(f"\nLO TENÍA Y FALLÓ · {iid}: {clip(it.question, 80)}")
    print(f"   clave {it.answer_key} · clave entregada: {row['key_delivered']} · "
          f"se abstuvo: {row['abstained']}")
    print(wrap(row["answer_text"], max_lines=3))

# %%
# Las tres cuentas que el enunciado pide aparte, porque son afirmaciones del modelo, no métricas:
best = ROWS["mejor recuperación"]
hallucinated = [i for i in IDS if best[i].get("cited_not_delivered")]
abstained_with_key = [i for i in ANSWERABLE
                      if best[i]["abstained"] and best[i]["key_delivered"]]
inconsistent = [i for i in IDS if best[i].get("inconsistent")]
failures = [i for i in IDS if best[i]["category"] == "fallo"]
print(f"citó un pasaje que no se le entregó: {len(hallucinated)} de {len(IDS)} "
      f"{[(i, best[i]['cited_not_delivered'][:1]) for i in hallucinated[:4]]}")
print(f"se abstuvo con la clave en su contexto:  {len(abstained_with_key)} de {len(ANSWERABLE)} "
      f"{abstained_with_key[:6]}")
print(f"booleano y texto en desacuerdo:          {len(inconsistent)} {inconsistent[:6]}")
print(f"llamadas en la categoría «fallo» (ni abstención ni respuesta): {len(failures)} "
      f"{[(i, clip(str(best[i]['error']), 60)) for i in failures[:3]]}")

# %% [markdown]
r"""
## 5 · Mirar saltar la guarda, una vez y a propósito

Con `strict_ctx=True` (el default) hay **dos** guardas: la nuestra, que cuenta el prompt con la
plantilla antes de mandarlo y levanta `ContextOverflow`; y la del servidor, al que se le pide que
**rechace** en lugar de truncar. Con `strict_ctx=False` no hay ninguna de las dos: el servidor
conserva los primeros tokens y la **cola**, y descarta el medio — se va el sistema con la regla
de abstención y se van los pasajes mejor rankeados, sobrevive la pregunta del final. El modelo
responde con fluidez, `format` igual devuelve JSON válido, y la conclusión equivocada que sale de
ahí es «el reranker no ayuda a la generación».

Se hace en un ítem de **dev**, con un `num_ctx` chico a propósito.
"""

# %%
SMALL_CTX = 2048
it = probe_item
passages = R.retrieve(it.question, BEST_CFG)
msgs = pr.messages(it.question, passages)          # sin presupuestar: a propósito no entra
mine = llm.count_prompt_tokens(msgs, MODEL)
print(f"prompt de {mine} tokens contra num_ctx {SMALL_CTX} (reserva {256})")

_, blocked = pr.answer(msgs, model=MODEL, num_ctx=SMALL_CTX, num_predict=256, strict_ctx=True)
print(f"\ncon la guarda puesta: {clip(str(blocked.error), 150)}")
print("   no se mandó nada: la guarda cuenta el prompt ANTES de la llamada")

loose, res = pr.answer(msgs, model=MODEL, num_ctx=SMALL_CTX, num_predict=256, strict_ctx=False)
print(f"\ncon la guarda apagada: nuestra cuenta {mine} · el servidor evaluó {res.prompt_tokens} "
      f"· done_reason {res.done_reason}")
print(f"   error: {clip(str(res.error), 200)}")
print(f"   el servidor perdió {(mine or 0) - (res.prompt_tokens or 0)} tokens del medio del prompt")
if loose:
    print(f"   y aun así respondió: can_answer={loose.can_answer} · {clip(loose.answer, 160)}")
    print("   fluida, con JSON válido, y construida con los pasajes peor rankeados")
run.note("guarda apagada", {"nuestra cuenta": mine, "servidor": res.prompt_tokens,
                            "error": res.error})

# %% [markdown]
r"""
## 6 · Posición del pasaje (opcional, y con su tamaño de prueba declarado)

El mismo conjunto de pasajes, permutado: el que trae la respuesta **primero**, **en el medio** o
**último**. Misma cantidad de pasajes y mismos tokens en las tres —es una permutación, no un
recorte—, así que lo único que cambia es la posición.

Lo que esta prueba **no** puede hacer: con ~10 pasajes en 3.000 tokens, un resultado nulo no
refuta *Lost in the Middle*. El efecto necesita una ventana en la que valga la pena perderse.
El tamaño declarado está impreso abajo: con este n, hacen falta `noise.detectable()`
discordantes todos para el mismo lado para bajar de p < 0,05.
"""

# %%
if POSITION_TEST:
    def permute(ps: list[dict], at: int) -> list[dict]:
        rest = [p for j, p in enumerate(ps) if j != at]
        g = ps[at]
        mid = len(rest) // 2
        return {"primero": [g] + rest, "medio": rest[:mid] + [g] + rest[mid:],
                "último": rest + [g]}


    eligible = []
    for it in TEST:
        if not it.gold or not it.answer_key:
            continue
        ps = R.retrieve(it.question, BEST_CFG)
        hit = [j for j, p in enumerate(ps) if R.in_gold_section(p["section_id"], it)]
        if hit and len(ps) >= 5:
            eligible.append((it, ps, hit[0]))
    print(f"ítems donde el pasaje con la respuesta llegó: {len(eligible)} · "
          f"{int(np.median([len(p) for _, p, _ in eligible])) if eligible else 0} pasajes de "
          f"mediana")

    pos_rows: dict[str, list] = {"primero": [], "medio": [], "último": []}
    for it, ps, at in eligible:
        variants = permute(ps, at)
        for where, ordered in variants.items():
            arm = f"posición: {where}"
            if not log.done(arm, it.item_id):
                log.write({"arm": arm, **run_item(it, ordered)})
    skipped_any = 0
    for where in pos_rows:
        rows = {r["item_id"]: r for r in log.rows(f"posición: {where}")}
        pos_rows[where] = [rows[it.item_id]["correct"] for it, _, _ in eligible
                           if it.item_id in rows]
        skipped_any += sum(r["n_skipped"] for r in rows.values())
    print(verdict(skipped_any == 0,
                  "las tres posiciones entregaron el mismo conjunto de pasajes: es una "
                  "permutación, no un recorte",
                  f"{skipped_any} pasajes salteados: el presupuesto dejó afuera pasajes "
                  "distintos según el orden, y entonces esto no mide posición"))

    print_table("posición del pasaje con la respuesta", ["correctas", "n"],
                [(where, {"correctas": f"{sum(bool(v) for v in vals) / max(len(vals), 1):.0%}",
                          "n": len(vals)}) for where, vals in pos_rows.items()],
                smoke=SMOKE_TEST, name_header="posición")
    p_first_last = noise.paired(pos_rows["último"], pos_rows["primero"])
    p_first_mid = noise.paired(pos_rows["medio"], pos_rows["primero"])
    print(f"primero vs último: {p_first_last} · primero vs medio: {p_first_mid}")
    print(f"tamaño de prueba: hacen falta {noise.detectable()} discordantes para el mismo lado; "
          f"hubo {p_first_last['gana'] + p_first_last['pierde']} entre primero y último")
    print(verdict(p_first_last["p"] < 0.05,
                  "la posición movió la fila por encima del piso de ruido",
                  "a esta escala la posición no se distingue del ruido -- y con ~10 pasajes en "
                  f"{DELIVER_TOKENS} tokens la prueba no podía mostrarlo salvo que el efecto "
                  "fuera enorme"))
else:
    pos_rows, p_first_last = {}, {}
    print("prueba de posición apagada (POSITION=0)")

# %% [markdown]
r"""
## 7 · Las respuestas del informe

**De las preguntas en las que llegó el pasaje con la respuesta, ¿cuántas se respondieron bien?**
Es la fila «recuperó bien» de la 2×2: los aciertos a la izquierda, y a la derecha las que *lo
tenían* y fallaron igual. Los tres ejemplos impresos arriba son esas fallas, con su clave y su
texto, para poder decir **cómo** fallaron (se abstuvo, respondió otra cosa, o respondió lo que
decía otro pasaje).

**¿Qué tan grande es la brecha entre el oráculo y la mejor recuperación? ¿Y entre libro cerrado y
la mejor?** Las dos están en la tabla de §3 con su test pareado. La primera acota lo que queda
por ganar mejorando el recuperador; la segunda dice cuánto aporta el corpus sobre lo que el
modelo ya sabía — y en un corpus de documentación de Python, que es público y viejo, esperar que
libro cerrado dé cero es ingenuo.

**¿Con qué frecuencia citó algo que no se le dio, o se abstuvo con la respuesta en el contexto?**
Las dos cuentas están al final de §4. La primera es la que vuelve verificable una cita; la
segunda es la falla que una demo nunca muestra.

**Un prompt demasiado grande**: §5, con nuestra cuenta, la del servidor y lo que respondió.

**¿Importó la posición?** §6, con el tamaño de prueba declarado.
"""

# %%
gap_oracle = rate("oráculo", ANSWERABLE, "correct") - rate("mejor recuperación", ANSWERABLE, "correct")
gap_closed = rate("mejor recuperación", ANSWERABLE, "correct") - rate("libro cerrado", ANSWERABLE, "correct")
print(f"brecha oráculo - mejor recuperación: {gap_oracle:+.0%} (lo que queda por ganar "
      f"arreglando el recuperador)")
print(f"brecha mejor recuperación - libro cerrado: {gap_closed:+.0%} (lo que aporta el corpus)")

store.write_json(ART / "stage4_rows.json", {
    "model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
    "table": {n: v for n, v in table},
    "two_by_two": {f"{k[0]}|{k[1]}": v for k, v in cell.items()},
    "gaps": {"oracle_minus_best": gap_oracle, "best_minus_closed": gap_closed},
    "cited_not_delivered": hallucinated, "abstained_with_key": abstained_with_key,
    "inconsistent": inconsistent, "failures": failures,
    "oversize": {"ours": mine, "server": res.prompt_tokens, "error": res.error},
    "position": {k: [bool(v) for v in vals] for k, vals in pos_rows.items()},
})

run.note("correctas (mejor recuperación)", round(rate("mejor recuperación", ANSWERABLE, "correct"), 3))
run.note("correctas (oráculo)", round(rate("oráculo", ANSWERABLE, "correct"), 3))
run.note("correctas (libro cerrado)", round(rate("libro cerrado", ANSWERABLE, "correct"), 3))
run.note("lo tenía y falló", cell[(True, False)])
run.note("fiel al pasaje equivocado", cell[(False, False)])
run.note("citas inventadas", len(hallucinated))
run.note("abstenciones con la clave entregada", len(abstained_with_key))
run.finish()
