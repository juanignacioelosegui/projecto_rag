# %% [markdown]
r"""
# Etapa 7a · Una segunda lectura: un juez, calibrado antes de leerlo

Las Etapas 4-6 puntuaron cada respuesta por **contención**: la clave tiene que estar en el texto,
el sí/no tiene que coincidir, y en un ítem sin respuesta lo correcto es abstenerse. Es barato,
reproducible y ciego a la paráfrasis: `os.path.join` no contiene `pathlib`, pero «uní las partes
con `Path(a) / b`» puede ser la misma respuesta. Acá un **juez LLM** relee las respuestas
guardadas — sin generar ni una nueva — y la pregunta de la etapa no es «qué puntaje da el juez»
sino **cuándo coincide con la contención, cuándo no, y en cuál de los dos confiar**.

**Entra**: las filas ya escritas de `results/04_generation.jsonl`, `05_dialogue_b.jsonl` y
`06_memory.jsonl` (el `answer_text` de cada una) y el conjunto congelado, que da la pregunta y la
clave. **Sale**: `artifacts/<perfil>/results/07_judge.jsonl`, `stage7_rows.json`, y
`stage7_audit.txt` para leer a mano.

**Las decisiones que esta etapa no puede no tomar:**

1. **El juez recibe campos separados y etiquetados**: pregunta, clave y respuesta, cada uno en su
   bloque. Nunca el contexto recuperado: acá no se mide fidelidad al pasaje, y con el pasaje
   delante el juez premia ser fiel al pasaje equivocado.
2. **Una rúbrica por tipo de ítem** (reglas al estilo LongMemEval): con clave, sin respuesta en el
   corpus (abstención), actualización de conocimiento (vale el valor más nuevo) y preferencia.
3. **Veredicto estructurado, temperatura 0**, con `analysis` declarado primero para que el modelo
   razone antes de comprometerse (el mismo orden de claves que midió `llm.setup` en la Etapa 4).
4. **Autopreferencia, nombrada y medida.** El juez principal es el **mismo modelo que generó**
   las respuestas. Una segunda pasada, separada, usa un juez de **otra familia** (Granite, de
   IBM) sobre las mismas respuestas guardadas: si los dos jueces discrepan sistemáticamente en un
   sentido, eso es una medida de cuánto pesa la autopreferencia.
5. **Un juez es un instrumento**: su promedio no se lee hasta haberlo comparado contra la
   contención y contra una lectura a mano de max(30, 20 %) de sus juicios.
"""

# %%
import environment  # noqa: F401  (efectos secundarios: guarda de JAX, caché de HF, bundle TLS, UTF-8)

environment.summary()

import collections
import os
import random
import time
from typing import Literal

from pydantic import BaseModel, Field

import llm
import noise
import retrieval as rt
import store
from console import StageRun, clip, print_table, verdict, wrap
from schema import load_eval

SMOKE_TEST = os.environ.get("SMOKE_TEST", "1").lower() not in ("0", "false", "no")
ART = environment.artifacts_dir(SMOKE_TEST)
PROFILE = "smoke" if SMOKE_TEST else "full"
EVAL = environment.ROOT / ("eval/smoke" if SMOKE_TEST else "eval")
GENERATOR = rt.models(SMOKE_TEST)["generator"]

# el juez principal ES el generador (autopreferencia); el segundo es de otra familia
JUDGES = {"mismo modelo": GENERATOR,
          "otra familia": "granite3.3:2b" if SMOKE_TEST else "granite3.3:8b"}
llm.HF_TOKENIZERS.setdefault("granite3.3:2b", "ibm-granite/granite-3.3-2b-instruct")
llm.HF_TOKENIZERS.setdefault("granite3.3:8b", "ibm-granite/granite-3.3-8b-instruct")

NUM_CTX = 4096            # el juez lee pregunta + clave + respuesta: no necesita la ventana de 8.192
NUM_PREDICT = 400
AUDIT_MIN, AUDIT_FRAC = 30, 0.20
SEED = 0

environment.seed_everything(SEED)
run = StageRun.start("07_second_read", ART, profile=PROFILE, config={
    "judges": JUDGES, "generator": GENERATOR, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
    "seed": SEED,
})

ev = load_eval(EVAL, require_frozen=not SMOKE_TEST,
               documents_path=None if SMOKE_TEST else ART / "documents.jsonl")

# %% [markdown]
r"""
## 1 · Las respuestas guardadas

Cada archivo de resultados puede tener filas de varias configuraciones (una corrida vieja, un
cambio de prompt): `ResultsLog` las separa por `config_key`. Acá se toma la **última**
configuración escrita en cada archivo —la que produjo las tablas de esa etapa— y, dentro de ella,
la última fila de cada (brazo, ítem). Las filas que fallaron (`correct is None`: sin respuesta
parseable) no se juzgan: no hay texto que leer.

De la Etapa 4 entran solo las tres filas de la tabla principal (las de posición repiten las mismas
preguntas con los pasajes reordenados). Cada fila se vuelve un **caso**: la pregunta que vio el
modelo (en la Etapa 5, la reescritura oráculo si el turno depende del historial, para que el juez
entienda de qué se habla), la clave, el tipo de rúbrica y el veredicto de la contención.
"""


# %%
def latest_config(path) -> list[dict]:
    if not path.is_file():
        return []
    rows = list(store.read_jsonl(path))
    if not rows:
        return []
    key = rows[-1].get("config_key")
    latest = {}
    for r in rows:
        if r.get("config_key") == key:
            latest[(r["arm"], r["item_id"])] = r
    return list(latest.values())


ITEMS = {i.item_id: i for i in ev.items}
TURNS = {f"{d.dialogue_id}#{t.turn}": (d, t) for d in ev.dialogues for t in d.turns}
PROBES = {f"{s.scenario_id}#{p.probe_id}": (s, p) for s in ev.scenarios for p in s.probes}
STAGE4_ARMS = ("libro cerrado", "mejor recuperación", "oráculo")


def rubric_of(target, ability: str | None = None) -> str:
    if not target.answerable:
        return "abstención"
    if ability == "knowledge_update":
        return "actualización"
    if ability == "preference":
        return "preferencia"
    if target.answer_key or target.expected_verdict:
        return "clave"
    return "sin clave"


CASES: list[dict] = []
for r in latest_config(ART / "results" / "04_generation.jsonl"):
    it = ITEMS.get(r["item_id"])
    if r["arm"] in STAGE4_ARMS and it is not None:
        CASES.append({"stage": "04", "arm": r["arm"], "item_id": r["item_id"], "question": it.question,
                      "target": it, "rubric": rubric_of(it), "row": r})
for r in latest_config(ART / "results" / "05_dialogue_b.jsonl"):
    if r["item_id"] in TURNS:
        d, t = TURNS[r["item_id"]]
        CASES.append({"stage": "05", "arm": r["arm"], "item_id": r["item_id"],
                      "question": t.oracle_rewrite or t.user, "target": t, "rubric": rubric_of(t),
                      "row": r})
for r in latest_config(ART / "results" / "06_memory.jsonl"):
    if r["item_id"] in PROBES:
        s, p = PROBES[r["item_id"]]
        CASES.append({"stage": "06", "arm": r["arm"], "item_id": r["item_id"], "question": p.question,
                      "target": p, "rubric": rubric_of(p, p.ability), "row": r})

n_failed = sum(c["row"].get("correct") is None for c in CASES)
CASES = [c for c in CASES if c["row"].get("correct") is not None]
print_table("casos a juzgar", ["casos", "rúbricas"],
            [(f"Etapa {st}", {"casos": sum(c["stage"] == st for c in CASES),
                              "rúbricas": dict(collections.Counter(c["rubric"] for c in CASES
                                                                   if c["stage"] == st))})
             for st in ("04", "05", "06")], smoke=SMOKE_TEST, name_header="etapa")
print(f"{len(CASES)} casos · {n_failed} filas sin respuesta parseable quedan afuera "
      "(su propia categoría en cada etapa, no algo que un juez pueda leer)")
print(verdict(len(CASES) > 0, "hay respuestas guardadas para juzgar",
              "no hay filas: corré las Etapas 4-6 en este perfil antes que esta"))
run.note("casos", len(CASES))

# %% [markdown]
r"""
## 2 · El juez

El prompt de sistema dice **qué** se juzga y nada sobre quién respondió. La rúbrica cambia por
tipo de caso, porque la misma respuesta («no está en la documentación») es correcta para una
pregunta fuera del corpus e incorrecta para una que sí tiene respuesta:

| rúbrica | correcta si… |
|---|---|
| `clave` | dice lo que dice la clave (paráfrasis, otra notación u otro orden valen); con sí/no, arranca con el correcto |
| `abstención` | dice que no sabe o que la documentación no lo cubre; inventar una respuesta es incorrecto |
| `actualización` | usa el valor **más reciente** de la clave; mencionar el viejo como historia vale, darlo como vigente no |
| `preferencia` | respeta la preferencia que fija la clave |
| `sin clave` | responde la pregunta de forma correcta y concreta para Python (el juez sin referencia: se reporta aparte) |

La rúbrica `sin clave` es la que **más** depende del conocimiento propio del juez, y por eso no
entra en la matriz contra la contención (que ahí da «correcta» por defecto): se cuenta aparte.
"""

# %%
SYSTEM = ("Sos un evaluador de respuestas sobre la documentación de Python en español. Recibís una "
          "PREGUNTA, una CLAVE (lo que una respuesta correcta tiene que decir) y una RESPUESTA a "
          "evaluar, cada una en su propio bloque. Juzgá SOLO si la RESPUESTA es correcta según la "
          "REGLA dada. No premies el largo, el tono ni la seguridad. Escribí primero un análisis "
          "breve (1-3 oraciones) y después el veredicto.")

RULES = {
    "clave": ("La respuesta es correcta si afirma lo que dice la CLAVE, aunque lo diga con otras "
              "palabras, en otra notación o en otro orden. Es incorrecta si la contradice, si no lo "
              "dice, o si se abstiene. Si la clave incluye un sí/no esperado, la respuesta tiene que "
              "dar ese sí/no."),
    "abstención": ("La pregunta NO tiene respuesta en la documentación disponible. La respuesta es "
                   "correcta solo si se abstiene: dice que no sabe, que no tiene la información o que "
                   "la documentación no lo cubre. Es incorrecta si da una respuesta concreta."),
    "actualización": ("El usuario cambió un dato a lo largo del tiempo; la CLAVE tiene el valor "
                      "VIGENTE. La respuesta es correcta si da el valor vigente. Mencionar el valor "
                      "anterior como historia está bien; darlo como el actual es incorrecto."),
    "preferencia": ("La CLAVE indica una preferencia del usuario sobre cómo responder. La respuesta es "
                    "correcta si la respeta y responde la pregunta."),
    "sin clave": ("No hay clave de referencia. La respuesta es correcta si responde la pregunta de "
                  "forma correcta y concreta según la documentación de Python; es incorrecta si es "
                  "falsa, evasiva o se abstiene sin motivo."),
}


class Judgment(BaseModel):
    analysis: str = Field(description="1-3 oraciones: qué dice la respuesta frente a la regla")
    verdict: Literal["correcta", "incorrecta"]


def key_text(target) -> str:
    parts = []
    if target.answer_key:
        parts.append("; ".join(target.answer_key))
    if target.expected_verdict:
        parts.append(f"sí/no esperado: {'sí' if target.expected_verdict == 'yes' else 'no'}")
    if not target.answerable:
        parts.append("(sin respuesta en la documentación)")
    return " · ".join(parts) or "(sin clave)"


def judge_messages(case: dict) -> list[dict]:
    user = (f"REGLA:\n{RULES[case['rubric']]}\n\n"
            f"PREGUNTA:\n{case['question']}\n\n"
            f"CLAVE:\n{key_text(case['target'])}\n\n"
            f"RESPUESTA:\n{case['row'].get('answer_text') or '(vacía)'}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


demo = next((c for c in CASES if c["rubric"] == "clave"), CASES[0] if CASES else None)
if demo:
    print(wrap(judge_messages(demo)[1]["content"], width=90, max_lines=20))

# %% [markdown]
r"""
## 3 · Las dos pasadas

Una pasada por juez, cada una con su `llm.setup` (pull, sondas, `placement`) y su propia caché de
llamadas: con `OLLAMA_MAX_LOADED_MODELS=1` el servidor descarga un modelo para cargar el otro, y
las dos pasadas nunca se mezclan en VRAM. Las filas van a `07_judge.jsonl` con el juez en el
brazo, así que un corte a mitad de una pasada se reanuda donde quedó.
"""

# %%
log = store.ResultsLog(ART / "results" / "07_judge.jsonl",
                       run_spec={"stage": "07_second_read", "eval": ev.key(), "judges": JUDGES,
                                 "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
                                 "system": store.sha256_text(SYSTEM)[:16],
                                 "rules": store.sha256_json(RULES)[:16]})


def case_id(c: dict) -> str:
    return f"{c['stage']}|{c['arm']}|{c['item_id']}"


PLACEMENT = {}
for jname, jmodel in JUDGES.items():
    todo = [c for c in CASES if not log.done(jname, case_id(c))]
    if not todo:
        print(f"{jname} ({jmodel}): {len(CASES)} casos ya juzgados")
        continue
    llm.use_cache(ART / "llm_cache" / f"judge-{jmodel.replace(':', '-')}.jsonl")
    try:
        info = llm.setup(jmodel, num_ctx=NUM_CTX)
    except Exception as e:            # sin el segundo juez la etapa sigue, y lo dice
        print(f"CHECK -- {jname} ({jmodel}) no se pudo preparar: {type(e).__name__}: {e}")
        run.note(f"juez {jname}", f"no corrió: {e}")
        continue
    PLACEMENT[jname] = info.get("placement")
    t0 = time.perf_counter()
    for c in todo:
        j, res = llm.structured(judge_messages(c), Judgment, model=jmodel, num_ctx=NUM_CTX,
                                num_predict=NUM_PREDICT, temperature=0.0)
        log.write({"arm": jname, "item_id": case_id(c), "judge_model": jmodel,
                   "stage": c["stage"], "gen_arm": c["arm"], "case": c["item_id"],
                   "rubric": c["rubric"], "contention": bool(c["row"]["correct"]),
                   "judge": (j.verdict == "correcta") if j else None,
                   "analysis": j.analysis if j else "", "error": res.error,
                   "prompt_tokens": res.prompt_tokens, "output_tokens": res.output_tokens,
                   "ms": res.ms})
    print(f"{jname} ({jmodel}): {len(todo)} casos en {time.perf_counter() - t0:.0f} s")
run.note("placement de los jueces", PLACEMENT)

J = {jname: {r["item_id"]: r for r in log.rows(jname)} for jname in JUDGES}

# %% [markdown]
r"""
## 4 · El juez contra la contención

La matriz es por juez y por etapa, sobre los casos **con referencia** (todas las rúbricas menos
`sin clave`). Las dos celdas fuera de la diagonal no son lo mismo:

- **juez sí / contención no**: casi siempre paráfrasis (la clave no aparece literal) — o un juez
  indulgente. Es la celda donde el juez *puede* agregar información.
- **juez no / contención sí**: la clave está pero la respuesta dice otra cosa alrededor («no es
  `pathlib`, es `os.path`» contiene `pathlib`), o un juez estricto.

Cuál de las dos lecturas es la cierta no lo dice ninguna de las dos: lo dice §6.
"""


# %%
def confusion(rows: list[dict]) -> dict:
    cells = collections.Counter((r["judge"], r["contention"]) for r in rows if r["judge"] is not None)
    n = sum(cells.values())
    agree = cells[(True, True)] + cells[(False, False)]
    return {"n": n, "juez sí·cont sí": cells[(True, True)], "juez sí·cont no": cells[(True, False)],
            "juez no·cont sí": cells[(False, True)], "juez no·cont no": cells[(False, False)],
            "acuerdo": round(agree / n, 3) if n else None,
            "sin veredicto": sum(r["judge"] is None for r in rows)}


MATRIX = {}
for jname in JUDGES:
    keyed = [r for r in J[jname].values() if r["rubric"] != "sin clave"]
    if not keyed:
        continue
    rows = [(f"Etapa {st}", confusion([r for r in keyed if r["stage"] == st])) for st in ("04", "05", "06")]
    rows.append(("todas", confusion(keyed)))
    MATRIX[jname] = dict(rows)
    print_table(f"juez «{jname}» ({JUDGES[jname]}) vs contención — casos con referencia",
                ["n", "juez sí·cont sí", "juez sí·cont no", "juez no·cont sí", "juez no·cont no",
                 "acuerdo", "sin veredicto"], rows, smoke=SMOKE_TEST, name_header="etapa")

for jname in MATRIX:
    by_rubric = collections.defaultdict(list)
    for r in J[jname].values():
        by_rubric[r["rubric"]].append(r)
    print_table(f"juez «{jname}» por rúbrica", ["n", "acuerdo", "juez sí·cont no", "juez no·cont sí"],
                [(k, confusion(v)) for k, v in sorted(by_rubric.items())],
                smoke=SMOKE_TEST, name_header="rúbrica")

# %% [markdown]
r"""
## 5 · Autopreferencia: el mismo modelo contra otra familia

Sobre los mismos casos, ¿el juez que generó las respuestas las aprueba más que un juez ajeno? La
comparación es **pareada**: solo informan los casos en que los dos jueces discrepan, y el test de
signos dice si la asimetría es más que ruido. Una diferencia que va en el sentido del mismo
modelo es *compatible* con autopreferencia, no la prueba: el juez de otra familia también puede
ser más estricto en general. Por eso se mira también **contra la contención**: si el juez propio
se aparta de ella más hacia el «sí» que el ajeno, el exceso es el candidato a autopreferencia.
"""

# %%
SELF, OTHER = "mismo modelo", "otra familia"
both = [k for k in J[SELF] if k in J.get(OTHER, {}) and J[SELF][k]["judge"] is not None
        and J[OTHER][k]["judge"] is not None]
SELFPREF = None
if both:
    a = [J[SELF][k]["judge"] for k in both]
    b = [J[OTHER][k]["judge"] for k in both]
    SELFPREF = noise.paired(a, b)
    lenient = {jn: sum(J[jn][k]["judge"] and not J[jn][k]["contention"] for k in both) for jn in (SELF, OTHER)}
    print_table("aprobación pareada: mismo modelo vs otra familia", ["aprueba", "de", "sí sin contención"],
                [(jn, {"aprueba": sum(J[jn][k]["judge"] for k in both), "de": len(both),
                       "sí sin contención": lenient[jn]}) for jn in (SELF, OTHER)],
                smoke=SMOKE_TEST, name_header="juez")
    print(f"pareado: {SELFPREF}")
    print(f"acuerdo entre jueces: {sum(x == y for x, y in zip(a, b))} de {len(both)}")
    run.note("autopreferencia (pareado)", SELFPREF)
else:
    print("CHECK -- no hay casos juzgados por los dos jueces: la autopreferencia queda sin medir")

# %% [markdown]
r"""
## 6 · La auditoría a mano

El enunciado pide leer al menos 30 juicios o el 20 %, lo que sea mayor. La muestra no es al azar
puro: primero entran **todos** los casos donde el juez principal y la contención discrepan (es
donde hay algo que decidir), y el resto se completa al azar con semilla fija entre los que
coinciden, para poder medir también si el acuerdo es acuerdo en lo correcto.

`stage7_audit.txt` deja cada caso con su pregunta, clave, respuesta, el análisis del juez y los
dos veredictos. El veredicto propio se carga en `stage7_audit.json` como
`{"<id del caso>": true | false, ...}`. Si ese archivo no está, la matriz contra la persona
queda **PENDIENTE**: ningún script puede decidir si una respuesta es correcta en lugar de quien
la lee.
"""

# %%
rng = random.Random(SEED)
primary = [r for r in J[SELF].values() if r["judge"] is not None]
n_audit = min(len(primary), max(AUDIT_MIN, round(AUDIT_FRAC * len(primary))))
disagree = [r for r in primary if r["judge"] != r["contention"]]
agree = [r for r in primary if r["judge"] == r["contention"]]
rng.shuffle(disagree)
rng.shuffle(agree)
SAMPLE = (disagree + agree)[:n_audit]
CASE_BY_ID = {case_id(c): c for c in CASES}

lines = [f"Auditoría del juez — perfil {PROFILE} — {len(SAMPLE)} casos de {len(primary)} "
         f"({len([r for r in SAMPLE if r['judge'] != r['contention']])} con desacuerdo juez/contención)",
         "Cargá tu veredicto en stage7_audit.json como {\"<id>\": true|false}.", ""]
for r in SAMPLE:
    c = CASE_BY_ID.get(r["item_id"])
    if c is None:
        continue
    other = J.get(OTHER, {}).get(r["item_id"], {}).get("judge")
    lines += [f"== {r['item_id']}  [rúbrica: {r['rubric']}]",
              wrap(f"pregunta: {c['question']}", width=90, indent="  "),
              wrap(f"clave: {key_text(c['target'])}", width=90, indent="  "),
              wrap(f"respuesta: {c['row'].get('answer_text') or '(vacía)'}", width=90, indent="  "),
              wrap(f"juez ({JUDGES[SELF]}): {'correcta' if r['judge'] else 'incorrecta'} — {r['analysis']}",
                   width=90, indent="  "),
              f"  otra familia: {'—' if other is None else ('correcta' if other else 'incorrecta')}"
              f" · contención: {'correcta' if r['contention'] else 'incorrecta'}",
              "  tu veredicto -> ____", ""]
(ART / "stage7_audit.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"{len(SAMPLE)} casos en {ART / 'stage7_audit.txt'} (hay que leerlos a mano)")

HUMAN = None
try:
    human = store.read_json(ART / "stage7_audit.json")
    HUMAN = {}
    for source in (SELF, OTHER, "contención"):
        rows = []
        for cid, mine in human.items():
            if source == "contención":
                c = CASE_BY_ID.get(cid)
                said = bool(c["row"]["correct"]) if c else None
            else:
                said = J.get(source, {}).get(cid, {}).get("judge")
            if said is not None and mine is not None:
                rows.append({"judge": said, "contention": bool(mine)})
        HUMAN[source] = confusion(rows)
    print_table("contra la lectura a mano (columnas «cont» = la persona)",
                ["n", "juez sí·cont sí", "juez sí·cont no", "juez no·cont sí", "juez no·cont no", "acuerdo"],
                HUMAN.items(), smoke=SMOKE_TEST, name_header="fuente")
except FileNotFoundError:
    print("falta stage7_audit.json: la matriz del juez contra la persona está PENDIENTE "
          f"(a mano sobre {ART / 'stage7_audit.txt'})")

# %%
example = [r for r in disagree if r["judge"] and not r["contention"]][:2] + \
          [r for r in disagree if not r["judge"] and r["contention"]][:2]
for r in example:
    c = CASE_BY_ID[r["item_id"]]
    print(f"\n{r['item_id']} · juez {'sí' if r['judge'] else 'no'} / contención "
          f"{'sí' if r['contention'] else 'no'}")
    print(wrap(f"clave: {key_text(c['target'])}", width=90))
    print(wrap(f"respuesta: {clip(c['row'].get('answer_text') or '', 300)}", width=90))
    print(wrap(f"juez: {r['analysis']}", width=90))

no_key = {jn: [r for r in J[jn].values() if r["rubric"] == "sin clave" and r["judge"] is not None]
          for jn in J}
store.write_json(ART / "stage7_rows.json", {
    "judges": JUDGES, "placement": PLACEMENT, "cases": len(CASES), "failed_rows_skipped": n_failed,
    "matrix": MATRIX, "self_preference": SELFPREF, "audit_sample": len(SAMPLE), "human": HUMAN,
    "no_key_approved": {jn: [sum(r["judge"] for r in v), len(v)] for jn, v in no_key.items()},
})

for jname, m in MATRIX.items():
    run.note(f"acuerdo con la contención ({jname})", m["todas"]["acuerdo"])
run.note("auditoría a mano", "hecha" if HUMAN else f"PENDIENTE ({len(SAMPLE)} casos)")
run.finish()

# %% [markdown]
r"""
## 7 · Las respuestas del informe

**¿Dónde el juez no coincidió con la contención y con la auditoría, y en cuál confiás?** §4 da
las dos celdas de desacuerdo por etapa y por rúbrica, y §6 las cruza con la lectura a mano. La
regla para elegir es la que la tabla contra la persona sostenga: si el juez acierta en las celdas
«juez sí / contención no», la contención estaba castigando paráfrasis y el juez es mejor
instrumento *ahí*; si no, la contención sigue siendo la métrica principal y el juez una alarma.

**¿Cuánto importa la autopreferencia, y cómo lo sabés?** §5: aprobación pareada del juez propio
contra el de otra familia, con el test de signos sobre los discordantes, y cuántos «sí» de cada
juez van contra la contención. Que el juez propio apruebe más es compatible con autopreferencia y
también con que Granite sea más estricto: la lectura a mano de §6 es lo que separa las dos.
"""
