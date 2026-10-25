# %% [markdown]
r"""
# Etapa 6 · Memoria entre sesiones

Entre sesiones no sobrevive nada que no escribas. Acá la memoria es un **corpus que el sistema
escribe en tiempo de ejecución**, a partir de una fuente que puede estar equivocada —su propia
extracción— y que hay que mantener actualizado.

El vocabulario es el de CoALA (Apéndice B.4), y cada pieza de este archivo ocupa un lugar en él:

- la ventana ensamblada de `prompting.messages` es la memoria **de trabajo**;
- las transcripciones de las sesiones pasadas (el brazo «reproducción») son memoria **episódica**;
- los hechos extraídos con un schema (el brazo «almacén») son memoria **semántica**;
- las preferencias sobre *cómo* responder («prefiero pathlib», «sin dependencias externas») son
  memoria **procedimental**, y se guardan con `kind="procedural"` para poder contarlas aparte.

**Entra**: los escenarios congelados de la Etapa 2 (`eval/scenarios.jsonl`), el recuperador de
`retrieval.py` y el ensamblador y el puntaje de `prompting.py` —los mismos de las Etapas 4 y 5.
**Sale**: `artifacts/<perfil>/memory/` (el almacén y su log de operaciones),
`artifacts/<perfil>/results/06_memory.jsonl`, `stage6_rows.json` y dos archivos para leer a mano
(`stage6_records.txt`, `stage6_filter.txt`).

**Las cuatro decisiones que esta etapa no puede no tomar:**

1. **El régimen se declara antes de medir.** La línea de base honesta es «reproducir todo lo que
   entre», y en la evaluación de Mem0 esa línea de base *gana* en exactitud. A nuestra escala la
   reproducción entra siempre (§1: 98-286 tokens de espacio por estado), así que la comparación se corre en
   **dos regímenes de espacio de memoria** —uno donde la reproducción entra entera y uno donde no—
   y cada afirmación dice en cuál está.
2. **Los dos brazos comparten el mismo espacio.** La transcripción y los registros van los dos al
   **espacio de memoria** del ensamblador (el `memory=` de `pr.messages`), con el mismo
   presupuesto. Si uno fuera al espacio de memoria y el otro mensajes de verdad, la comparación
   sería de representación *más* posición *más* marcadores de rol. La Etapa 5 ya midió mensajes vs
   bloque para el historial de *una* sesión; acá las sesiones pasadas no son la conversación
   actual, y una transcripción fechada dentro del contexto es lo que un sistema real reproduce.
3. **Se guardan solo los campos del schema que queremos conservar.** Una respuesta estructurada
   guardada entera arrastra su campo `analysis`, y entonces el almacén tiene **hipótesis escritas
   como hechos** («el usuario probablemente usa 3.11») que después se recuperan como cosas que el
   usuario dijo. Acá el registro se arma campo por campo, y §3 cuenta los registros con marcas de
   razonamiento o de hipótesis, más las filtraciones de `</think>` que contó `llm.chat`.
4. **Invalidar, no borrar.** «¿Qué había dicho antes?» tiene que seguir teniendo respuesta: un
   UPDATE le pone `valid_to` al registro viejo y escribe uno nuevo. La vista «al momento de la
   sonda» sale de comparar los timestamps **simulados** con la fecha de la sonda, así que un
   registro invalidado en una sesión posterior sigue vigente para una sonda anterior sin tener que
   guardar dos copias del almacén.

**El piso de ruido, dicho antes de las tablas**: 19 sondas de 3 escenarios. El test de signos no
puede bajar de p < 0,05 con menos de `noise.detectable()` sondas discordantes todas para el mismo
lado, y las sondas de un escenario no son independientes entre sí. Por eso la tabla principal es de
**conteos por habilidad** (3/4, no 75 %), como pide el enunciado.
"""

# %%
import environment  # noqa: F401  (efectos secundarios: guarda de JAX, caché de HF, bundle TLS, UTF-8)

environment.summary()

import collections
import re
import time
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

import lexical
import llm
import noise
import prompting as pr
import retrieval as rt
import store
from console import StageRun, clip, print_table, verdict, wrap
from retrieval import RetrievalConfig
from schema import load_eval

import os

SMOKE_TEST = os.environ.get("SMOKE_TEST", "1").lower() not in ("0", "false", "no")
ART = environment.artifacts_dir(SMOKE_TEST)
PROFILE = "smoke" if SMOKE_TEST else "full"
EVAL = environment.ROOT / ("eval/smoke" if SMOKE_TEST else "eval")
MODELS = rt.models(SMOKE_TEST)
MODEL = MODELS["generator"]

NUM_CTX = 8192          # fijo en TODOS los brazos, como en las Etapas 4 y 5
NUM_PREDICT = 700
PASS_TOKENS = 1000      # pasajes: fijo en todos los brazos. Lo que se mueve es la memoria
SLOT_WIDE = 400         # el espacio de memoria del §7.3 (300-1.000): la reproducción entra entera
SLOT_TIGHT = 120        # el régimen donde NO entra entera, pero SÍ entra algo (§1, con la cuenta)
MEM_MIN, MEM_MAX = 3, 10    # recuerdos por sonda (§7.3)
RECENCY_PER_HOUR = 0.995    # solo sobre los timestamps SIMULADOS del escenario
PROBE_GAP_H = 24        # la sonda se pregunta en una sesión NUEVA, un día después de la última
WRITE_NEIGHBOURS = 3    # recuerdos parecidos que ve el decisor de escritura
MAX_FACTS = 4           # hechos candidatos por sesión
SEED = 0

environment.seed_everything(SEED)
run = StageRun.start("06_memory", ART, profile=PROFILE, config={
    "model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "pass_tokens": PASS_TOKENS,
    "slots": [SLOT_WIDE, SLOT_TIGHT], "recency_per_hour": RECENCY_PER_HOUR,
    "probe_gap_h": PROBE_GAP_H, "neighbours": WRITE_NEIGHBOURS, "seed": SEED,
})

ev = load_eval(EVAL, require_frozen=not SMOKE_TEST,
               documents_path=None if SMOKE_TEST else ART / "documents.jsonl")
SCENARIOS = [s for s in ev.scenarios if s.split == "test"]
DEV_SCENARIOS = [s for s in ev.scenarios if s.split == "dev"]
PROBES = [(s, p) for s in SCENARIOS for p in s.probes]
ABILITIES = collections.Counter(p.ability for _, p in PROBES)
print(f"escenarios de test: {len(SCENARIOS)} · {sum(len(s.sessions) for s in SCENARIOS)} sesiones "
      f"· {len(PROBES)} sondas · {len(DEV_SCENARIOS)} escenario(s) de dev")
print(f"habilidades: {dict(ABILITIES)}")
print(f"sondas sin respuesta (nunca dicho): {sum(1 for _, p in PROBES if not p.answerable)} · "
      f"con gold del corpus: {sum(1 for _, p in PROBES if p.gold)}")

# %%
llm.use_cache(ART / "llm_cache" / f"{MODEL.replace(':', '-')}.jsonl")
info = llm.setup(MODEL, num_ctx=NUM_CTX)
run.note("sondas del servidor", {k: info.get(k) for k in ("server", "digest", "placement")})

R = rt.load(smoke=SMOKE_TEST, art=ART, docs_path=ART / "documents.jsonl")
CFG = RetrievalConfig(dense=True, prefix=True, lexical=True, rerank=True,
                      budget_tokens=PASS_TOKENS)


def count(text: str) -> int:
    return llm.count_tokens(text, MODEL) or 0


# La regla 5 es la que hace que este prompt no sea el de la Etapa 4: sin ella, un sistema que dice
# «respondé SOLO con los pasajes» obliga a abstenerse en toda sonda que se contesta con la memoria.
SYSTEM = pr.SYSTEM + (
    "\n5. Además de los pasajes, recibís la MEMORIA del usuario: lo que contó en sesiones "
    "anteriores, con sus fechas. Los datos del usuario (su versión de Python, su sistema, sus "
    "preferencias, qué decidió y cuándo) salen de la memoria; los datos de Python salen de los "
    "pasajes. Si algo no está ni en la memoria ni en los pasajes, poné can_answer=false.\n"
    "6. Si la memoria dice que el usuario usa una versión y los pasajes son de otra, decilo en "
    "vez de responder como si fueran la misma.")

# %% [markdown]
r"""
## 1 · El régimen: ¿entra la reproducción?

El enunciado pide **declarar el régimen** antes de comparar, porque «la memoria le gana al contexto
completo» no significa nada sin decir si el contexto completo entraba. Acá la respuesta es
incómoda y hay que decirla: **a nuestra escala la reproducción entra siempre**. Los escenarios
congelados tienen 2-3 sesiones de 1-2 turnos (56 tokens por sesión en promedio) y la transcripción
entera de la más larga son 206 tokens —286 contando lo que el espacio de memoria cobra encima:
32 tokens de encabezado con la fecha simulada y 16 por cada fecha de sesión—. Contra un `num_ctx`
de 8.192, o contra el espacio de memoria de 300-1.000 tokens que sugiere §7.3, no hay nada que no
entre.

Hay dos maneras de salir de eso sin inventar datos. La primera es la **proyección**: con los tokens
por sesión que de verdad tienen estos escenarios, cuántas sesiones hacen falta para cruzar cada
límite. La segunda es correr el experimento en **dos regímenes del mismo espacio de memoria**: uno
ancho (`SLOT_WIDE`), donde la reproducción entra entera y la predicción de Mem0 —el almacén gana en
costo y pierde en exactitud— es contrastable; y uno angosto (`SLOT_TIGHT`), elegido para que la
reproducción **no** entre en la mayoría de las sondas, que es donde empieza a importar qué
conservaste.

`SLOT_TIGHT = 120` tokens no es un despliegue realista y no se presenta como tal: es el valor
elegido con la cuenta medida a la vista, y tiene que cumplir **dos** cosas a la vez. Que la
transcripción entera no entre —a 120 se recorta en 15 de las 19 sondas— y que lo que entre sea
**una sesión**, no media: 120 − 32 de encabezado − 16 de fecha ≈ 72 tokens, y la sesión promedio
mide 56. A 80 tokens también recorta (en 19 de 19, con 48 tokens para contenido), pero ahí no entra
ni una sesión completa y el corte cae en mitad de un turno: eso mide dónde cayó la tijera, no qué
sesiones decidiste conservar, que es la política que el brazo quiere evaluar. La unidad que la
reproducción tira es la sesión; el espacio tiene que poder contener al menos una.

Lo que vale es la equivalencia que imprime la tabla de abajo: 120 tokens contra estos escenarios es
lo mismo que un espacio de 400 contra un usuario con unas pocas sesiones más, y la proyección dice
cuántas.

La tabla recorre los **estados** del experimento —un escenario y la sesión después de la cual se
pregunta— porque es ahí donde se decide si entra: una sonda después de `s1` reproduce una sesión y
una después de `s3`, tres.
"""

# %%
SESSIONS = {s.scenario_id: {x.session_id: x for x in s.sessions} for s in SCENARIOS}
DATE = {(s.scenario_id, x.session_id): datetime.fromisoformat(x.date)
        for s in SCENARIOS for x in s.sessions}


def sessions_upto(scenario, session_id: str) -> list:
    """Las sesiones hasta `session_id` inclusive, en orden cronológico (el schema ya lo garantiza)."""
    out = []
    for x in scenario.sessions:
        out.append(x)
        if x.session_id == session_id:
            break
    return out


def session_text(x) -> str:
    return "\n".join(f"Usuario: {t.user}" + (f"\nAsistente: {t.assistant}" if t.assistant else "")
                     for t in x.turns)


def probe_time(scenario, probe) -> datetime:
    """La sonda se pregunta en una sesión NUEVA: la fecha de la última sesión más `PROBE_GAP_H`.

    No es el reloj de pared. Con el reloj, «¿desde cuándo uso logging?» se contesta contra 2026-10
    y la recencia de todos los recuerdos es la misma (la trampa del término casi constante).
    """
    return DATE[(scenario.scenario_id, probe.after_session)] + timedelta(hours=PROBE_GAP_H)


def slot_header(now: datetime, kind: str) -> str:
    today = f"Hoy es {now:%Y-%m-%d} (fecha simulada)."
    if kind == "store":
        return f"Memoria del usuario, de sesiones anteriores. {today}"
    return f"Transcripción de las sesiones anteriores con este usuario. {today}"


def replay_tokens(scenario, session_id: str) -> int:
    """Lo que de verdad cuesta reproducir hasta `session_id`: la transcripción MÁS lo que el
    espacio de memoria le cobra encima (su encabezado con la fecha simulada y la fecha de cada
    sesión). El encabezado son ~32 tokens: contra un espacio de 80 es el 40 %, así que comparar
    solo la transcripción contra el límite declara «entra» cuando no entra."""
    sessions = sessions_upto(scenario, session_id)
    now = DATE[(scenario.scenario_id, session_id)] + timedelta(hours=PROBE_GAP_H)
    extra = sum(count(f"(sesión del {DATE[(scenario.scenario_id, x.session_id)]:%Y-%m-%d})")
                for x in sessions)
    return count(slot_header(now, "replay")) + extra + sum(count(session_text(x)) for x in sessions)


STATES = sorted({(s.scenario_id, p.after_session) for s, p in PROBES},
                key=lambda k: (k[0], k[1]))
regime = []
for sid, ses_id in STATES:
    sc = next(s for s in SCENARIOS if s.scenario_id == sid)
    txt = "\n".join(session_text(x) for x in sessions_upto(sc, ses_id))
    n = replay_tokens(sc, ses_id)
    n_probes = sum(1 for s, p in PROBES if s.scenario_id == sid and p.after_session == ses_id)
    regime.append((f"{sid} tras {ses_id}", {
        "sesiones": len(sessions_upto(sc, ses_id)), "sondas": n_probes,
        "tokens de la transcripción": count(txt),
        "tokens del espacio (con encabezados)": n,
        f"entra en {SLOT_WIDE}": "sí" if n <= SLOT_WIDE else "NO",
        f"entra en {SLOT_TIGHT}": "sí" if n <= SLOT_TIGHT else "NO",
    }))
print_table("el régimen, estado por estado (un estado = escenario + sesión tras la cual se pregunta)",
            list(regime[0][1]), regime, smoke=SMOKE_TEST, name_header="estado")

TIGHT_PROBES = [f"{s.scenario_id}#{p.probe_id}" for s, p in PROBES
                if replay_tokens(s, p.after_session) > SLOT_TIGHT]
wide_probes = [f"{s.scenario_id}#{p.probe_id}" for s, p in PROBES
               if replay_tokens(s, p.after_session) > SLOT_WIDE]
print(f"\nsondas donde la reproducción NO entra en {SLOT_TIGHT} tokens: {len(TIGHT_PROBES)} de "
      f"{len(PROBES)} · en {SLOT_WIDE} tokens: {len(wide_probes)}")
print(verdict(len(TIGHT_PROBES) >= 0.5 * len(PROBES),
              "el régimen angosto restringe en la mayoría de las sondas: ahí la comparación "
              "almacén/reproducción decide algo",
              "el régimen angosto no restringe en la mayoría de las sondas: hay que bajar "
              "SLOT_TIGHT antes de correr, o la tabla va a decir «da igual» cuando lo que pasa "
              "es que el espacio nunca se llenó"))

# %%
per_session = [count(session_text(x)) for s in SCENARIOS for x in s.sessions]
mean_s = float(np.mean(per_session))
overhead = pr.overhead("x" * 60, MODEL, system=SYSTEM)
window_room = NUM_CTX - NUM_PREDICT - overhead - PASS_TOKENS
print(f"una sesión de estos escenarios mide {mean_s:.0f} tokens de mediana "
      f"{int(np.median(per_session))} (mín {min(per_session)}, máx {max(per_session)})")
head_cost = count(slot_header(datetime(2026, 1, 1), "replay"))
print(f"la proyección, a {mean_s:.0f} tokens por sesión y {head_cost} de encabezado: la "
      f"reproducción cruza {SLOT_TIGHT} tokens a las {(SLOT_TIGHT - head_cost) / mean_s:.1f} "
      f"sesiones · {SLOT_WIDE} a las {(SLOT_WIDE - head_cost) / mean_s:.1f} · y el lugar que queda "
      f"en la ventana ({NUM_CTX} - {NUM_PREDICT} de reserva - {overhead} de prompt vacío - "
      f"{PASS_TOKENS} de pasajes = {window_room}) a las {window_room / mean_s:.0f} sesiones")
print(f"o sea: correr el régimen angosto con 3 sesiones mide lo mismo que un espacio de "
      f"{SLOT_WIDE} tokens con {(SLOT_WIDE - head_cost) / mean_s:.0f} sesiones, y la ventana "
      f"entera no se cruza "
      f"hasta ~{window_room / mean_s:.0f} sesiones de este tamaño -- que es la respuesta honesta a "
      f"«¿en qué punto deja de entrar?»: con este usuario, mucho después de las tres sesiones que "
      f"tiene el conjunto congelado")

# %% [markdown]
r"""
## 2 · El camino de escritura

Después de cada sesión, cuatro pasos: extraer hechos candidatos con un schema, recuperar los
recuerdos existentes más parecidos, decidir **ADD / UPDATE / NOOP / INVALIDATE**, y guardar con
`created_at`, `valid_from` y `valid_to`.

Decisiones que no son de estilo:

- **El `analysis` va primero en los dos schemas y no se guarda en ninguno.** Va primero porque un
  servidor que escribe las claves en orden alfabético haría decidir antes de razonar (`analysis` <
  `decision` < `target_id` < `text`), y no se guarda porque es exactamente el texto que convierte
  una hipótesis en un hecho guardado. El registro se arma campo por campo: `kind`, `subject`,
  `text`, `importance`. Nada más.
- **Los timestamps son los simulados.** `created_at` y `valid_from` son la fecha de la sesión, no
  `datetime.now()`: el reloj de pared pondría las 2-3 sesiones de un escenario a segundos de
  distancia y la recencia del camino de lectura mediría el orden en que corrió el script.
- **Un vocabulario corto de `subject`**, con `python_version` adentro. Sin eso, el mismo hecho se
  guarda como `python_version` en una sesión y `version_de_python` en la siguiente, el UPDATE no
  dispara nunca y el almacén acumula versiones contradictorias las dos «vigentes». El vocabulario
  se ajustó **sobre el escenario de dev** (`s-rutas`), que no entra en ninguna tabla, y
  `python_version` está ahí porque es el hecho que se corresponde con los metadatos del corpus:
  es el gancho de memoria que §5 usa como filtro.
- **UPDATE no edita: cierra y agrega.** Le pone `valid_to` al registro viejo y escribe uno nuevo con
  `supersedes`. Editar en el lugar dejaría «¿qué habíamos decidido en enero?» sin respuesta, que es
  justo una de las habilidades que se mide.
- **Una decisión inválida se cuenta, no se obedece.** Un `target_id` que no existe en la lista que
  se le mostró es una invención del modelo: un UPDATE así baja a ADD y un INVALIDATE así a NOOP, y
  las dos cosas van al log de operaciones con su motivo.
"""


# %%
class Fact(BaseModel):
    """Un hecho candidato. Lo que el extractor propone, antes de decidir qué hacer con él."""

    kind: Literal["semantic", "episodic", "procedural"]
    subject: str = Field(max_length=40)
    text: str = Field(max_length=160)
    importance: int = Field(default=5, ge=1, le=10)


class Extraction(BaseModel):
    analysis: str = Field(max_length=400)      # primero en el schema, y NO se guarda
    facts: list[Fact]


class MemoryRecord(BaseModel):
    memory_id: str
    kind: Literal["semantic", "episodic", "procedural"]
    subject: str                       # un único valor ACTUAL por subject
    text: str
    created_at: datetime               # cuándo lo escribió el sistema (tiempo SIMULADO)
    valid_from: datetime               # cuándo pasó a ser cierto
    valid_to: datetime | None = None   # se setea en INVALIDATE/UPDATE; el registro se queda
    source_session: str
    importance: int = Field(default=5, ge=1, le=10)
    supersedes: str | None = None


class WriteDecision(BaseModel):
    """El razonamiento declarado primero, obligatorio, y con un nombre que también quede primero:
    `analysis` < `decision` < `target_id` < `text`."""

    analysis: str = Field(max_length=400)
    decision: Literal["ADD", "UPDATE", "NOOP", "INVALIDATE"]
    target_id: str | None = None
    text: str | None = None


SUBJECTS = ("python_version", "sistema_operativo", "proyecto", "dependencias", "preferencia_api",
            "almacenamiento_fechas", "registro", "despliegue")

EXTRACT_SYSTEM = (
    "Extraé de la sesión los hechos del USUARIO que convenga recordar en sesiones futuras.\n"
    "Reglas:\n"
    "1. Solo lo que el usuario DIJO. No infieras, no supongas, no completes. Si el usuario no lo "
    "dijo con esas palabras, no es un hecho.\n"
    "2. Un hecho por `subject`. Usá estos nombres cuando apliquen: " + ", ".join(SUBJECTS) +
    ". Si ninguno encaja, inventá uno en snake_case.\n"
    "3. En `text`, una oración corta con los valores EXACTOS que dijo el usuario (versiones, "
    "módulos, sistemas, nombres). Sin condicionales y sin «probablemente».\n"
    "4. `kind`: semantic = un dato estable del usuario o su proyecto; procedural = una preferencia "
    "sobre cómo responderle; episodic = algo que pasó en esta sesión y no vale como dato estable.\n"
    "5. `importance` de 1 a 10: 10 cambia cómo responderle siempre, 1 es un detalle de paso.\n"
    f"6. Hasta {MAX_FACTS} hechos. Si no hay nada que valga la pena guardar, devolvé la lista vacía."
)

DECIDE_SYSTEM = (
    "Decidí qué hacer con un hecho nuevo frente a los recuerdos que ya están guardados.\n"
    "Reglas:\n"
    "1. ADD: no hay ningún recuerdo sobre lo mismo.\n"
    "2. UPDATE: hay un recuerdo sobre lo MISMO con otro valor, y el valor nuevo lo reemplaza. "
    "`target_id` obligatorio y `text` con el valor nuevo.\n"
    "3. NOOP: ya está guardado y dice lo mismo.\n"
    "4. INVALIDATE: el usuario dijo que algo guardado dejó de ser cierto y no hay valor nuevo que "
    "lo reemplace. `target_id` obligatorio.\n"
    "5. `target_id` tiene que ser uno de los identificadores de la lista. No inventes "
    "identificadores.\n"
    "6. Un cambio de versión, de sistema o de política es UPDATE, no ADD: si agregás, quedan dos "
    "valores vigentes para lo mismo."
)

# %%
STORE: dict[str, list[MemoryRecord]] = {s.scenario_id: [] for s in SCENARIOS}
OPS: list[dict] = []
WRITE_COST: dict[tuple[str, str], dict] = {}
_VEC: dict[str, np.ndarray] = {}


def vec(text: str) -> np.ndarray:
    if text not in _VEC:
        _VEC[text] = R.encode([text])[0]
    return _VEC[text]


def active(records: list[MemoryRecord], now: datetime) -> list[MemoryRecord]:
    """Los vigentes AL MOMENTO `now`: escritos antes, y no cerrados todavía."""
    return [r for r in records if r.created_at <= now and (r.valid_to is None or r.valid_to > now)]


def neighbours(records: list[MemoryRecord], fact: Fact, now: datetime,
               k: int = WRITE_NEIGHBOURS) -> list[MemoryRecord]:
    """Los recuerdos vigentes más parecidos al hecho candidato, con el mismo embebedor del corpus.

    El mismo `subject` entra siempre, aunque el embebedor no lo ponga arriba: es el caso que tiene
    que terminar en UPDATE, y dejarlo afuera de la lista fuerza un ADD por omisión.
    """
    cand = active(records, now)
    if not cand:
        return []
    q = vec(f"{fact.subject}: {fact.text}")
    sims = [float(vec(f"{r.subject}: {r.text}") @ q) for r in cand]
    order = sorted(range(len(cand)), key=lambda i: -sims[i])[:k]
    same = [i for i, r in enumerate(cand) if r.subject == fact.subject and i not in order]
    return [cand[i] for i in order + same]


def decide(fact: Fact, cands: list[MemoryRecord], *, session_date: datetime) -> tuple[WriteDecision | None, llm.ChatResult]:
    listing = "\n".join(
        f"- {r.memory_id} [{r.subject}] {r.text} (vigente desde {r.valid_from:%Y-%m-%d})"
        for r in cands) or "(el almacén está vacío)"
    body = (f"Fecha de la sesión: {session_date:%Y-%m-%d}\n\n"
            f"Hecho nuevo: [{fact.subject}] {fact.text}\n\n"
            f"Recuerdos guardados parecidos:\n{listing}")
    return llm.structured([{"role": "system", "content": DECIDE_SYSTEM},
                           {"role": "user", "content": body}],
                          WriteDecision, model=MODEL, num_ctx=2048, num_predict=300)


def write_session(scenario, x) -> None:
    """Un paso del camino de escritura: una sesión, el almacén del escenario adentro."""
    records = STORE[scenario.scenario_id]
    when = DATE[(scenario.scenario_id, x.session_id)]
    cost = {"calls": 0, "prompt": 0, "output": 0, "facts": 0, "errors": 0}

    out, res = llm.structured(
        [{"role": "system", "content": EXTRACT_SYSTEM},
         {"role": "user", "content": f"Fecha de la sesión: {when:%Y-%m-%d}\n\n{session_text(x)}"}],
        Extraction, model=MODEL, num_ctx=2048, num_predict=600)
    cost["calls"] += 1
    cost["prompt"] += res.prompt_tokens or 0
    cost["output"] += res.output_tokens or 0
    if out is None:
        # un extractor que falla no escribe nada y se cuenta: una sesión sin hechos es un dato
        cost["errors"] += 1
        OPS.append({"scenario": scenario.scenario_id, "session": x.session_id, "op": "EXTRACT",
                    "accepted": False, "why": res.error})
        WRITE_COST[(scenario.scenario_id, x.session_id)] = cost
        return

    for fact in out.facts[:MAX_FACTS]:
        cost["facts"] += 1
        cands = neighbours(records, fact, when)
        d, dres = decide(fact, cands, session_date=when)
        cost["calls"] += 1
        cost["prompt"] += dres.prompt_tokens or 0
        cost["output"] += dres.output_tokens or 0
        op = {"scenario": scenario.scenario_id, "session": x.session_id,
              "fact": fact.model_dump(), "candidates": [r.memory_id for r in cands],
              "accepted": True, "why": ""}
        if d is None:
            cost["errors"] += 1
            OPS.append({**op, "op": "DECIDE", "accepted": False, "why": dres.error})
            continue

        by_id = {r.memory_id: r for r in cands}
        decision, target = d.decision, by_id.get(d.target_id or "")
        if decision in ("UPDATE", "INVALIDATE") and target is None:
            # un id que no estaba en la lista es una invención: se cuenta y se degrada
            op["why"] = f"target_id inválido ({d.target_id!r}) -> " + \
                        ("ADD" if decision == "UPDATE" else "NOOP")
            op["accepted"] = False
            decision = "ADD" if decision == "UPDATE" else "NOOP"

        if decision == "NOOP":
            OPS.append({**op, "op": "NOOP", "target": None, "text": None})
            continue
        if decision == "INVALIDATE":
            target.valid_to = when
            OPS.append({**op, "op": "INVALIDATE", "target": target.memory_id, "text": None})
            continue
        if decision == "UPDATE":
            target.valid_to = when
        text = (d.text or fact.text).strip() or fact.text
        rec = MemoryRecord(
            memory_id=f"{scenario.scenario_id}#m{len(records) + 1:03d}",
            kind=fact.kind, subject=fact.subject, text=text,   # campo por campo: el `analysis` no
            created_at=when, valid_from=when, source_session=x.session_id,
            importance=fact.importance,
            supersedes=target.memory_id if decision == "UPDATE" else None)
        records.append(rec)
        OPS.append({**op, "op": decision, "target": rec.memory_id,
                    "text": text, "supersedes": rec.supersedes})
    WRITE_COST[(scenario.scenario_id, x.session_id)] = cost


t0 = time.perf_counter()
for sc in SCENARIOS:
    for x in sc.sessions:
        write_session(sc, x)
    print(f"{sc.scenario_id:12} {len(STORE[sc.scenario_id]):2} registros tras "
          f"{len(sc.sessions)} sesiones · "
          f"{sum(WRITE_COST[(sc.scenario_id, y.session_id)]['calls'] for y in sc.sessions)} "
          f"llamadas")
print(f"camino de escritura: {len(OPS)} operaciones en {time.perf_counter() - t0:.0f} s")

ops_table = []
for name in ("ADD", "UPDATE", "INVALIDATE", "NOOP", "EXTRACT", "DECIDE"):
    rows = [o for o in OPS if o["op"] == name]
    if not rows:
        continue
    ops_table.append((name, {"veces": len(rows),
                             "rechazadas o degradadas": sum(1 for o in rows if not o["accepted"]),
                             "motivo (una)": clip(next((o["why"] for o in rows if o["why"]), ""), 60)}))
print_table("el log de operaciones del camino de escritura", list(ops_table[0][1]), ops_table,
            smoke=SMOKE_TEST, name_header="operación")
store.write_jsonl(ART / "memory" / "records.jsonl",
                  [r.model_dump(mode="json") for s in SCENARIOS for r in STORE[s.scenario_id]])
store.write_jsonl(ART / "memory" / "ops.jsonl", OPS)
print(f"almacén y log en {ART / 'memory'}")

# %% [markdown]
r"""
## 3 · Qué quedó guardado, y qué de eso está mal

Tres chequeos sobre el almacén, antes de usarlo:

1. **Contaminación de razonamiento.** Ningún registro debería contener un `<think>` ni una
   hipótesis escrita como hecho. El primero es mecánico: `llm.chat` saca las filtraciones de
   `</think>` del `content` y las **cuenta**, y ese conteo es una línea del informe. El segundo se
   aproxima con marcas de cobertura («probablemente», «parece», «supongo», «asumo», «quizás»):
   cada una es un registro que guarda una conjetura y que después se lee como algo que el usuario
   dijo.
2. **«Nunca se dijo».** Para cada registro, qué fracción de sus palabras de contenido aparece en el
   texto de la sesión de la que salió, con la **misma** normalización del canal léxico
   (`lexical.tokenize`, stemmer español). Es un proxy, no un veredicto: una paráfrasis fiel baja el
   solapamiento y un invento puede reusar las palabras del usuario. Sirve para elegir qué leer.
3. **Dos valores vigentes para lo mismo.** Un `subject` con más de un registro activo al mismo
   tiempo es un UPDATE que no se hizo, y es el error que rompe el filtro de §5: no se puede filtrar
   por «la versión del usuario» si el almacén dice dos.

El conteo que pide el enunciado —cuántos registros están mal o registran algo que nunca se dijo—
**no se puede automatizar**: `stage6_records.txt` deja cada registro al lado del texto de su sesión
y de su solapamiento, y las cuentas se cargan a mano en `stage6_record_audit.json`. Si el archivo no
está, esto imprime PENDIENTE en vez de inventar un número.
"""

# %%
HEDGES = ("probablemente", "parece", "supongo", "asumo", "quizas", "quizá", "tal vez",
          "seguramente", "podria estar")
THINK = ("<think>", "</think>", "analysis")
ALL_RECORDS = [(s, r) for s in SCENARIOS for r in STORE[s.scenario_id]]


def overlap_with_source(scenario, rec: MemoryRecord) -> float:
    """La fracción de palabras de contenido del registro que están en su sesión de origen."""
    src = SESSIONS[scenario.scenario_id][rec.source_session]
    hay = set(lexical.tokenize(session_text(src)))
    toks = lexical.tokenize(rec.text)
    return sum(w in hay for w in toks) / max(len(toks), 1)


n_think = sum(any(m in r.text.lower() for m in THINK) for _, r in ALL_RECORDS)
n_hedge = sum(any(m in rt.strip_accents(r.text) for m in HEDGES) for _, r in ALL_RECORDS)
overlaps = [overlap_with_source(s, r) for s, r in ALL_RECORDS]
LOW_OVERLAP = [(s, r, o) for (s, r), o in zip(ALL_RECORDS, overlaps) if o < 0.5]

dupes = []
for s in SCENARIOS:
    for r in STORE[s.scenario_id]:
        same = [q for q in active(STORE[s.scenario_id], r.created_at)
                if q.subject == r.subject and q.memory_id != r.memory_id]
        if same:
            dupes.append((r.memory_id, [q.memory_id for q in same]))

kinds = collections.Counter(r.kind for _, r in ALL_RECORDS)
print_table("el almacén, chequeado", ["valor", "qué significa"], [
    ("registros", {"valor": len(ALL_RECORDS),
                   "qué significa": f"{dict(kinds)} · {sum(1 for _, r in ALL_RECORDS if r.valid_to)}"
                                    f" cerrados por UPDATE/INVALIDATE"}),
    ("con marcas de razonamiento", {"valor": n_think,
                                    "qué significa": "tiene que ser 0: el registro se arma campo "
                                                     "por campo y el `analysis` no se copia"}),
    ("con marcas de hipótesis", {"valor": n_hedge,
                                 "qué significa": "una conjetura guardada como hecho"}),
    ("filtraciones de </think> (llm.chat)", {"valor": llm.STATS["think_leaks"],
                                             "qué significa": "texto de razonamiento que el "
                                                              "servidor metió en el content"}),
    ("solapamiento con su sesión < 0,5", {"valor": len(LOW_OVERLAP),
                                          "qué significa": "candidatos a «nunca se dijo», para "
                                                           "leer a mano"}),
    ("subjects con dos valores vigentes", {"valor": len(dupes),
                                           "qué significa": "un UPDATE que no se hizo"}),
], smoke=SMOKE_TEST, name_header="chequeo")
print(verdict(n_think == 0,
              "ningún registro contiene razonamiento: guardar campo por campo alcanzó",
              f"{n_think} registros traen un bloque de razonamiento adentro: el almacén está "
              f"guardando hipótesis como hechos y hay que arreglar el camino de escritura antes "
              f"de leer cualquier tabla de abajo"))
if dupes:
    print(f"subjects duplicados: {dupes[:4]}")
for s, r, o in LOW_OVERLAP[:3]:
    print(f"\n{r.memory_id} [{r.subject}] solapamiento {o:.2f}\n  registro: {r.text}")
    print(wrap(f"sesión {r.source_session}: "
               f"{session_text(SESSIONS[s.scenario_id][r.source_session])}", width=90, indent="  "))

lines = []
for s, r in ALL_RECORDS:
    src = SESSIONS[s.scenario_id][r.source_session]
    lines += [f"=== {r.memory_id} · {r.kind} · [{r.subject}] · importancia {r.importance}",
              f"  texto    : {r.text}",
              f"  vigencia : {r.valid_from:%Y-%m-%d} -> "
              + (f"{r.valid_to:%Y-%m-%d} (cerrado)" if r.valid_to else "vigente")
              + (f" · reemplaza a {r.supersedes}" if r.supersedes else ""),
              f"  solapamiento con su sesión: {overlap_with_source(s, r):.2f}",
              wrap(f"sesión {r.source_session}: {session_text(src)}", width=90, indent="  "),
              "  ¿es cierto? ¿se dijo? -> ____", ""]
(ART / "stage6_records.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n{len(ALL_RECORDS)} registros en {ART / 'stage6_records.txt'} (hay que leerlos a mano)")
try:
    audit = store.read_json(ART / "stage6_record_audit.json")
    print_table("auditoría del almacén (lectura a mano)", ["cuenta", "de"],
                [(k, {"cuenta": v, "de": audit.get("records_read", len(ALL_RECORDS))})
                 for k, v in audit.items() if k != "records_read"],
                smoke=SMOKE_TEST, name_header="categoría")
except FileNotFoundError:
    print("falta stage6_record_audit.json: el conteo de registros MAL o NUNCA DICHOS está "
          f"PENDIENTE (a mano sobre {ART / 'stage6_records.txt'}; las claves esperadas son "
          "records_read, falsos, nunca_dichos, hipotesis)")

# %% [markdown]
r"""
## 4 · El camino de lectura

Tres términos al estilo de Generative Agents —**relevancia** (el mismo embebedor del corpus),
**recencia** (0,995 por hora sobre los timestamps **simulados**) e **importancia** (la del
registro)—, cada uno llevado a [0, 1] y sumado. La función es la del enunciado, con su guarda: se
normaliza **solo** si el término no es exactamente constante.

Y el enunciado pide imprimir el rango de cada término antes de normalizar, así que acá está, con el
resultado incómodo a la vista: a estos escenarios **la recencia no es casi constante, es casi
degenerada**. Entre dos sesiones pasan unos 70 días, y 0,995^(70 × 24) ≈ 2 × 10⁻⁴: después del
min-max, el recuerdo más nuevo se lleva 1 y todos los demás quedan en ~0. El término no aporta un
orden, aporta un *one-hot* del último registro. Es una consecuencia directa de combinar la constante
que sugiere §7.3 (pensada para horas) con escenarios espaciados en meses, y se reporta así: con
estos datos, «relevancia + recencia + importancia» es en la práctica «relevancia + importancia +
un bonus para el último registro».

Lo que va al espacio de memoria:

- **el brazo `almacén`** pone los `MEM_MIN`-`MEM_MAX` registros mejor puntuados que entren,
  **incluidos los cerrados**, marcados con su vigencia: «¿qué habíamos decidido en enero?» se
  contesta con un registro que ya no está vigente, y un almacén que solo muestra lo vigente no
  puede contestar una sonda temporal por construcción;
- **el brazo `reproducción`** pone la transcripción fechada, y cuando no entra descarta las
  sesiones **más viejas** (es la decisión natural: lo viejo es lo que un sistema real tira);
- las dos cosas llevan un encabezado con la fecha simulada de **hoy**, y el encabezado se paga del
  presupuesto. Sin él, «¿desde cuándo uso logging?» no tiene contra qué resolverse y el modelo
  contesta contra la fecha de su entrenamiento.
"""


# %%
def minmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-6:          # SOLO si es exactamente constante
        return [0.0] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]


def memory_score(recency: list[float], importance: list[float],
                 relevance: list[float]) -> list[float]:
    return [a + b + c for a, b, c in zip(minmax(recency), minmax(importance), minmax(relevance))]


RANGES: list[dict] = []        # el rango de cada término, por sonda: la trampa, medida
EMPTY_SLOT = "(nada entró en el espacio de memoria.)"


def render_record(r: MemoryRecord, now: datetime) -> str:
    closed = r.valid_to is not None and r.valid_to <= now
    when = (f"del {r.valid_from:%Y-%m-%d} al {r.valid_to:%Y-%m-%d}, ya no vale" if closed
            else f"desde el {r.valid_from:%Y-%m-%d}")
    return f"- [{r.subject}] {r.text} ({when})"


def read_store(scenario, probe, budget: int) -> dict:
    """Los recuerdos que entran en `budget`, y el rastro de cómo se eligieron."""
    now = probe_time(scenario, probe)
    cand = [r for r in STORE[scenario.scenario_id] if r.created_at <= now]
    head = slot_header(now, "store")
    if not cand:
        return {"text": None, "used": 0, "records": [], "sessions": set(), "n_candidates": 0}
    q = vec(probe.question)
    rel = [float(vec(f"{r.subject}: {r.text}") @ q) for r in cand]
    rec = [RECENCY_PER_HOUR ** ((now - r.valid_from).total_seconds() / 3600) for r in cand]
    imp = [float(r.importance) for r in cand]
    RANGES.append({"probe": f"{scenario.scenario_id}#{probe.probe_id}", "n": len(cand),
                   "relevancia": (min(rel), max(rel)), "recencia": (min(rec), max(rec)),
                   "importancia": (min(imp), max(imp))})
    scored = sorted(zip(memory_score(rec, imp, rel), cand), key=lambda t: -t[0])

    chosen, used = [], count(head)
    for _, r in scored[:MEM_MAX]:
        n = count(render_record(r, now))
        if used + n > budget:
            continue            # saltear, no cortar: un registro más corto todavía puede entrar
        chosen.append(r)
        used += n
    chosen.sort(key=lambda r: r.valid_from)      # cronológico: la trayectoria se lee en orden
    # Un encabezado solo («Memoria del usuario:») con nada abajo es peor que nada: el modelo lee
    # que hay memoria y la completa. Si no entró ningún recuerdo, el espacio lo DICE.
    text = "\n".join([head] + ([render_record(r, now) for r in chosen] if chosen else [EMPTY_SLOT]))
    return {"text": text, "used": used, "records": chosen,
            "sessions": {r.source_session for r in chosen}, "n_candidates": len(cand),
            "scores": [round(s, 3) for s, _ in scored[:MEM_MAX]]}


def read_replay(scenario, probe, budget: int) -> dict:
    """La transcripción de las sesiones pasadas que entra en `budget`, tirando las más viejas."""
    now = probe_time(scenario, probe)
    head = slot_header(now, "replay")
    lines = []      # (session, línea)
    for x in sessions_upto(scenario, probe.after_session):
        for t in x.turns:
            lines.append((x, f"Usuario: {t.user}"))
            if t.assistant:
                lines.append((x, f"Asistente: {t.assistant}"))
    kept, used, seen = [], count(head), set()
    for x, text in reversed(lines):
        extra = count(text) + (0 if x.session_id in seen
                               else count(f"(sesión del {DATE[(scenario.scenario_id, x.session_id)]:%Y-%m-%d})"))
        if used + extra > budget:
            break                 # acá tirar las sesiones MÁS VIEJAS es la estrategia
        kept.insert(0, (x, text))
        used += extra
        seen.add(x.session_id)
    out, last = [head], None
    for x, text in kept:
        if x.session_id != last:
            out.append(f"(sesión del {DATE[(scenario.scenario_id, x.session_id)]:%Y-%m-%d})")
            last = x.session_id
        out.append(text)
    if not kept:
        out.append(EMPTY_SLOT)
    return {"text": "\n".join(out), "used": used, "records": [],
            "sessions": {x.session_id for x, _ in kept}, "n_candidates": len(lines),
            "dropped": len(lines) - len(kept)}


def read_memory(kind: str, scenario, probe, budget: int) -> dict:
    if kind == "none":
        return {"text": None, "used": 0, "records": [], "sessions": set(), "n_candidates": 0}
    return read_store(scenario, probe, budget) if kind == "store" \
        else read_replay(scenario, probe, budget)


# %%
# El régimen angosto, verificado sobre el lector de verdad y no sobre la cuenta a mano: un espacio
# donde no entra NINGÚN turno no mide qué conservaste, mide «sin memoria» con otro nombre.
tight_slots = [read_replay(s, p, SLOT_TIGHT) for s, p in PROBES]
n_empty = sum(1 for d in tight_slots if not d["sessions"])
n_partial = sum(1 for d in tight_slots if d["sessions"] and d["dropped"])
print(f"la reproducción en {SLOT_TIGHT} tokens: {n_empty} sonda(s) sin nada, {n_partial} "
      f"recortada(s), {len(PROBES) - n_empty - n_partial} entera(s) · "
      f"{np.median([d['used'] for d in tight_slots]):.0f} tokens usados de mediana")
print(verdict(n_empty == 0 and n_partial >= 0.5 * len(PROBES),
              "el régimen angosto recorta sin vaciarse: el brazo mide qué sesiones conservó",
              f"el espacio angosto se vacía en {n_empty} sonda(s): subir SLOT_TIGHT, o el brazo "
              f"«reproducción angosta» es «sin memoria» con un encabezado arriba"))

# %%
# Los rangos, antes de normalizar: el chequeo que pide el enunciado.
for s, p in PROBES:
    read_store(s, p, SLOT_WIDE)
terms = ("relevancia", "recencia", "importancia")
range_table = []
for term in terms:
    widths = [hi - lo for r in RANGES for lo, hi in [r[term]]]
    range_table.append((term, {
        "mínimo observado": round(min(lo for r in RANGES for lo, _ in [r[term]]), 6),
        "máximo observado": round(max(hi for r in RANGES for _, hi in [r[term]]), 6),
        "ancho del rango (mediana)": round(float(np.median(widths)), 6),
        "sondas con rango exactamente 0": sum(w < 1e-6 for w in widths),
        "sondas con rango < 0,01": sum(w < 0.01 for w in widths),
    }))
print_table("el rango de cada término ANTES del min-max (la trampa del término casi constante)",
            list(range_table[0][1]), range_table, smoke=SMOKE_TEST, name_header="término")
second = [sorted((RECENCY_PER_HOUR ** ((probe_time(s, p) - r.valid_from).total_seconds() / 3600))
                 for r in STORE[s.scenario_id] if r.created_at <= probe_time(s, p))[-2:]
          for s, p in PROBES]
pairs = [x for x in second if len(x) == 2]
print(f"la recencia, en detalle: el registro más nuevo puntúa "
      f"{np.median([x[-1] for x in pairs]):.4f} de mediana y el segundo "
      f"{np.median([x[-2] for x in pairs]):.6f} -- después del min-max eso es un one-hot del "
      f"último registro, no un orden por antigüedad")
print(verdict(np.median([x[-2] for x in pairs]) > 0.01,
              "la recencia separa los registros por antigüedad a esta escala de tiempo",
              f"a {RECENCY_PER_HOUR}/hora y meses entre sesiones la recencia colapsa: se reporta "
              f"como «relevancia + importancia + un bonus para el último registro», y no como "
              f"tres términos"))
RANGE_ROWS = {n: v for n, v in range_table}

# %% [markdown]
r"""
## 5 · La memoria alimenta la recuperación (y a veces la arruina)

El hecho `python_version` se corresponde con un metadato del corpus, así que se puede convertir en
un **filtro**: `R.retrieve(..., allow=...)` restringe los candidatos a los `doc_id` de la versión
que el almacén dice que usa el usuario. Esta es la parte del enunciado donde hay que *buscar* el
daño, no solo la mejora, y en este corpus el daño está garantizado por dos razones distintas:

1. **El corpus tiene 3.12 y 3.13; el usuario de `s-etl` arranca en 3.11.** Filtrar por una versión
   que el corpus no tiene deja **cero** candidatos. `retrieval.retrieve` lo hace a propósito —«un
   filtro de memoria que deja cero DEBE dejar cero: es la señal»— en vez de caer de vuelta a la
   lista sin filtrar, que es lo que hace el `metadata_filter` de la Etapa 3 cuando la *pregunta*
   nombra una versión que no está. Son dos políticas opuestas y las dos están justificadas: una
   versión que el usuario *escribió* en una pregunta puede ser un error de tipeo; una que el
   sistema *guardó* como hecho del usuario es una afirmación del sistema sobre su propio usuario, y
   silenciarla esconde el problema.
2. **Y el pasaje que se pierde es justo el que hacía falta.** La sonda `p-batched-antes` pregunta si
   puede usar `itertools.batched` corriendo 3.11; la respuesta correcta es *no*, y el gold es la
   sección de `itertools` de **3.12**, donde dice desde qué versión existe. El filtro por 3.11 saca
   exactamente el pasaje que le permitiría al modelo decir «eso no está disponible en tu versión».
   Lo mismo del otro lado: después de `s3` el usuario está en 3.13 y el gold sigue siendo el de
   3.12.

Así que el filtro se mide **aparte**, sobre las sondas que tienen gold, y no se mete en los brazos
principales: una variable por vez. La tabla de abajo es de recuperación; el brazo
`+ filtro de versión` de §6 lo lleva hasta la respuesta.
"""

# %%
_VER_IN_TEXT = re.compile(r"\b3\.\d{1,2}\b")


def user_version(scenario, probe) -> str | None:
    """La versión que el almacén dice que usa el usuario, al momento de la sonda."""
    now = probe_time(scenario, probe)
    recs = [r for r in active(STORE[scenario.scenario_id], now) if r.subject == "python_version"]
    for r in sorted(recs, key=lambda r: r.valid_from, reverse=True):
        m = _VER_IN_TEXT.search(r.text)
        if m:
            return m.group(0)
    return None


def allow_for(version: str | None) -> set[str] | None:
    if version is None:
        return None            # sin hecho guardado no hay filtro: no es lo mismo que un filtro vacío
    return {d for d, md in R.metadata.items() if md["version"] == version}


GOLD_PROBES = [(s, p) for s, p in PROBES if p.gold]
FILTER_ROWS: dict[str, dict] = {}
filt_table, filt_lines = [], []
for s, p in GOLD_PROBES:
    iid = f"{s.scenario_id}#{p.probe_id}"
    v = user_version(s, p)
    allow = allow_for(v)
    tr_a, tr_b = {}, {}
    plain = R.retrieve(p.question, CFG, trace=tr_a)
    filtered = R.retrieve(p.question, CFG, trace=tr_b, allow=allow) if allow is not None else plain
    row = {
        "versión en memoria": v or "(no guardada)",
        "docs permitidos": len(allow) if allow is not None else len(R.metadata),
        "candidatos tras el filtro": f"{len(tr_b.get('ranked', []))} de {len(tr_a.get('ranked', []))}"
        if allow is not None else "sin filtro",
        "sección gold sin filtro": "sí" if R.section_hit(plain, p) else "no",
        "sección gold con filtro": "sí" if R.section_hit(filtered, p) else "no",
        "pasajes entregados": f"{len(filtered)} vs {len(plain)}",
    }
    FILTER_ROWS[iid] = {**row, "version": v, "n_allow": len(allow) if allow is not None else None,
                        "hit_plain": R.section_hit(plain, p),
                        "hit_filtered": R.section_hit(filtered, p),
                        "zero": allow is not None and not filtered,
                        "gold_versions": sorted({g.doc_id.split(":")[0] for g in p.gold})}
    filt_table.append((f"{p.probe_id} ({p.ability})", row))
    filt_lines += [f"=== {iid} · {p.ability} · versión en memoria {v}",
                   f"  pregunta: {p.question}",
                   f"  gold    : {[g.section_id for g in p.gold]}",
                   f"  sin filtro: {[u['id'] for u in plain]}",
                   f"  con filtro: {[u['id'] for u in filtered]}", ""]
print_table(f"la memoria como filtro del recuperador · {len(GOLD_PROBES)} sondas con gold",
            list(filt_table[0][1]), filt_table, smoke=SMOKE_TEST, name_header="sonda")
(ART / "stage6_filter.txt").write_text("\n".join(filt_lines), encoding="utf-8")

n_zero = sum(1 for r in FILTER_ROWS.values() if r["zero"])
n_lost = sum(1 for r in FILTER_ROWS.values() if r["hit_plain"] and not r["hit_filtered"])
n_won = sum(1 for r in FILTER_ROWS.values() if r["hit_filtered"] and not r["hit_plain"])
print(f"\nel filtro dejó CERO candidatos en {n_zero} de {len(GOLD_PROBES)} sondas, perdió el gold "
      f"en {n_lost} y lo ganó en {n_won}")
print(verdict(n_zero > 0,
              f"la guarda de `allow` se activó: {n_zero} sonda(s) quedó sin pasajes porque el "
              f"corpus no tiene la versión que el usuario dijo usar (3.11), y eso es la señal que "
              f"queríamos ver, no un error",
              "ninguna sonda quedó sin candidatos: con este almacén el filtro nunca pidió una "
              "versión ausente del corpus, así que el daño hay que buscarlo en la fila «perdió el "
              "gold»"))

# %% [markdown]
r"""
## 6 · El experimento: sin memoria / reproducción / almacén, en dos regímenes

Cinco brazos, y la variable que se mueve es **qué ocupa el espacio de memoria** y **cuánto mide ese
espacio**:

| brazo | espacio | qué pone |
|---|---|---|
| `sin memoria` | 0 | nada: el piso. Solo los pasajes del corpus |
| `reproducción · 400` | `SLOT_WIDE` | la transcripción entera (entra, §1) |
| `reproducción · 120` | `SLOT_TIGHT` | la transcripción recortada: pierde las sesiones viejas |
| `almacén · 400` | `SLOT_WIDE` | los registros mejor puntuados |
| `almacén · 120` | `SLOT_TIGHT` | los registros que entren |

Fijo en todos: los escenarios y su orden, `num_ctx`, la reserva de salida, el prompt de sistema, el
recuperador y su presupuesto de pasajes. La recuperación se hace **una vez por sonda** y se comparte:
los cinco brazos ven los mismos pasajes, así que lo único que los separa es la memoria. La consulta
del recuperador es la pregunta de la sonda tal cual —la Etapa 5 ya midió qué pasa cuando se la
reescribe, y meter eso acá movería dos cosas a la vez.

El sexto brazo, `almacén · 400 + filtro de versión`, corre **solo en las sondas con gold** y es el
único que cambia los pasajes: es §5 llevada hasta la respuesta.

La predicción de Mem0 es explícita: donde la reproducción entra, gana en exactitud y pierde en
costo. Las dos columnas para contrastarla son «correctas» y las de §7.
"""

# %%
PASSAGES: dict[str, list[dict]] = {}
t0 = time.perf_counter()
for s, p in PROBES:
    PASSAGES[f"{s.scenario_id}#{p.probe_id}"] = R.retrieve(p.question, CFG)
print(f"recuperación: {len(PASSAGES)} sondas en {time.perf_counter() - t0:.0f} s "
      f"(una vez por sonda, compartida por los brazos)")

WIDE_REPLAY = f"reproducción · {SLOT_WIDE}"
TIGHT_REPLAY = f"reproducción · {SLOT_TIGHT}"
WIDE_STORE = f"almacén · {SLOT_WIDE}"
TIGHT_STORE = f"almacén · {SLOT_TIGHT}"
FILTER_ARM = f"{WIDE_STORE} + filtro de versión"
ARMS = {"sin memoria": ("none", 0),
        WIDE_REPLAY: ("replay", SLOT_WIDE),
        TIGHT_REPLAY: ("replay", SLOT_TIGHT),
        WIDE_STORE: ("store", SLOT_WIDE),
        TIGHT_STORE: ("store", SLOT_TIGHT)}

log = store.ResultsLog(ART / "results" / "06_memory.jsonl",
                       run_spec={"stage": "06_memory", "eval": ev.key(), "model": MODEL,
                                 "index": R.index_key, "num_ctx": NUM_CTX,
                                 "num_predict": NUM_PREDICT, "pass_tokens": PASS_TOKENS,
                                 "slots": [SLOT_WIDE, SLOT_TIGHT],
                                 "recency": RECENCY_PER_HOUR, "probe_gap_h": PROBE_GAP_H,
                                 "system": store.sha256_text(SYSTEM)[:16],
                                 "records": store.sha256_json(
                                     [r.model_dump(mode="json") for _, r in ALL_RECORDS])})


def run_probe(arm: str, kind: str, budget: int, s, p, *, filtered: bool = False) -> None:
    iid = f"{s.scenario_id}#{p.probe_id}"
    if log.done(arm, iid):
        return
    slot = read_memory(kind, s, p, budget)
    if filtered:
        allow = allow_for(user_version(s, p))
        delivered = R.retrieve(p.question, CFG, allow=allow) if allow is not None \
            else PASSAGES[iid]
    else:
        delivered = PASSAGES[iid]
    msgs = pr.messages(p.question, delivered, memory=slot["text"], system=SYSTEM)
    t1 = time.perf_counter()
    ans, res = pr.answer(msgs, model=MODEL, num_ctx=NUM_CTX, num_predict=NUM_PREDICT)
    row = pr.score(ans, res, p, delivered, in_gold=lambda sid: R.in_gold_section(sid, p))
    needed = set(p.evidence_sessions)
    log.write({"arm": arm, "item_id": iid, **row,
               "scenario": s.scenario_id, "probe_id": p.probe_id, "ability": p.ability,
               "after_session": p.after_session, "kind": kind, "slot_budget": budget,
               "slot_tokens": slot["used"], "n_memories": len(slot["records"]),
               "slot_candidates": slot["n_candidates"],
               "slot_sessions": sorted(slot["sessions"]),
               "evidence_needed": sorted(needed),
               "evidence_covered": sorted(needed & slot["sessions"]),
               "tight": iid in TIGHT_PROBES, "filtered": filtered,
               "wall_s": round(time.perf_counter() - t1, 2),
               "answer_text": (ans.answer if ans else ""),
               "slot_text": slot["text"] or ""})


for arm, (kind, budget) in ARMS.items():
    t0, n_new = time.perf_counter(), 0
    for s, p in PROBES:
        n_new += not log.done(arm, f"{s.scenario_id}#{p.probe_id}")
        run_probe(arm, kind, budget, s, p)
    print(f"{arm:28} {n_new:3} sondas nuevas en {time.perf_counter() - t0:6.1f} s")

t0, n_new = time.perf_counter(), 0
for s, p in GOLD_PROBES:
    n_new += not log.done(FILTER_ARM, f"{s.scenario_id}#{p.probe_id}")
    run_probe(FILTER_ARM, "store", SLOT_WIDE, s, p, filtered=True)
print(f"{FILTER_ARM:28} {n_new:3} sondas nuevas en {time.perf_counter() - t0:6.1f} s")

ALL_ARMS = list(ARMS) + [FILTER_ARM]
ROWS = {a: {r["item_id"]: r for r in log.rows(a)} for a in ALL_ARMS}
IDS = [f"{s.scenario_id}#{p.probe_id}" for s, p in PROBES]
PROBE_BY_ID = {f"{s.scenario_id}#{p.probe_id}": (s, p) for s, p in PROBES}
ANSWERABLE = [i for i in IDS if PROBE_BY_ID[i][1].answerable]
UNANSWERABLE = [i for i in IDS if not PROBE_BY_ID[i][1].answerable]
GOLD_IDS = [f"{s.scenario_id}#{p.probe_id}" for s, p in GOLD_PROBES]


# %%
def counts(arm: str, ids: list[str], field: str = "correct") -> tuple[int, int]:
    """Conteos, no porcentajes: con 19 sondas un porcentaje esconde el denominador."""
    vals = [ROWS[arm][i][field] for i in ids if i in ROWS[arm]]
    vals = [v for v in vals if v is not None]
    return sum(bool(v) for v in vals), len(vals)


def frac(arm: str, ids: list[str], field: str = "correct") -> str:
    k, n = counts(arm, ids, field)
    return f"{k}/{n}" if n else "—"


def rate(arm: str, ids: list[str], field: str = "correct") -> float:
    k, n = counts(arm, ids, field)
    return k / n if n else float("nan")


def paired(a: str, b: str, ids: list[str], field: str = "correct") -> dict:
    ids = [i for i in ids if i in ROWS[a] and i in ROWS[b]]
    return noise.paired([ROWS[a][i][field] for i in ids], [ROWS[b][i][field] for i in ids])


main = []
for a in ALL_ARMS:
    ids = GOLD_IDS if a == FILTER_ARM else IDS
    rows = [ROWS[a][i] for i in ids if i in ROWS[a]]
    ans_ids = [i for i in ids if i in ANSWERABLE]
    un_ids = [i for i in ids if i in UNANSWERABLE]
    main.append((a, {
        "correctas": frac(a, ans_ids),
        "sondas con respuesta": len(ans_ids),
        "abstuvo bien": frac(a, un_ids, "abstained") if un_ids else "—",
        "clave en la respuesta": frac(a, ans_ids, "key_contained"),
        "citas ⊆ entregadas": frac(a, ids, "citations_subset"),
        "tokens del espacio": int(np.mean([r["slot_tokens"] for r in rows])) if rows else 0,
        "recuerdos": round(float(np.mean([r["n_memories"] for r in rows])), 1) if rows else 0,
        "tokens de prompt": int(np.mean([r["prompt_tokens"] or 0 for r in rows])) if rows else 0,
        "fallos": sum(1 for r in rows if r["category"] == "fallo"),
    }))
print_table("los brazos, sobre las sondas de test (el brazo del filtro corre solo en las de gold)",
            list(main[0][1]), main, smoke=SMOKE_TEST, name_header="brazo")

# %%
by_ability = []
for ab in sorted(ABILITIES, key=lambda k: -ABILITIES[k]):
    ids = [i for i in ANSWERABLE if ROWS["sin memoria"][i]["ability"] == ab] \
        or [i for i in UNANSWERABLE if ROWS["sin memoria"][i]["ability"] == ab]
    field = "correct"
    by_ability.append((f"{ab} (n={len(ids)})", {a: frac(a, ids, field) for a in ARMS}))
print_table("exactitud por habilidad, en conteos (3/4, no 75 %)", list(ARMS), by_ability,
            smoke=SMOKE_TEST, name_header="habilidad")

regime_table = [
    (f"todas ({len(ANSWERABLE)})", {a: frac(a, ANSWERABLE) for a in ARMS}),
    (f"la reproducción entra ({len([i for i in ANSWERABLE if i not in TIGHT_PROBES])})",
     {a: frac(a, [i for i in ANSWERABLE if i not in TIGHT_PROBES]) for a in ARMS}),
    (f"la reproducción NO entra ({len([i for i in ANSWERABLE if i in TIGHT_PROBES])})",
     {a: frac(a, [i for i in ANSWERABLE if i in TIGHT_PROBES]) for a in ARMS}),
]
print_table("correctas por régimen: el único corte que hace contrastable la predicción de Mem0",
            list(ARMS), regime_table, smoke=SMOKE_TEST, name_header="régimen")

for a in list(ARMS)[1:]:
    print(f"{a:28} vs sin memoria        -> {paired('sin memoria', a, ANSWERABLE)}")
print(f"{WIDE_STORE:28} vs {WIDE_REPLAY:21} -> {paired(WIDE_REPLAY, WIDE_STORE, ANSWERABLE)}")
print(f"{TIGHT_STORE:28} vs {TIGHT_REPLAY:21} -> {paired(TIGHT_REPLAY, TIGHT_STORE, ANSWERABLE)}")
print(f"{FILTER_ARM:28} vs {WIDE_STORE:21} -> {paired(WIDE_STORE, FILTER_ARM, GOLD_IDS)}")
print(f"\ncon {len(ANSWERABLE)} sondas con respuesta hacen falta {noise.detectable()} discordantes "
      f"todas para el mismo lado para bajar de p < 0,05. Y son {len(SCENARIOS)} escenarios: las "
      f"sondas de uno comparten persona, sesiones y almacén, así que no son independientes y no "
      f"se publica un IC por sonda -- la tabla que se lee es la de conteos por habilidad")

# %% [markdown]
r"""
## 7 · Recall de las sesiones con evidencia, y el costo

**Recall sin juez.** Cada sonda trae etiquetadas en `evidence_sessions` las sesiones donde se dijo
lo que hace falta, así que se puede medir sin leer ninguna respuesta: de las sesiones que la sonda
necesitaba, cuántas llegaron al espacio de memoria. Para la reproducción es cuáles sobrevivieron al
recorte; para el almacén, de qué sesiones venían los registros elegidos (`source_session`). Es la
métrica que separa «no lo sabía» de «lo tenía y no lo usó»: una sonda con recall 1 y respuesta mal
es un problema del generador, y una con recall 0 es un problema de la memoria.

**El costo.** Las dos líneas de base no pagan en el mismo lugar. La reproducción no escribe nada
—cero llamadas al LLM entre sesiones— y paga tokens de prompt en **cada** sonda. El almacén paga
llamadas de extracción y de decisión **una vez por sesión** y después pone menos tokens. Así que la
pregunta no es cuál es más barato, es **a partir de cuántas sondas** el almacén se paga, y eso es
una división: lo que costó escribirlo, dividido lo que ahorra por sonda.
"""

# %%
ev_table = []
for a in ALL_ARMS:
    ids = [i for i in (GOLD_IDS if a == FILTER_ARM else IDS)
           if i in ROWS[a] and ROWS[a][i]["evidence_needed"]]
    need = sum(len(ROWS[a][i]["evidence_needed"]) for i in ids)
    got = sum(len(ROWS[a][i]["evidence_covered"]) for i in ids)
    full = sum(1 for i in ids
               if set(ROWS[a][i]["evidence_covered"]) == set(ROWS[a][i]["evidence_needed"]))
    ev_table.append((a, {
        "sesiones con evidencia presentes": f"{got}/{need}",
        "sondas con TODA su evidencia": f"{full}/{len(ids)}",
        "correctas cuando la evidencia estaba":
            frac(a, [i for i in ids
                     if set(ROWS[a][i]["evidence_covered"]) == set(ROWS[a][i]["evidence_needed"])]),
        "correctas cuando faltaba":
            frac(a, [i for i in ids
                     if set(ROWS[a][i]["evidence_covered"]) != set(ROWS[a][i]["evidence_needed"])]),
    }))
print_table("recall de las sesiones con evidencia (sin juez: están etiquetadas)",
            list(ev_table[0][1]), ev_table, smoke=SMOKE_TEST, name_header="brazo")

# %%
write_calls = sum(c["calls"] for c in WRITE_COST.values())
write_tokens = sum(c["prompt"] + c["output"] for c in WRITE_COST.values())
cost_table = []
for sc in SCENARIOS:
    cs = [WRITE_COST[(sc.scenario_id, x.session_id)] for x in sc.sessions]
    cost_table.append((sc.scenario_id, {
        "sesiones": len(cs), "llamadas de escritura": sum(c["calls"] for c in cs),
        "hechos candidatos": sum(c["facts"] for c in cs),
        "tokens de escritura": sum(c["prompt"] + c["output"] for c in cs),
        "llamadas por sesión": round(sum(c["calls"] for c in cs) / len(cs), 1),
        "registros": len(STORE[sc.scenario_id]),
        "errores": sum(c["errors"] for c in cs),
    }))
print_table("el costo del camino de escritura (lo que la reproducción no paga)",
            list(cost_table[0][1]), cost_table, smoke=SMOKE_TEST, name_header="escenario")

slot_mean = {a: float(np.mean([ROWS[a][i]["slot_tokens"] for i in IDS if i in ROWS[a]]))
             for a in ARMS}
prompt_mean = {a: float(np.mean([ROWS[a][i]["prompt_tokens"] or 0 for i in IDS if i in ROWS[a]]))
               for a in ARMS}
saved = prompt_mean[WIDE_REPLAY] - prompt_mean[WIDE_STORE]
print(f"\ntokens de prompt por sonda: {WIDE_REPLAY} {prompt_mean[WIDE_REPLAY]:.0f} · "
      f"{WIDE_STORE} {prompt_mean[WIDE_STORE]:.0f} · sin memoria "
      f"{prompt_mean['sin memoria']:.0f}")
print(f"el almacén ahorra {saved:.0f} tokens de prompt por sonda y costó {write_tokens:,} tokens "
      f"y {write_calls} llamadas escribirlo")
if saved > 0:
    print(f"-> se paga a las {write_tokens / saved:.0f} sondas por usuario; este conjunto tiene "
          f"{len(PROBES)}")
else:
    print("-> no ahorra tokens de prompt a esta escala: los registros de un usuario de 2-3 "
          "sesiones no son más cortos que la transcripción, porque la transcripción ya es corta")
print(verdict(write_tokens / max(saved, 1e-9) > len(PROBES),
              f"a esta escala el almacén NO se amortiza: cuesta {write_tokens:,} tokens escribirlo "
              f"y el conjunto tiene {len(PROBES)} sondas. La ventaja de costo de Mem0 aparece "
              f"cuando la transcripción es grande, no cuando mide {max(per_session)} tokens por sesión",
              "el almacén se amortiza dentro de este conjunto de sondas"))

# %% [markdown]
r"""
### Una actualización de conocimiento, de punta a punta

Lo que pide el enunciado: qué se guardó, qué se invalidó, y qué obtuvo la sonda. Se elige
automáticamente el `subject` que tuvo un UPDATE —en los escenarios congelados es `python_version`
de `s-etl`, el par 3.11 → 3.13— y se imprime la línea de tiempo del almacén, las operaciones que la
escribieron y lo que contestó cada brazo en las sondas de ese escenario que dependen del cambio.
"""

# %%
UPDATED = [(o["scenario"], o["fact"]["subject"]) for o in OPS
           if o["op"] == "UPDATE" and o.get("fact")]
if UPDATED:
    sid, subj = UPDATED[0]
    print(f"escenario {sid} · subject «{subj}»\n")
    for r in sorted((r for r in STORE[sid] if r.subject == subj), key=lambda r: r.valid_from):
        print(f"  {r.memory_id}  {r.valid_from:%Y-%m-%d} -> "
              + (f"{r.valid_to:%Y-%m-%d} (cerrado)" if r.valid_to else "vigente      ")
              + f"  [{r.source_session}] {r.text}"
              + (f"  (reemplaza a {r.supersedes})" if r.supersedes else ""))
    print("\n  las operaciones que lo escribieron:")
    for o in OPS:
        if o["scenario"] == sid and o.get("fact", {}).get("subject") == subj:
            print(f"    {o['session']}  {o['op']:11} target={o.get('target')} "
                  f"{clip(o.get('text') or '', 60)}"
                  + (f"  [{o['why']}]" if o["why"] else ""))
    demo_ids = [i for i in ANSWERABLE if i.startswith(f"{sid}#")
                and ROWS["sin memoria"][i]["ability"] in ("temporal", "knowledge_update",
                                                          "multi_session")]
    print()
    print_table(f"lo que obtuvieron las sondas de {sid} que dependen del cambio",
                [ROWS["sin memoria"][i]["probe_id"] for i in demo_ids],
                [(a, {ROWS["sin memoria"][i]["probe_id"]:
                      ("correcta" if ROWS[a][i]["correct"] else
                       "FALLO" if ROWS[a][i]["category"] == "fallo" else
                       "abstuvo" if ROWS[a][i]["abstained"] else "mal")
                      for i in demo_ids}) for a in ARMS],
                smoke=SMOKE_TEST, name_header="brazo")
    for i in demo_ids[:2]:
        s, p = PROBE_BY_ID[i]
        print(f"\n{i} · {p.ability} · clave {p.answer_key} · «{p.question}»")
        for a in (WIDE_STORE, TIGHT_REPLAY):
            print(f"  {a}:")
            print(wrap(ROWS[a][i]["answer_text"] or "(sin respuesta)", width=86, indent="    ",
                       max_lines=4))
else:
    print("ningún UPDATE en este perfil: no hay actualización de punta a punta que mostrar "
          "(en el perfil completo, el par 3.11 -> 3.13 de s-etl)")

# %% [markdown]
r"""
## 8 · Las respuestas del informe

**¿En qué habilidades la memoria le ganó a no tener memoria, y en cuáles la reproducción le ganó al
almacén? ¿Con qué costo en tokens?** La tabla de conteos por habilidad de §6, leída junto con la de
régimen: el piso («sin memoria») no puede contestar `extraction`, `preference` ni `multi_session`
por construcción —esos datos no están en el corpus—, así que lo que hay que mirar ahí no es que
pierda, es *si abstuvo* en vez de inventar. La comparación que de verdad informa es
reproducción vs almacén **dentro de cada régimen**, con las columnas de costo de §7 al lado.

**Una actualización de conocimiento de punta a punta.** El final de §7: la línea de tiempo de
`python_version`, las operaciones que la escribieron (ADD en `s1`, UPDATE en `s3` con `valid_to` en
el registro viejo) y lo que contestó cada brazo en las sondas que dependen del cambio.

**¿Qué guardó el extractor que no era cierto, o que no se dijo?** §3: los chequeos mecánicos
(marcas de razonamiento, marcas de hipótesis, solapamiento con la sesión de origen, `subjects` con
dos valores vigentes) y el conteo a mano sobre `stage6_records.txt`, que se reporta como PENDIENTE
mientras no esté `stage6_record_audit.json`.

**¿Alguna memoria del usuario mejoró la recuperación como filtro, o la perjudicó?** §5. En este
corpus la perjudicó de las dos maneras posibles, y las dos están medidas: dejando **cero**
candidatos cuando el almacén dice una versión que el corpus no tiene (3.11), y sacando el pasaje
gold cuando el gold está en otra versión que la del usuario. El brazo `+ filtro de versión` de §6
muestra qué hizo el generador con eso: lo interesante no es que baje, es si **se abstuvo** o si
contestó igual sin pasajes.

**¿En qué punto el escenario deja de entrar en la ventana, y qué cambia entonces?** §1: no deja de
entrar. A 56 tokens por sesión, la transcripción cruza el espacio de 400 tokens a las ~7 sesiones y
el lugar que queda en la ventana a las ~90. Por eso el experimento se corre en dos regímenes del
espacio de memoria en vez de afirmar un cruce que estos escenarios no tienen: el régimen angosto es
el sustituto declarado del usuario de muchas sesiones que no está en el conjunto congelado.
"""

# %%
store.write_json(ART / "stage6_rows.json", {
    "model": MODEL, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT,
    "pass_tokens": PASS_TOKENS, "slots": {"wide": SLOT_WIDE, "tight": SLOT_TIGHT},
    "regime": {"per_state": {n: v for n, v in regime},
               "tight_probes": TIGHT_PROBES,
               "tight_slot": {"empty": n_empty, "trimmed": n_partial,
                              "whole": len(PROBES) - n_empty - n_partial},
               "tokens_per_session": {"mean": round(mean_s, 1),
                                      "median": int(np.median(per_session)),
                                      "max": max(per_session)},
               "slot_header_tokens": head_cost,
               "sessions_to_cross": {"tight": round((SLOT_TIGHT - head_cost) / mean_s, 1),
                                     "wide": round((SLOT_WIDE - head_cost) / mean_s, 1),
                                     "window": round(window_room / mean_s, 1)}},
    "write": {"ops": collections.Counter(o["op"] for o in OPS),
              "records": len(ALL_RECORDS),
              "kinds": dict(kinds),
              "closed": sum(1 for _, r in ALL_RECORDS if r.valid_to),
              "rejected": sum(1 for o in OPS if not o["accepted"]),
              "calls": write_calls, "tokens": write_tokens,
              "checks": {"think_markers": n_think, "hedges": n_hedge,
                         "think_leaks": llm.STATS["think_leaks"],
                         "low_overlap": len(LOW_OVERLAP), "duplicate_subjects": len(dupes)}},
    "read": {"term_ranges": RANGE_ROWS},
    "filter": {"rows": FILTER_ROWS, "zero": n_zero, "lost_gold": n_lost, "won_gold": n_won},
    "arms": {n: v for n, v in main},
    "by_ability": {n: v for n, v in by_ability},
    "by_regime": {n: v for n, v in regime_table},
    "evidence": {n: v for n, v in ev_table},
    "cost": {"per_scenario": {n: v for n, v in cost_table},
             "slot_tokens_mean": {a: round(v, 1) for a, v in slot_mean.items()},
             "prompt_tokens_mean": {a: round(v, 1) for a, v in prompt_mean.items()},
             "break_even_probes": round(write_tokens / saved, 1) if saved > 0 else None},
    "paired": {f"{a} vs sin memoria": paired("sin memoria", a, ANSWERABLE)
               for a in list(ARMS)[1:]}
    | {f"{WIDE_STORE} vs {WIDE_REPLAY}": paired(WIDE_REPLAY, WIDE_STORE, ANSWERABLE),
       f"{TIGHT_STORE} vs {TIGHT_REPLAY}": paired(TIGHT_REPLAY, TIGHT_STORE, ANSWERABLE),
       f"{FILTER_ARM} vs {WIDE_STORE}": paired(WIDE_STORE, FILTER_ARM, GOLD_IDS)},
})

run.note("régimen", f"la reproducción entra en {SLOT_WIDE} en todas las sondas; en "
                    f"{SLOT_TIGHT} no entra en {len(TIGHT_PROBES)} de {len(PROBES)}")
run.note("correctas", {a: frac(a, ANSWERABLE) for a in ARMS})
run.note("por habilidad", {n: v for n, v in by_ability})
run.note("escritura", {"registros": len(ALL_RECORDS), "llamadas": write_calls,
                       "tokens": write_tokens,
                       "rechazadas": sum(1 for o in OPS if not o["accepted"])})
run.note("filtro de versión", {"cero candidatos": n_zero, "perdió el gold": n_lost,
                               "ganó el gold": n_won})
run.note("pendiente a mano", ["contar registros falsos o nunca dichos (stage6_records.txt -> "
                              "stage6_record_audit.json)"])
run.finish()
