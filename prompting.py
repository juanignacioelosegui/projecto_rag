"""
El ensamblador de prompts y el puntaje sin juez.

    import prompting as pr
    delivered, skipped = pr.assemble(passages, budget, count=lambda t: llm.count_tokens(t, MODEL))
    msgs = pr.messages(question, delivered)
    ans, r = pr.answer(msgs, model=MODEL, num_ctx=8192, num_predict=700)
    pr.score(ans, r, item, delivered)

Vive en su propio módulo porque lo reusan las Etapas 4, 5 y 6 (el enunciado lo pide así). Las
tres tienen que puntuar **igual** para que sus filas se comparen, y las tres tienen que armar el
prompt con la misma reserva de salida: una fila que arma el prompt distinto no está midiendo su
estrategia de contexto, está midiendo su ensamblador.

Decisiones que no son de estilo:

- **El campo de razonamiento va primero y se llama `analysis`.** Según el servidor, las claves se
  escriben en el orden del schema o en orden alfabético; `analysis` < `answer` < `can_answer` <
  `citations` cumple los dos. `llm.probe_key_order` mide qué hace el servidor de turno.
- **La abstención es un booleano** (`can_answer`), nunca una búsqueda de texto: el modelo puede
  parafrasear «no lo sé», o escribirlo en el análisis mientras responde en el campo de respuesta.
- **Las citas son una afirmación del modelo**, no un dato: se verifican contra los ids que
  realmente se entregaron.
- **Un error no es una abstención.** `r.error` (prompt truncado, JSON que no parsea, rechazo del
  servidor) y `llm.ContextOverflow` son su propia categoría: ni abstención, ni respuesta mal.
- **Saltear, no cortar.** Un pasaje que no entra se saltea y se sigue: un `break` en el primero
  que no entra descarta también todos los más chicos que vienen detrás.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

from pydantic import BaseModel, Field

import llm

# --- el schema ----------------------------------------------------------------------------------
class GroundedAnswer(BaseModel):
    """El razonamiento ANTES de la respuesta, en cualquier servidor (ver el docstring)."""

    # Los cuatro OBLIGATORIOS: un campo opcional se va al final de la gramática y el modelo
    # puede omitirlo, y una lista de citas ausente no se distingue de «no usé ningún pasaje».
    analysis: str = Field(max_length=600)   # un tope: un análisis largo se come num_predict
    answer: str = Field(max_length=1200)
    can_answer: bool
    citations: list[str]


SYSTEM = (
    "Sos un asistente que responde SOLO con lo que dicen los pasajes que te entregan, en "
    "español.\n"
    "Reglas:\n"
    "1. Si los pasajes no alcanzan para responder, poné can_answer=false y decilo en answer. "
    "No completes con lo que sepas de Python por tu cuenta.\n"
    "2. Citá en citations los identificadores EXACTOS, entre corchetes, de los pasajes que "
    "usaste. No cites un pasaje que no esté en la lista.\n"
    "3. En analysis, escribí primero en qué pasaje está la respuesta y por qué; después "
    "respondé en answer, en una o dos oraciones.\n"
    "4. Si la pregunta es de sí o no, que answer empiece con «Sí» o «No»."
)

PASSAGE_HEADER = "Pasajes:"
QUESTION_HEADER = "Pregunta:"


def marker(passage: dict) -> str:
    """El identificador que el modelo tiene que citar: `doc_id#sección`, que es el `section_id`
    de la Etapa 1. Un id del corpus (y no una etiqueta P1/P2) hace detectable el caso que
    importa: citar algo que existe pero no se entregó."""
    return passage.get("section_id") or passage["doc_id"]


def render(passages: list[dict]) -> str:
    return "\n\n".join(f"[{marker(p)}]\n{p['text']}" for p in passages)


def messages(question: str, passages: list[dict], *, history: list[dict] | None = None,
             memory: str | None = None, system: str = SYSTEM) -> list[dict]:
    """Sistema + (memoria) + (historial) + pasajes delimitados + pregunta.

    El historial va como mensajes de verdad (`user`/`assistant`), no resumido dentro del mensaje
    del usuario: así la Etapa 5 puede medir qué parte de la ventana se come, y el modelo lo lee
    como lo que es.
    """
    out = [{"role": "system", "content": system if not memory else f"{system}\n\n{memory}"}]
    out += history or []
    body = (f"{PASSAGE_HEADER}\n{render(passages)}\n\n{QUESTION_HEADER} {question}"
            if passages else f"(No se entregaron pasajes.)\n\n{QUESTION_HEADER} {question}")
    out.append({"role": "user", "content": body})
    return out


# --- el ensamblador -----------------------------------------------------------------------------
def assemble(passages: list[dict], budget: int,
             count: Callable[[str], int]) -> tuple[list[dict], list[dict]]:
    """Los pasajes que entran en `budget` tokens, en orden, y los que se saltearon."""
    delivered, skipped, used = [], [], 0
    for p in passages:
        n = count(p["text"])
        if used + n > budget:
            skipped.append(p)
            continue
        delivered.append(p)
        used += n
    return delivered, skipped


def overhead(question: str, model: str, *, system: str = SYSTEM,
             history: list[dict] | None = None, memory: str | None = None) -> int:
    """Lo que ocupa el prompt SIN pasajes, con la plantilla de chat aplicada.

    `count_tokens` sobre el texto pelado no ve los marcadores de rol ni el prompt de generación,
    y es justo lo que hace que una cuenta a ojo entre y el prompt real no.
    """
    msgs = messages(question, [], history=history, memory=memory, system=system)
    return llm.count_prompt_tokens(msgs, model) or 0


def passage_budget(question: str, model: str, *, num_ctx: int, num_predict: int,
                   margin: int = 128, **kw) -> int:
    """Lo que queda para pasajes: la ventana, menos la reserva de salida, menos el prompt vacío,
    menos un margen (los marcadores de cada pasaje, que `count_tokens` no cuenta)."""
    return max(num_ctx - num_predict - overhead(question, model, **kw) - margin, 0)


def answer(msgs: list[dict], *, model: str, num_ctx: int, num_predict: int = 700,
           strict_ctx: bool = True, **kw) -> tuple[GroundedAnswer | None, llm.ChatResult]:
    """`llm.structured` con la guarda de contexto atrapada: un `ContextOverflow` vuelve como un
    `ChatResult` con `error`, para que una fila entera no se caiga por un prompt grande."""
    try:
        return llm.structured(msgs, GroundedAnswer, model=model, num_ctx=num_ctx,
                              num_predict=num_predict, strict_ctx=strict_ctx, **kw)
    except llm.ContextOverflow as e:
        r = llm.ChatResult(model=model, error=f"CONTEXT_OVERFLOW (pre-flight): {e}")
        r.expected_prompt_tokens = llm.count_prompt_tokens(msgs, model)
        return None, r


# --- el puntaje, sin juez -----------------------------------------------------------------------
_NUM = re.compile(r"(?<=\d)[.,](?=\d)")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Minúsculas, sin acentos, espacios colapsados y los separadores decimales unificados.

    `3,14` y `3.14` son la misma respuesta; `1.000` y `1000` también. Sin esto, la clave de
    respuesta falla por tipografía y el informe cuenta como error de generación lo que es un
    error de comparación de cadenas.
    """
    t = unicodedata.normalize("NFD", text)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn").lower()
    t = _NUM.sub(".", t)
    t = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", t)      # separador de miles
    return _WS.sub(" ", t).strip()


def key_contained(text: str, item) -> bool | None:
    """La clave es CONJUNTIVA: todas sus cadenas tienen que estar."""
    if not getattr(item, "answer_key", None):
        return None
    hay = normalize(text)
    return all(normalize(k) in hay for k in item.answer_key)


def verdict_of(text: str) -> str | None:
    """El sí/no de una respuesta, leído solo del arranque (que el sistema pide explícito)."""
    head = normalize(text)[:24]
    if head.startswith("si") and not head.startswith("sin "):
        return "yes"
    if head.startswith("no"):
        return "no"
    return None


def cited(ans: GroundedAnswer) -> list[str]:
    """Las citas, sin corchetes ni espacios: lo que el modelo dijo que usó."""
    return [c.strip().strip("[]").strip() for c in ans.citations if c and c.strip()]


def score(ans: GroundedAnswer | None, r: llm.ChatResult, item, delivered: list[dict], *,
          in_gold: Callable[[str], bool] | None = None) -> dict:
    """Una fila de puntaje. `in_gold(section_id)` decide si una cita cae en el gold (la contención
    por spans del recuperador); sin ella, se compara contra los ids gold tal cual."""
    ids = [marker(p) for p in delivered]
    row = {"error": r.error, "done_reason": r.done_reason,
           "prompt_tokens": r.prompt_tokens, "expected_prompt_tokens": r.expected_prompt_tokens,
           "output_tokens": r.output_tokens, "ms": r.ms, "n_delivered": len(delivered),
           "delivered_tokens": sum(p.get("n_gen", 0) for p in delivered)}
    if ans is None:                      # su propia categoría: ni abstención ni respuesta
        return {**row, "category": "fallo", "abstained": None, "correct": None,
                "citations_subset": None, "cited_gold": None, "verdict_ok": None,
                "inconsistent": None}

    want_gold = in_gold or (lambda sid: any(g.section_id == sid for g in item.gold))
    cites = cited(ans)
    answerable = getattr(item, "answerable", True)
    key = key_contained(ans.answer, item)
    expected = getattr(item, "expected_verdict", None)
    said = verdict_of(ans.answer)
    return {
        **row,
        "category": "abstención" if not ans.can_answer else "respuesta",
        "abstained": not ans.can_answer,
        "should_abstain": not answerable,
        # correcta = la clave está (si hay clave) y el veredicto coincide (si hay veredicto);
        # en un ítem sin respuesta en el corpus, lo correcto ES abstenerse.
        "correct": (not ans.can_answer if not answerable
                    else (bool(ans.can_answer) and key is not False
                          and (expected is None or said == expected))),
        "key_contained": key,
        "citations": cites,
        "n_citations": len(cites),
        "citations_subset": all(c in ids for c in cites) if cites else None,
        "cited_not_delivered": [c for c in cites if c not in ids],
        "cited_gold": any(want_gold(c) for c in cites) if cites else False,
        "verdict_ok": None if expected is None else said == expected,
        # el booleano dice una cosa y el texto otra: la inconsistencia que hay que contar
        "inconsistent": (ans.can_answer and normalize(ans.answer).startswith(
            ("no se", "no puedo", "no hay suficiente", "no esta en el contexto")))
        or (not ans.can_answer and len(normalize(ans.answer)) > 400),
    }
