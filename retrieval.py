"""
El recuperador: cómo se corta el corpus, cómo se indexa, y cómo se entrega un contexto.

    import retrieval as rt
    R = rt.load(smoke=True)                       # índice + matrices + reranker, cacheados
    passages = R.retrieve("¿qué hace os.fspath?", rt.RetrievalConfig(dense=True, prefix=True))

Vive en su propio módulo porque lo usan **cuatro** etapas: la 3 lo mide, la 4 le pone un
generador encima, la 5 lo llama cuatro veces por turno con consultas distintas y la 6 le pasa
un filtro de memoria. Tres copias del mismo `retrieve` divergirían en el primer arreglo, y la
tabla de la Etapa 3 dejaría de describir lo que corren las Etapas 5 y 6.

Lo que este módulo NO hace: no imprime tablas ni saca conclusiones (eso es de cada etapa), y no
toca el conjunto de evaluación. Las métricas que están acá (`section_hit`, `answer_delivered`)
son las que las cuatro etapas tienen que calcular **igual** para que sus filas se comparen.
"""

from __future__ import annotations

import collections
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import environment
import lexical
import llm
import store

# --- las constantes de corte y de entrega -------------------------------------------------------
# Rangos del enunciado (§7.1/7.2): hijo 200-400 tokens del reranker con 10-20 % de solapamiento,
# padre 1.000-1.500 del generador, 30-100 candidatos por canal, RRF k=60, entrega 1.500-5.000.
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DENSE_MAX_SEQ = 512       # bge-m3 declara 8.192 y trunca en silencio en lo que fijemos
RERANK_MAX_LEN = 512      # la ventana del cross-encoder
CHILD_TOKENS = 350        # tokens DEL RERANKER, prefijo incluido
CHILD_OVERLAP = 0.15
PARENT_TOKENS = 1200      # tokens DEL GENERADOR
CANDIDATES = 60           # por canal
RRF_K = 60                # la constante del paper, no un hiperparámetro para 45 preguntas
BUDGET_TOKENS = 3000      # entregados al generador, contados en SU tokenizer
VERSIONS = ("3.13", "3.12")


def models(smoke: bool) -> dict:
    """Los tres modelos del perfil. El embebedor chico de smoke tampoco usa prefijos de
    instrucción, así que el camino de código es el mismo y lo único que cambia es el tamaño."""
    return {
        "generator": "qwen3:1.7b" if smoke else "qwen3:8b",
        "dense": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" if smoke
                  else "BAAI/bge-m3"),
        "rerank": RERANK_MODEL,
    }


# --- 1 · el texto propio de cada sección --------------------------------------------------------
def own_spans(doc: dict) -> dict[str, list[tuple[int, int]]]:
    """section_id -> los tramos de `doc['text']` que le pertenecen a ella y a nadie más.

    Las secciones son un árbol y sus spans se ANIDAN: el de la raíz cubre el documento entero.
    Indexar los spans tal cual mete el mismo párrafo en cinco chunks de cinco niveles. El resto
    puede no ser contiguo (prosa antes y después de las hijas), así que es una lista de tramos.
    """
    kids: dict[str | None, list[dict]] = collections.defaultdict(list)
    for s in doc["sections"]:
        kids[s["parent_id"]].append(s)
    out: dict[str, list[tuple[int, int]]] = {}
    for s in doc["sections"]:
        spans, cur = [], s["start"]
        for c in sorted(kids[s["section_id"]], key=lambda c: c["start"]):
            if c["start"] > cur:
                spans.append((cur, c["start"]))
            cur = max(cur, c["end"])
        if s["end"] > cur:
            spans.append((cur, s["end"]))
        out[s["section_id"]] = [sp for sp in spans if sp[1] > sp[0]]
    return out


def title_path(doc: dict, section_id: str) -> str:
    by_id = {s["section_id"]: s for s in doc["sections"]}
    names, cur = [], by_id[section_id]
    while cur is not None:
        if cur["title"]:
            names.append(cur["title"])
        cur = by_id.get(cur["parent_id"]) if cur["parent_id"] else None
    return " · ".join(reversed(names))


def prefix_of(doc: dict, section_id: str) -> str:
    """El prefijo contextual que pide el enunciado, de los metadatos: título del documento ·
    ruta de títulos · versión. Es parte del TEXTO del pasaje, no un prefijo de instrucción."""
    return f"{doc['title']} · {title_path(doc, section_id)} · Python {doc['metadata']['version']}"


def build_units(raw_docs: list[dict]) -> tuple[list[dict], dict]:
    units, n_noncontig = [], 0
    for doc in raw_docs:
        spans_by_sec = own_spans(doc)
        for s in doc["sections"]:
            spans = spans_by_sec[s["section_id"]]
            if not spans:
                continue
            n_noncontig += len(spans) > 1
            units.append({"doc_id": doc["doc_id"], "section_id": s["section_id"],
                          "spans": spans, "prefix": prefix_of(doc, s["section_id"]),
                          "text": "\n\n".join(doc["text"][a:b] for a, b in spans)})
    stats = {"units": len(units), "sections": sum(len(d["sections"]) for d in raw_docs),
             "noncontiguous": n_noncontig,
             "indexable_chars": sum(len(u["text"]) for u in units),
             "corpus_chars": sum(len(d["text"]) for d in raw_docs)}
    return units, stats


# --- 2 · hijos y padres -------------------------------------------------------------------------
_SENT = re.compile(r"(?<=[.;:])\s+")


def paragraphs(text: str) -> list[str]:
    return [p for p in (p.strip() for p in text.split("\n")) if p]


def split_long(para: str, size: int, budget: int) -> list[str]:
    """Un párrafo más largo que el presupuesto, partido por oraciones.

    Hace falta de verdad: la marca `¶` de Sphinx termina una línea por elemento anclable, pero un
    elemento puede ser una tabla entera o un bloque de ejemplo. El párrafo más largo del corpus
    pasa los 4.000 tokens del reranker, y un hijo así se trunca **en silencio** en 512: el
    reranker puntuaría el principio y nunca vería la respuesta. Si una oración sola tampoco
    entra, se corta por caracteres -- es feo, y es mejor que puntuar un pasaje mutilado.
    """
    if size <= budget:
        return [para]
    pieces, cur = [], ""
    approx = max(1, int(len(para) * budget / max(size, 1) * 0.9))   # caracteres que entran
    for sent in _SENT.split(para):
        while len(sent) > approx:
            pieces.append(sent[:approx])
            sent = sent[approx:]
        if len(cur) + len(sent) + 1 > approx and cur:
            pieces.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def pack(items: list[str], sizes: list[int], budget: int) -> list[tuple[int, int]]:
    """Rangos [i, j) de párrafos que entran en `budget`."""
    out, i = [], 0
    while i < len(items):
        j, total = i, 0
        while j < len(items) and (j == i or total + sizes[j] <= budget):
            total += sizes[j]
            j += 1
        out.append((i, j))
        i = j
    return out


def cut(units: list[dict], rr_len: Callable[[list[str]], list[int]],
        gen_len: Callable[[list[str]], list[int]]) -> tuple[list[dict], list[dict], dict]:
    """Los hijos y los padres, cada uno medido en el tokenizer que lo va a leer.

    El hijo se corta DENTRO del padre: así el mapa hijo -> padre es exacto por construcción, en
    vez de un argmin de distancias que acierta casi siempre.
    """
    t0 = time.perf_counter()
    for u in units:
        u["paras"] = paragraphs(u["text"]) or [u["text"]]

    def retokenize() -> None:
        flat = [p for u in units for p in u["paras"]]
        flat_rr, flat_gen = rr_len(flat), gen_len(flat)
        at = 0
        for u in units:
            n = len(u["paras"])
            u["rr"], u["gen"] = flat_rr[at:at + n], flat_gen[at:at + n]
            at += n

    retokenize()
    n_before = sum(len(u["paras"]) for u in units)
    # El prefijo se entrega CON el pasaje, así que se le descuenta del presupuesto de los dos:
    # un hijo de 350 tokens más 30 de prefijo son 380 que el reranker trunca en 512 cuando la
    # consulta se suma adelante, y un padre de 1.200 más su prefijo ya no es un padre de 1.200.
    prefixes = [u["prefix"] for u in units]
    for u, pr_rr, pr_gen in zip(units, rr_len(prefixes), gen_len(prefixes)):
        u["room"] = max(CHILD_TOKENS - pr_rr, 64)
        u["room_gen"] = max(PARENT_TOKENS - pr_gen, 128)
        u["pre_gen"] = pr_gen
    for u in units:
        if max(u["rr"], default=0) <= u["room"] and max(u["gen"], default=0) <= u["room_gen"]:
            continue
        u["paras"] = [piece for p, rr in zip(u["paras"], u["rr"])
                      for piece in split_long(p, rr, min(u["room"], u["room_gen"]))]
    extra = sum(len(u["paras"]) for u in units) - n_before
    if extra:
        retokenize()

    children: list[dict] = []
    parents: list[dict] = []
    for u in units:
        room = u["room"]
        for pi, (a, b) in enumerate(pack(u["paras"], u["gen"], u["room_gen"])):
            pid = f"{u['section_id']}::p{pi}"
            ptext = "\n".join(u["paras"][a:b])
            parents.append({"parent_id": pid, "doc_id": u["doc_id"], "section_id": u["section_id"],
                            "text": f"{u['prefix']}\n{ptext}",
                            "n_gen": sum(u["gen"][a:b]) + u["pre_gen"]})
            for ci, (x, y) in enumerate(pack(u["paras"][a:b], u["rr"][a:b], room)):
                back = x
                if ci and CHILD_OVERLAP:   # el hijo arranca unos párrafos antes, si entra
                    while back > 0 and sum(u["rr"][a + back - 1:a + y]) <= room * (1 + CHILD_OVERLAP):
                        back -= 1
                    back = max(back, 0)
                children.append({
                    "chunk_id": f"{pid}::c{ci}", "parent_id": pid, "doc_id": u["doc_id"],
                    "section_id": u["section_id"], "prefix": u["prefix"],
                    "text": "\n".join(u["paras"][a + back:a + y]),
                })

    # Los dos conteos son del texto ENTREGADO (prefijo incluido): el presupuesto de la Etapa 3 y
    # el del ensamblador de la Etapa 4 tienen que contar lo mismo que el modelo va a leer.
    child_rr = rr_len([c["prefix"] + "\n" + c["text"] for c in children])
    child_gen = gen_len([c["prefix"] + "\n" + c["text"] for c in children])
    for c, r, g in zip(children, child_rr, child_gen):
        c["n_rr"], c["n_gen"] = r, g
    stats = {"paragraphs": n_before, "split_extra": extra, "tokenizing_s": time.perf_counter() - t0,
             "child_rr": child_rr, "child_gen": child_gen,
             "over_window": sum(n > RERANK_MAX_LEN for n in child_rr)}
    return children, parents, stats


# --- 3 · una configuración, un flag por pieza ---------------------------------------------------
@dataclass(frozen=True)
class RetrievalConfig:
    dense: bool = True
    prefix: bool = False
    lexical: bool = False
    rerank: bool = False
    parent: bool = False
    metadata_filter: bool = False
    budget_tokens: int = BUDGET_TOKENS


def diff_rows(prev: RetrievalConfig, cur: RetrievalConfig) -> str:
    """Antes de llamar «acumulativa» a una fila: ¿cambió exactamente UNA variable?"""
    changed = [k for k, v in asdict(cur).items() if asdict(prev)[k] != v]
    if not changed:
        return "configuración idéntica -- una fila duplicada, no una nueva"
    return (f"una variable: {changed[0]}" if len(changed) == 1
            else f"{len(changed)} variables difieren ({', '.join(changed)}) -- NO es acumulativa")


def rrf(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


_VER = re.compile(r"\b(3\.\d{1,2})\b")


def asked_version(question: str) -> str | None:
    """La versión que la pregunta nombra, si el corpus la tiene. `3.9` no filtra nada: filtrar
    por una versión que no está en el corpus deja cero candidatos y mide la nada."""
    for v in _VER.findall(question):
        if v in VERSIONS:
            return v
    return None


# --- 4 · el recuperador -------------------------------------------------------------------------
@dataclass
class Retrieval:
    """Todo lo que hace falta para recuperar, y el rastro de dónde salió."""
    children: list[dict]
    parents: dict[str, dict]
    section_spans: dict[str, tuple[int, int]]
    metadata: dict[str, dict]
    matrix: np.ndarray | None = None
    prefixed_matrix: np.ndarray | None = None
    bm25: object | None = None
    reranker: object | None = None
    encode: Callable[[list[str]], np.ndarray] | None = None
    encoder: object | None = None      # el SentenceTransformer, por su tokenizer (Etapa 5A)
    rerank_tokenizer: object | None = None
    index_key: str = ""
    spec: dict | None = None
    stats: dict | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.by_chunk = {c["chunk_id"]: i for i, c in enumerate(self.children)}

    def child(self, chunk_id: str) -> dict:
        return self.children[self.by_chunk[chunk_id]]

    def child_text(self, chunk_id: str) -> str:
        c = self.child(chunk_id)
        return c["prefix"] + "\n" + c["text"]

    def retrieve(self, query: str, cfg: RetrievalConfig, *, trace: dict | None = None,
                 allow: set[str] | None = None) -> list[dict]:
        """Los pasajes entregados, por PRESUPUESTO DE TOKENS del generador -- no por k.

        Cada flag que pide algo que no está cargado levanta una excepción: una pieza que avisa y
        sigue produce una fila idéntica a la anterior, y uno lee «sin efecto» en una tabla que
        nunca corrió la pieza.

        `allow`, si viene, restringe los candidatos a un conjunto de `doc_id` (así es como la
        Etapa 6 usa la memoria: como filtro del recuperador, no como texto en el prompt).
        """
        if cfg.prefix and self.prefixed_matrix is None:
            raise RuntimeError("prefix=True pero no hay matriz de hijos con prefijo cargada")
        if cfg.lexical and self.bm25 is None:
            raise RuntimeError("lexical=True pero no hay índice BM25 cargado")
        if cfg.rerank and self.reranker is None:
            raise RuntimeError("rerank=True pero no hay reranker cargado")
        if cfg.parent and not self.parents:
            raise RuntimeError("parent=True pero no hay mapa hijo -> padre cargado")
        if cfg.metadata_filter and not self.metadata:
            raise RuntimeError("metadata_filter=True pero no hay metadatos de documento cargados")
        if cfg.dense and (self.matrix is None or self.encode is None):
            raise RuntimeError("dense=True pero no hay matriz de hijos o embebedor cargado")
        if not (cfg.dense or cfg.lexical):
            raise RuntimeError("ni dense ni lexical: no hay de dónde sacar candidatos")

        rankings: list[list[str]] = []
        names: list[str] = []
        top_score: dict[str, float] = {}
        if cfg.dense:
            mat = self.prefixed_matrix if cfg.prefix else self.matrix
            q = self.encode([query])[0]
            sims = mat @ q
            order = np.argsort(-sims)[:CANDIDATES]
            rankings.append([self.children[i]["chunk_id"] for i in order])
            names.append("dense")
            top_score["dense"] = float(sims[order[0]])
        if cfg.lexical:
            res, scr = self.bm25.retrieve([lexical.tokenize(query)], k=CANDIDATES,
                                          show_progress=False)
            rankings.append([self.children[int(i)]["chunk_id"] for i in res[0]])
            names.append("bm25")
            top_score["bm25"] = float(scr[0][0])

        fused = rrf(rankings) if len(rankings) > 1 else rankings[0]
        if trace is not None:
            trace["channels"] = dict(zip(names, rankings))

        if allow is not None:
            kept = [cid for cid in fused if self.child(cid)["doc_id"] in allow]
            if trace is not None:
                trace["allow"] = {"docs": len(allow), "kept": len(kept), "of": len(fused)}
            fused = kept        # un filtro de memoria que deja cero DEBE dejar cero: es la señal

        if cfg.metadata_filter:
            want = asked_version(query)
            if want:
                kept = [cid for cid in fused
                        if self.metadata[self.child(cid)["doc_id"]]["version"] == want]
                if trace is not None:
                    trace["filter"] = {"version": want, "kept": len(kept), "of": len(fused)}
                fused = kept or fused   # dejar cero no filtra: deja la fila sin candidatos
            elif trace is not None:
                trace["filter"] = {"version": None}

        if cfg.rerank and fused:
            cand = fused[:CANDIDATES]
            scores = self.reranker.predict([(query, self.child_text(c)) for c in cand],
                                           batch_size=32 if self.device == "cuda" else 8,
                                           show_progress_bar=False)
            fused = [c for _, c in sorted(zip(scores, cand), key=lambda t: -t[0])]
            top_score["rerank"] = float(max(scores)) if len(scores) else float("-inf")
        if trace is not None:
            trace["top_score"] = top_score
            trace["ranked"] = fused[:20]

        out, seen, spent = [], set(), 0
        for cid in fused:
            child = self.child(cid)
            unit = self.parents[child["parent_id"]] if cfg.parent else child
            key = child["parent_id"] if cfg.parent else cid
            if key in seen:
                continue
            cost = unit["n_gen"]
            if spent + cost > cfg.budget_tokens and out:
                continue                # no cortamos: una unidad más chica todavía puede entrar
            seen.add(key)
            spent += cost
            out.append({"id": key, "doc_id": unit["doc_id"], "section_id": unit["section_id"],
                        "text": unit["text"] if cfg.parent else self.child_text(cid),
                        "n_gen": cost, "rank": len(out) + 1})
            if spent >= cfg.budget_tokens:
                break
        return out

    # --- las métricas que las cuatro etapas calculan igual --------------------------------------
    def in_gold_section(self, section_id: str, item) -> bool:
        """Una sección acierta si ES una gold o está CONTENIDA en una gold. Por spans, no por
        igualdad de ids: con ids, un hijo que salió de una subsección del gold sería un fallo de
        recuperación cuando en realidad es una decisión de corte."""
        want = [g.section_id for g in item.gold if g.section_id]
        if section_id in want:
            return True
        if section_id not in self.section_spans:
            return False
        a, b = self.section_spans[section_id]
        doc = section_id.split("#")[0]
        return any(w.split("#")[0] == doc and self.section_spans[w][0] <= a
                   and b <= self.section_spans[w][1]
                   for w in want if w in self.section_spans)

    def section_hit(self, delivered: list[dict], item) -> bool:
        if not any(g.section_id for g in item.gold):
            return False
        return any(self.in_gold_section(u["section_id"], item) for u in delivered)

    def mrr_doc(self, ranked_chunks: list[str], item, k: int = 20) -> float:
        want = {g.doc_id for g in item.gold}
        for rank, cid in enumerate(ranked_chunks[:k], start=1):
            if self.child(cid)["doc_id"] in want:
                return 1.0 / rank
        return 0.0


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").lower()


def answer_delivered(delivered: list[dict], item) -> bool | None:
    """¿Está la clave de respuesta en el texto que se le va a dar al generador? La clave es
    CONJUNTIVA (todas sus cadenas), como dice el esquema de la Etapa 2."""
    if not item.answer_key:
        return None
    hay = strip_accents(" ".join(u["text"] for u in delivered))
    return all(strip_accents(k) in hay for k in item.answer_key)


# --- 5 · cargar (o construir) todo --------------------------------------------------------------
def model_revision(repo: str) -> str:
    try:
        from huggingface_hub import model_info
        return model_info(repo).sha or "unknown"
    except Exception as e:                       # sin red: la caché local ya fija el modelo
        print(f"   (revisión de {repo} no verificable: {type(e).__name__})")
        return "unknown"


def load(*, smoke: bool, art: Path | None = None, docs_path: Path | None = None,
         with_models: bool = True, verbose: bool = True) -> Retrieval:
    """El índice del perfil, construido una vez y reusado después.

    El manifiesto del índice guarda el hash del corpus, el chunker y el embebedor (id, revisión,
    `max_seq_length`), como pide el enunciado. El embebedor va como dato DESCRIPTIVO y no en la
    clave de frescura: los hijos, los padres y BM25 no dependen de él, y si estuviera en la clave
    correr lo mismo en otra máquina (fp16 vs fp32) re-cortaría el corpus entero para llegar a los
    mismos hijos. De la matriz, que sí depende del embebedor, se ocupa `store.cached_matrix`.
    """
    import bm25s

    art = art or environment.artifacts_dir(smoke)
    docs_path = docs_path or art / "documents.jsonl"
    M = models(smoke)
    raw_docs = store.read_jsonl(docs_path)
    docs = {d["doc_id"]: d for d in raw_docs}

    from transformers import AutoTokenizer
    rr_tok = AutoTokenizer.from_pretrained(M["rerank"])

    def rr_len(texts: list[str]) -> list[int]:
        return [len(ids) for ids in rr_tok(texts, add_special_tokens=False)["input_ids"]]

    def gen_len(texts: list[str]) -> list[int]:
        tok = llm._tokenizer(M["generator"])
        return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]

    units, ustats = build_units(raw_docs)
    children, parents, cstats = cut(units, rr_len, gen_len)
    if verbose:
        print(f"secciones con texto propio: {ustats['units']:,} de {ustats['sections']:,} "
              f"({ustats['noncontiguous']} con el texto propio partido en varios tramos)")
        print(f"{cstats['paragraphs']:,} párrafos tokenizados en los dos tokenizers en "
              f"{cstats['tokenizing_s']:.0f} s · {cstats['split_extra']:,} piezas más por partir "
              f"los largos")
        print(f"hijos {len(children):,} (máx {max(cstats['child_rr'])} tokens del reranker) · "
              f"padres {len(parents):,} (máx {max(p['n_gen'] for p in parents)} del generador)")

    chunker = {"child_tokens": CHILD_TOKENS, "child_overlap": CHILD_OVERLAP,
               "parent_tokens": PARENT_TOKENS, "rerank_tokenizer": M["rerank"],
               "generator_tokenizer": M["generator"], "unit": "own-text of a section"}
    embedder = {"model": M["dense"], "revision": model_revision(M["dense"]),
                "max_seq_length": DENSE_MAX_SEQ}
    spec = {"corpus_sha256": store.sha256_file(docs_path), "chunker": chunker,
            "lexical": {"tokenizer": "lexical.tokenize", "stemmer": "spanish",
                        "stopwords": sorted(lexical.STOP)}}
    index_dir = art / "index"

    def build_index(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        store.write_jsonl(path / "chunks.jsonl", children)
        store.write_jsonl(path / "parents.jsonl", parents)
        bm = bm25s.BM25()
        bm.index([lexical.tokenize(c["prefix"] + "\n" + c["text"]) for c in children],
                 show_progress=False)
        bm.save(str(path / "bm25"))

    store.build_or_load(index_dir, spec, build_index, embedder=embedder,
                        n_children=len(children), n_parents=len(parents))

    R = Retrieval(
        children=children, parents={p["parent_id"]: p for p in parents},
        section_spans={s["section_id"]: (s["start"], s["end"])
                       for d in raw_docs for s in d["sections"]},
        metadata={d: docs[d]["metadata"] for d in docs},
        bm25=bm25s.BM25.load(str(index_dir / "bm25"), load_corpus=False),
        rerank_tokenizer=rr_tok, index_key=store.sha256_json(spec), spec=spec,
        stats={**ustats, **cstats},
    )
    if not with_models:
        return R

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    device = environment.device()
    encoder = SentenceTransformer(M["dense"], device=device,
                                  model_kwargs={"torch_dtype": torch.float16}
                                  if device == "cuda" else {})
    encoder.max_seq_length = DENSE_MAX_SEQ

    def encode(texts: list[str]) -> np.ndarray:
        return encoder.encode(texts, batch_size=16 if device == "cuda" else 8,
                              normalize_embeddings=True, show_progress_bar=False,
                              convert_to_numpy=True).astype(np.float32)

    encoder_spec = {**embedder, "normalize": True, "query_prefix": None,
                    "dtype": "fp16" if device == "cuda" else "fp32"}
    ids = [c["chunk_id"] for c in children]
    R.matrix = store.cached_matrix("children", encoder_spec, ids,
                                   [c["text"] for c in children], encode, art_dir=art)
    R.prefixed_matrix = store.cached_matrix("children-prefixed", encoder_spec, ids,
                                            [c["prefix"] + "\n" + c["text"] for c in children],
                                            encode, art_dir=art)
    R.encode = encode
    R.encoder = encoder        # la Etapa 5A cuenta sus consultas en el tokenizer del embebedor
    R.reranker = CrossEncoder(M["rerank"], max_length=RERANK_MAX_LEN, device=device)
    R.device = device
    if verbose:
        print(f"matrices {R.matrix.shape} y {R.prefixed_matrix.shape} en {device}")
    return R
