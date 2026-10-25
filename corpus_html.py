"""Lectura de las páginas del HTML compilado de python-docs-es.

Por qué es un módulo y no una celda. La Etapa 0 (`00b_corpus_audit_ours`) mide el corpus y la
Etapa 1 (`01_documents`) lo convierte en `documents.jsonl`. Las dos tienen que leer las mismas
páginas de la misma forma: si la auditoría mide una extracción y el índice se construye sobre
otra, los números de la Etapa 0 no describen el corpus que terminás indexando, y la tabla de
"qué cambió la limpieza" compara dos cosas distintas.

Qué extrae de cada página, y por qué cada pieza le importa a una etapa posterior:

- `text` -- el cuerpo (`div.body`), concatenado en orden de documento. Es el texto sobre el que
  se cortan los chunks.
- `sections` -- un registro por `<section>`, con `start`/`end` como offsets dentro de `text`,
  `level` por anidamiento y `parent_id`. Son los `schema.Section` de la Etapa 1, y el grano de
  las etiquetas gold de sección de la Etapa 2.
- `anchors` -- un registro por `<dt id=...>`: el id estable de cada objeto de la API
  (`pathlib.Path.resolve`) con su firma y la sección que lo contiene. Es de donde salen las
  preguntas de identificador de la Etapa 2 y las citas verificables de la Etapa 4.
- `notes` -- las notas de versión (`versionadded` / `versionchanged` / `deprecated`), con el
  número de versión parseado. Alimentan los metadatos de versión y las preguntas temporales.
- `paragraphs` -- por párrafo, la marca de traducción de Sphinx (`translated` / `untranslated` /
  ninguna) y un veredicto tosco de idioma. Con esto se mide cuánto de la página está en inglés,
  que es lo que decide la regla del subconjunto.

Sobre la fijación de la fuente: el zip se recompila en el mismo lugar, así que su sha256 no es
un pin que otro pueda reproducir, es la huella de *nuestra* descarga. `fetch_archive` lo
devuelve para que quede en el manifiesto y en `corpus_card.md`.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import environment
import store

ARCHIVE_URL = "https://docs.python.org/es/{v}/archives/python-{v}-docs-html.zip"
PAGE_URL = "https://docs.python.org/es/{v}/{rel}"

# El subconjunto candidato: referencia + material de aprendizaje. Deja afuera `whatsnew/`
# (3,7 M caracteres, 66 % en inglés: son notas de release, no documentación de referencia),
# `c-api/`, `extending/`, `using/` y los directorios de páginas que Python ya no publica
# (`distutils/`, `install/`). La regla final, con los números que la justifican, está en
# `corpus_card.md`.
SUBSET = ("library", "builtins", "tutorial", "howto", "reference", "faq")

# --- política, no parseo ------------------------------------------------------------------------
# Esto no extrae nada: decide qué página entra al corpus. Vive acá, al lado del parser, por el
# mismo motivo que el parser es un módulo: la Etapa 0 mide el costo de la regla y la Etapa 1 la
# aplica. Si los umbrales estuvieran escritos en las dos notebooks, la tabla de la Etapa 0
# describiría un corpus distinto del que indexamos. El porqué de cada número está en
# `corpus_card.md` y en `decision_log.md` (entrada 3).
ENGLISH_CAP = 0.60       # arriba de esto la página es documentación en inglés, no en español
MIN_TOKENS = 100         # abajo de esto no hay nada que citar: es una página índice
STUB_MAX_TOKENS = 500    # un aviso de eliminación es corto; una referencia enorme no lo es
DEPRECATION_KINDS = frozenset({"deprecated", "deprecated-removed"})


def page_class(tokens: int, note_kinds) -> str:
    """`"aviso de eliminación"` | `"puntero"` | `"documentación"`. **El orden importa.**

    La prueba de eliminación va primera y acotada por largo. Si fuera al revés, los 24 módulos
    removidos (`aifc`, `asynchat`, ...) se descartarían por cortos y en inglés -- y son el mejor
    material de abstención que tiene el corpus, porque su contenido *es* la respuesta a "¿puedo
    usar X en 3.13?". Acotarla a `STUB_MAX_TOKENS` evita el error opuesto: sin el tope,
    `builtins/functions.html` (17.972 tokens, 98 % en inglés) se rescata por tener *alguna* nota
    de deprecación entre sus funciones.

    `"documentación"` todavía no dice si la página entra: eso lo decide `ENGLISH_CAP` sobre
    `Page.english_ratio`, que el llamador ya tiene.
    """
    if DEPRECATION_KINDS & set(note_kinds) and tokens < STUB_MAX_TOKENS:
        return "aviso de eliminación"
    if tokens < MIN_TOKENS:
        return "puntero"
    return "documentación"

# El termómetro de idioma de la auditoría resuelta (`00_corpus_audit`), palabra por palabra.
# No es un detector de idioma: es un conteo de palabras funcionales. Se reusa tal cual para que
# nuestros porcentajes sean comparables con los suyos.
ES_WORDS = set("de la que el en y los se del las un por con no una su para es al lo como más pero sus le "
               "si sin sobre este ya entre cuando todo esta ser son también puede todos así nos ni tiene "
               "donde mismo ese ahora cada otro otros esa eso hace otra ella".split())
EN_WORDS = set("the of and to in is that for it as with was on be by this are or an at from which not "
               "have has can if but when you will its their they should use returns return used also "
               "may all one these more than only such other into each".split())

_HEADING = re.compile(r"^h[1-6]$")
_NOTE_LABEL = re.compile(
    r"(Added in version|Changed in version|Deprecated since version|Deprecated-removed"
    r"|Soft deprecated since version|Nuevo en la versión|Distinto en la versión"
    r"|Obsoleto desde la versión)", re.I)
_VERSION = re.compile(r"\b(\d+\.\d+(?:\.\d+)?)\b")


def paragraph_language(text: str) -> str | None:
    """"es" / "en" por mayoría de palabras funcionales; None si hay menos de 4 para decidir."""
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    es, en = sum(w in ES_WORDS for w in words), sum(w in EN_WORDS for w in words)
    return None if es + en < 4 else ("en" if en > es else "es")


@dataclass
class Page:
    version: str
    rel: str                      # "library/pathlib.html"
    top: str                      # "library"
    page_id: str                  # "py3.13:library/pathlib" -- estable, legible, abre como URL
    url: str
    title: str
    text: str
    sections: list[dict] = field(default_factory=list)
    anchors: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    paragraphs: list[dict] = field(default_factory=list)

    @property
    def english_ratio(self) -> float:
        """Proporción de párrafos decidibles que el termómetro llama inglés."""
        decided = [p for p in self.paragraphs if p["lang"] is not None]
        return sum(p["lang"] == "en" for p in decided) / len(decided) if decided else 0.0

    @property
    def unmarked_ratio(self) -> float:
        """Párrafos que Sphinx no marcó ni traducidos ni sin traducir: nunca pasaron por la
        traducción. Un 100 % acá es una página que jamás se tradujo."""
        return sum(p["mark"] == "unmarked" for p in self.paragraphs) / len(self.paragraphs) \
            if self.paragraphs else 0.0


def fetch_archive(version: str, *, sha256: str | None = None,
                  on_mismatch: str = "warn") -> tuple[Path, str]:
    """Descarga (o reusa) el zip compilado. Devuelve (ruta, sha256 de lo que hay en disco)."""
    dest = environment.corpora_dir() / f"python-{version}-docs-html-es.zip"
    return store.fetch_url(ARCHIVE_URL.format(v=version), dest,
                           sha256=sha256, on_mismatch=on_mismatch)


class _Buffer:
    """Acumula texto y reporta offsets, para que `start`/`end` de cada sección sean reales."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0

    def add(self, chunk: str) -> None:
        if not chunk:
            return
        if self.parts:
            self.parts.append("\n\n")
            self.length += 2
        self.parts.append(chunk)
        self.length += len(chunk)

    def value(self) -> str:
        return "".join(self.parts)


def _own_text(node) -> str:
    """Texto propio de una `<section>`: todo menos sus secciones anidadas."""
    parts = []
    for child in node.children:
        if getattr(child, "name", None) == "section":
            continue
        text = child.get_text(" ", strip=True) if hasattr(child, "get_text") else str(child).strip()
        if text:
            parts.append(text)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _heading(node) -> str:
    h = node.find(_HEADING, recursive=False)
    # Sphinx cuelga un ancla "¶" de cada encabezado; sale con el texto y no es parte del título.
    return h.get_text(" ", strip=True).rstrip("¶").strip() if h else ""


def _walk_sections(node, buf: _Buffer, sections: list[dict], parent_id: str | None,
                   level: int) -> None:
    """Pre-orden: el texto propio de la sección, después sus hijas. `end` de una sección
    incluye a sus descendientes, así una sección padre abarca el span de sus hijas."""
    section_id = node.get("id") or f"_anon{len(sections)}"
    start = buf.length + (2 if buf.parts else 0)
    record = {"section_id": section_id, "title": _heading(node), "level": level,
              "start": start, "end": start, "parent_id": parent_id}
    sections.append(record)
    buf.add(_own_text(node))
    for child in node.find_all("section", recursive=False):
        _walk_sections(child, buf, sections, section_id, level + 1)
    record["end"] = buf.length


def parse_page(html: bytes, version: str, rel: str) -> Page | None:
    """None si la página es un stub de redirección o no tiene cuerpo."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    if soup.find("meta", attrs={"http-equiv": "refresh"}) is not None:
        return None
    body = soup.select_one("div.body")
    if body is None:
        return None

    buf = _Buffer()
    sections: list[dict] = []
    top_sections = body.find_all("section", recursive=False)
    if top_sections:
        for sec in top_sections:
            _walk_sections(sec, buf, sections, None, 1)
    else:
        # Páginas sin `<section>` (algunas de `faq/` y `builtins/`): el cuerpo entero es una
        # sección sin id. La Etapa 1 paga esto con chunks que no pueden citar una sección.
        buf.add(re.sub(r"\s+", " ", body.get_text(" ", strip=True)))
        sections.append({"section_id": "_body", "title": "", "level": 1,
                         "start": 0, "end": buf.length, "parent_id": None})
    text = buf.value()

    def enclosing(node) -> str:
        for parent in node.find_parents("section"):
            if parent.get("id"):
                return parent["id"]
        return ""

    anchors = []
    for dt in body.select("dt[id]"):
        signature = re.sub(r"\s+", " ", dt.get_text(" ", strip=True)).rstrip("¶").strip()
        anchors.append({"anchor_id": dt["id"], "signature": signature,
                        "section_id": enclosing(dt)})

    notes = []
    for note in body.select(".versionadded, .versionchanged, .deprecated, .deprecated-removed"):
        head = re.sub(r"\s+", " ", note.get_text(" ", strip=True))
        label = _NOTE_LABEL.match(head)
        version_match = _VERSION.search(head)
        classes = note.get("class") or []
        notes.append({"kind": classes[0] if classes else "",
                      "label": label.group(1) if label else "",
                      "version": version_match.group(1) if version_match else None,
                      "text": head[:300], "section_id": enclosing(note)})

    paragraphs = []
    for p in body.find_all("p"):
        classes = p.get("class") or []
        mark = ("translated" if "translated" in classes else
                "untranslated" if "untranslated" in classes else "unmarked")
        p_text = p.get_text(" ", strip=True)
        paragraphs.append({"mark": mark, "lang": paragraph_language(p_text),
                           "characters": len(p_text)})

    h1 = body.find("h1")
    title = h1.get_text(" ", strip=True).rstrip("¶").strip() if h1 else ""
    stem = rel[:-len(".html")] if rel.endswith(".html") else rel
    return Page(version=version, rel=rel, top=rel.split("/")[0] if "/" in rel else "(root)",
                page_id=f"py{version}:{stem}", url=PAGE_URL.format(v=version, rel=rel),
                title=title, text=text, sections=sections, anchors=anchors,
                notes=notes, paragraphs=paragraphs)


def iter_pages(version: str, *, zip_path: Path, subset: tuple[str, ...] = SUBSET,
               only: set[str] | None = None):
    """Itera las páginas del subconjunto. Los stubs de redirección se cuentan afuera: esta
    función los saltea, y `redirect_stubs` los enumera."""
    prefix = f"python-{version}-docs-html/"
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".html"))
        if not names:
            raise RuntimeError(f"{zip_path.name}: ninguna página bajo '{prefix}' -- el layout del "
                               "archivo cambió y toda medición posterior sería cero")
        for name in names:
            rel = name[len(prefix):]
            top = rel.split("/")[0] if "/" in rel else "(root)"
            if top not in subset or (only is not None and rel not in only):
                continue
            page = parse_page(zf.read(name), version, rel)
            if page is not None:
                yield page


def redirect_stubs(version: str, *, zip_path: Path,
                   subset: tuple[str, ...] = SUBSET) -> list[str]:
    """Las páginas del subconjunto que solo redirigen a otra. Un loader ingenuo las indexa
    como documentos vacíos."""
    prefix = f"python-{version}-docs-html/"
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not (name.startswith(prefix) and name.endswith(".html")):
                continue
            rel = name[len(prefix):]
            top = rel.split("/")[0] if "/" in rel else "(root)"
            if top not in subset:
                continue
            if b'http-equiv="refresh"' in zf.read(name)[:4096]:
                out.append(rel)
    return out


def ids_of(version: str, *, zip_path: Path) -> dict[str, set[str]]:
    """Los tres conjuntos de ids que una etiqueta gold puede nombrar. Comparar dos descargas
    con esto es lo que responde "¿el id sobrevive a una nueva descarga?"."""
    pages, sections, anchors = set(), set(), set()
    for page in iter_pages(version, zip_path=zip_path):
        pages.add(page.page_id)
        sections.update(f"{page.page_id}#{s['section_id']}" for s in page.sections)
        anchors.update(a["anchor_id"] for a in page.anchors)
    return {"pages": pages, "sections": sections, "anchors": anchors}
