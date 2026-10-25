# %% [markdown]
r"""
# Etapa 1 · De la fuente a los documentos

De acá cuelga todo lo que viene después: las etiquetas gold apuntan a un `doc_id`, los chunks se
cortan siguiendo las secciones, los filtros leen los metadatos, las citas nombran el id. Así que
la mayor parte de esta etapa no es código: es decidir **qué es un documento**.

**Nuestra definición.** Un documento es **una página compilada de una versión**:
`py3.13:library/pathlib`. No el módulo `pathlib` (eso uniría 3.12 y 3.13 en un registro con dos
respuestas contradictorias), ni una sección (eso tiraría el contexto del título de página, que es
lo que hace legible un pasaje). Una **sección** es un `<section>` de Sphinx, con el `id` que la
propia fuente le puso: `py3.13:library/pathlib#pure-paths`.

Lo que se rompe si una etapa posterior no está de acuerdo:

- si la Etapa 3 corta por largo e ignora `sections`, el gold de sección de la Etapa 2 deja de ser
  medible y solo queda hit@k a nivel de documento -- la métrica que la Etapa 0 ya midió como
  insensible, porque 3.12 tiene una copia del 86 % de los párrafos;
- si la Etapa 4 cita un offset en vez de un `section_id`, la cita deja de ser verificable en
  cuanto se recompila el zip.

**Entra**: los dos zips, fijados por el sha256 de *nuestra* descarga.
**Sale**: `artifacts/<profile>/documents.jsonl` + su manifiesto, y los números de limpieza que van
a `corpus_card.md` y al informe.
"""

# %%
import environment  # noqa: F401  (efectos secundarios: guarda de JAX, caché de HF, bundle TLS, UTF-8)

environment.summary()

import collections
import os
import re
import unicodedata

import corpus_html
import llm
import store
from console import StageRun, clip, print_table, verdict
from schema import DocMetadata, Document, Section, documents_hash, validate_documents

SMOKE_TEST = os.environ.get("SMOKE_TEST", "1").lower() not in ("0", "false", "no")
ART = environment.artifacts_dir(SMOKE_TEST)
PROFILE = "smoke" if SMOKE_TEST else "full"
GENERATOR = "qwen3:1.7b" if SMOKE_TEST else "qwen3:8b"
VERSIONS = ("3.13", "3.12")
CURRENT = "3.13"          # la versión vigente del corpus; la otra es el distractor de versión

# El sha256 de NUESTRA descarga del 2026-10-24, medido por la Etapa 0. No es un pin que otro
# pueda reproducir -- el zip se recompila en el mismo lugar -- es la huella de lo que auditamos.
# De acá en adelante va con on_mismatch="raise": si upstream recompiló, la corrida tiene que
# pararse, no seguir evaluando contra otro corpus. Cuando se caiga: bajá el zip nuevo, corré la
# Etapa 0 otra vez, y cambiá estos dos hashes con la fecha nueva en `corpus_card.md`.
PINNED = {
    "3.13": "59f4ffeee535aacf14786d7960b829aa047ea26d4aa76f9a841fce5d1fb3db67",
    "3.12": "ed1d2613c23620f83525a0453495fed896b0812ccc0a3e7393917dd5ba90b59d",
}
DOWNLOADED_ON = "2026-10-24"

SMOKE_PAGES = {"library/pathlib.html", "library/json.html", "library/itertools.html",
               "library/os.path.html", "library/aifc.html", "builtins/functions.html",
               "library/concurrent.html", "library/typing.html", "faq/general.html",
               "tutorial/introduction.html"}

run = StageRun.start("01_documents", ART, profile=PROFILE,
                     config={"versions": list(VERSIONS), "current": CURRENT,
                             "generator": GENERATOR, "downloaded_on": DOWNLOADED_ON})

# %% [markdown]
r"""
## 1 · Fijar la fuente

`on_mismatch="raise"`. La Etapa 0 corría con `"warn"` porque su trabajo era *descubrir* que el
zip se recompila; acá ya lo sabemos y lo que hace falta es que un corpus distinto no entre sin
avisar.
"""

# %%
zips = {}
for v in VERSIONS:
    path, digest = corpus_html.fetch_archive(v, sha256=PINNED[v], on_mismatch="raise")
    zips[v] = {"path": path, "sha256": digest, "bytes": path.stat().st_size}
    print(f"{v}: {path.name}  {digest[:16]}…  {zips[v]['bytes']:,} bytes")
print(f"\nfecha de descarga declarada: {DOWNLOADED_ON}  ·  "
      f"url: {corpus_html.ARCHIVE_URL.format(v='<v>')}")

# %% [markdown]
r"""
## 2 · La limpieza, una regla por defecto medido

Cada regla sale de un defecto que la Etapa 0 contó, no de una idea general de prolijidad. Los
contadores `CLEANING_HITS` / `CLEANING_CHARS` son el paso 8 del enunciado ("medí qué cambió la
limpieza") convertido en un print.

**Los offsets.** `start` / `end` apuntan al texto **ya limpio**, así que la limpieza no puede
correr después de calcularlos. La forma del texto nos salva: `corpus_html` arma el cuerpo como
el texto propio de cada sección, con los espacios ya colapsados, unidos por `"\n\n"`. O sea el
texto es una secuencia de bloques sin saltos de línea adentro, y los `start` / `end` de las
secciones caen siempre en un borde de bloque (verificado abajo con un assert, no asumido).
Entonces: partir por `"\n\n"`, limpiar bloque por bloque, y recalcular los offsets desde los
largos nuevos. El mapeo es exacto, no aproximado.

**La marca `¶` no se borra: se reemplaza por un salto de línea.** Es el defecto que un loader
ingenuo deja pasar sin ningún error, y borrarla a secas lo empeora. Sphinx cuelga un ancla `¶`
del final de cada encabezado y de cada firma, y en el texto plano esa marca es *lo único* que
separa el título del cuerpo: `"Excepciones ¶ exception pathlib. UnsupportedOperation ¶ Una
excepción que…"`. Borrarla da `"Excepciones exception pathlib.UnsupportedOperation Una
excepción que…"`, que pega el encabezado a la firma y la firma a la prosa. Es exactamente el
tipo de daño que no tira ninguna excepción: el documento se indexa, el pipeline corre, y los
pasajes recuperados tienen una frase inventada en el borde. Reemplazarla por `"\n"` conserva lo
que la marca significa (acá termina un elemento anclable).

**Las firmas partidas se arreglan con el id, no con una regex.** El texto trae
`"exception pathlib. UnsupportedOperation"` porque Sphinx pone el prefijo del módulo en un
`<span>` aparte. Una regex del estilo `r"(\w)\.\s+(\w)"` arreglaría eso y rompería cualquier
oración que termine en una abreviatura. Lo que usamos es el dato que la fuente ya nos dio: el
`anchor_id` (`pathlib.UnsupportedOperation`). Para cada ancla generamos las variantes con un
espacio después de cada punto y reemplazamos solo esas. Cero falsos positivos sobre prosa que
no contenga un identificador del propio módulo.

**Las etiquetas de nota de versión se traducen; el número no se toca.** 214 páginas traen
`"Added in version 3.13"` en medio del español, y 163 mezclan las dos formas en la misma página.
Un filtro o una pregunta temporal que busque `"Nuevo en la versión"` no encuentra la mitad del
corpus. El número de versión es el dato: se conserva literal, y hay un assert que lo comprueba.
"""

# %%
CLEANING_HITS: dict[str, int] = {}     # regla -> documentos que cambió
CLEANING_CHARS: dict[str, int] = {}    # regla -> caracteres que eliminó (negativo si agregó)


def apply(rule: str, fn, text: str) -> str:
    out = fn(text)
    if out != text:
        CLEANING_HITS[rule] = CLEANING_HITS.get(rule, 0) + 1
        CLEANING_CHARS[rule] = CLEANING_CHARS.get(rule, 0) + len(text) - len(out)
    return out


# Las etiquetas que Sphinx emite en inglés cuando la entrada no está traducida, y su forma en
# español tal como aparece en el resto del corpus. El número va aparte y queda intacto.
NOTE_LABELS = {
    "added in version": "Nuevo en la versión",
    "new in version": "Nuevo en la versión",
    "changed in version": "Distinto en la versión",
    "deprecated since version": "Obsoleto desde la versión",
    "soft deprecated since version": "Obsoleto desde la versión",
}
_EN_LABEL = re.compile("|".join(sorted((re.escape(k) for k in NOTE_LABELS), key=len, reverse=True)),
                       re.IGNORECASE)


def rule_nfc(text: str) -> str:
    """NFC. Sin esto, "información" con el acento descompuesto y con el acento compuesto son dos
    cadenas distintas para BM25 y para el puntaje por coincidencia exacta: ningún error, solo
    fallos silenciosos."""
    return unicodedata.normalize("NFC", text)


def rule_pilcrow(text: str) -> str:
    """La marca de ancla pasa a ser un salto de línea, que es lo que significa."""
    return re.sub(r"\s*¶\s*", "\n", text).strip()


def rule_note_label(text: str) -> str:
    return _EN_LABEL.sub(lambda m: NOTE_LABELS[m.group(0).lower()], text)


def rule_whitespace(text: str) -> str:
    """Colapsa espacios DENTRO de cada línea y conserva los saltos: los saltos son lo que
    `rule_pilcrow` acaba de usar para marcar los bordes de elemento."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def signature_fixer(anchors: list[dict]):
    """Un reemplazador por página, armado desde los ids de las anclas.

    Devuelve None si la página no tiene anclas con punto, así una página de tutorial no paga
    nada. La alternancia se compila una vez por página, no una vez por bloque.

    El primer intento generaba variantes del id **completo** (`pathlib.Path. open`), pero la firma
    que la página muestra es casi siempre la **corta** (`Path. open`): el id es del ancla, no del
    texto visible. Medido sobre el corpus ya construido quedaban 3.364 ocurrencias sin tocar, más
    del doble de lo que la regla arreglaba. Así que las variantes salen de **cada sufijo** del id.
    Las dos mitades siguen saliendo del ancla, y eso es lo que evita que
    `...devuelve un Path. Observá que...` se pegue: `Observá` no es cola de ningún ancla. De las
    202 ocurrencias con cola en mayúscula (`TLSVersion. TLSv1`, `blake2b. SALT_SIZE`) se
    inspeccionaron las 20 más frecuentes: todas eran firmas.
    """
    spaced: dict[str, str] = {}
    for a in anchors:
        parts = a["anchor_id"].split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            for m in re.finditer(r"\.", suffix):
                k = m.start()
                spaced[f"{suffix[:k + 1]} {suffix[k + 1:]}"] = suffix
    if not spaced:
        return None
    pattern = re.compile("|".join(sorted((re.escape(s) for s in spaced), key=len, reverse=True)))

    def fix(text: str) -> str:
        """Hasta punto fijo. Una pasada sola deja a medias las firmas con DOS puntos partidos
        (`socket. socket. type`): `re.sub` reemplaza `socket. socket` y sigue leyendo después
        del reemplazo, así que el segundo punto nunca se ve. Eran 4 ocurrencias en el corpus,
        todas en `py3.12:library/socket`, y las encontró el contador de firmas que quedan."""
        for _ in range(4):
            after = pattern.sub(lambda m: spaced[m.group(0)], text)
            if after == text:
                return text
            text = after
        return text

    return fix


def clean_block(text: str, fix_signature) -> str:
    text = apply("NFC", rule_nfc, text)
    text = apply("marca ¶ -> salto de línea", rule_pilcrow, text)
    if fix_signature is not None:
        text = apply("firma partida (pathlib. Path -> pathlib.Path)", fix_signature, text)
    text = apply("etiqueta de nota de versión al español", rule_note_label, text)
    return apply("espacios en blanco", rule_whitespace, text)


def clean_page(page) -> tuple[str, list[dict]]:
    """Texto limpio y secciones con los offsets recalculados contra ese texto.

    El contrato que verifica: cada `start` y cada `end` de sección cae en un borde de bloque.
    Si `corpus_html` cambiara la forma de concatenar, esto se cae acá en vez de desplazar todas
    las citas en silencio.
    """
    blocks = page.text.split("\n\n")
    fix_signature = signature_fixer(page.anchors)
    pos_map: dict[int, int] = {}
    old = new = 0
    out = []
    for i, block in enumerate(blocks):
        cleaned = clean_block(block, fix_signature)
        if block and not cleaned:
            raise RuntimeError(f"{page.page_id}: la limpieza vació un bloque no vacío")
        for o, n in ((old, new), (old + len(block), new + len(cleaned))):
            if pos_map.get(o, n) != n:
                raise RuntimeError(f"{page.page_id}: offset {o} se mapea a dos lugares")
            pos_map[o] = n
        out.append(cleaned)
        old += len(block) + 2
        new += len(cleaned) + 2
    text = "\n\n".join(out)

    sections = []
    for s in page.sections:
        for field in ("start", "end"):
            if s[field] not in pos_map:
                raise RuntimeError(f"{page.page_id}/{s['section_id']}: {field}={s[field]} no cae "
                                   f"en un borde de bloque -- cambió corpus_html.")
        sections.append({
            "section_id": f"{page.page_id}#{s['section_id']}",
            "title": s["title"], "level": s["level"],
            "start": pos_map[s["start"]], "end": pos_map[s["end"]],
            "parent_id": f"{page.page_id}#{s['parent_id']}" if s["parent_id"] else None,
        })
    return text, sections


# %% [markdown]
r"""
## 3 · Construir los documentos

La regla del subconjunto es `corpus_html.page_class` -- la misma función que la Etapa 0 usó para
medir lo que cuesta, no una copia. Se aplica a las dos versiones.

El mapeo de estado a `DocMetadata.status`, que solo acepta cinco valores:

| de la fuente | `status` | por qué |
|---|---|---|
| página de 3.13 | `current` | 3.13 es la rama vigente del corpus |
| página de 3.12 | `superseded` | la rama entera quedó atrás; el contenido no es falso, es viejo |
| aviso de eliminación (cualquier versión) | `deprecated` | la página dice de sí misma que el módulo se fue |

El valor original queda en `source_status`, que es una clave extra que `DocMetadata` admite.
"""

# %%
only = SMOKE_PAGES if SMOKE_TEST else None
docs: list[Document] = []
rows_meta: dict[str, dict] = {}        # doc_id -> lo que las celdas de abajo necesitan medir
classes: dict[str, list[str]] = collections.defaultdict(list)

for v in VERSIONS:
    for page in corpus_html.iter_pages(v, zip_path=zips[v]["path"], only=only):
        text, sections = clean_page(page)
        tokens = llm.count_tokens(text, GENERATOR) or 0
        klass = corpus_html.page_class(tokens, {n["kind"] for n in page.notes})
        classes[f"{v} · {klass}"].append(page.page_id)
        if klass == "puntero":
            continue
        if klass == "documentación" and page.english_ratio > corpus_html.ENGLISH_CAP:
            classes[f"{v} · documentación en inglés"].append(page.page_id)
            continue
        notice = klass == "aviso de eliminación"
        versions_named = sorted({n["version"] for n in page.notes if n["version"]})
        docs.append(Document(
            doc_id=page.page_id, title=page.title, text=text,
            sections=[Section(**s) for s in sections],
            metadata=DocMetadata(
                version=v,
                date=None,            # medido en la Etapa 0: el corpus no trae fechas. Etapa 6 las simula.
                status="deprecated" if notice else ("current" if v == CURRENT else "superseded"),
                source_url=page.url, lang="es",
                source_status="removed module notice" if notice else f"python {v} docs",
                top=page.top, rel=page.rel, doc_class=klass,
                english_ratio=round(page.english_ratio, 4),
                unmarked_ratio=round(page.unmarked_ratio, 4),
                tokens=tokens,
                versions_named=versions_named,
                anchors=[{"anchor_id": a["anchor_id"],
                          "section_id": f"{page.page_id}#{a['section_id']}"} for a in page.anchors],
                # Las notas viajan con el documento para que la Etapa 2 pueda DERIVAR las
                # preguntas temporales desde `documents.jsonl`, sin volver a abrir el zip: si
                # derivara desde otra extracción, el gold podría apuntar a secciones que el
                # corpus indexado no tiene.
                notes=[{"kind": n["kind"], "version": n["version"],
                        "section_id": f"{page.page_id}#{n['section_id']}"}
                       for n in page.notes if n["version"]],
            )))
        rows_meta[page.page_id] = {"page": page, "raw": page.text, "clean": text, "tokens": tokens}

by_version = collections.Counter(d.metadata.version for d in docs)
print(f"documentos: {len(docs)}  ({dict(by_version)})")
print(f"secciones:  {sum(len(d.sections) for d in docs):,}")
print(f"tokens:     {sum(d.metadata.tokens for d in docs):,}")
print_table("qué entró y qué no, por versión y clase", ["páginas"],
            [(k, {"páginas": len(ids)}) for k, ids in sorted(classes.items())],
            smoke=SMOKE_TEST, name_header="clase")

# %% [markdown]
r"""
### El agujero de `builtins`, y que no es el agujero que creíamos

La Etapa 0 cerró la regla del subconjunto con un costo escrito con nombre propio: *"`builtins/`
entero está sin traducir y la regla la descarta. Ninguna pregunta nuestra puede ser sobre `print`
o `len`."* **Aplicar la regla a las dos versiones mostró que esa frase estaba mal.**

Lo que pasó: 3.13 movió la referencia de las funciones integradas de `library/functions.html` a
`builtins/functions.html` y dejó un stub de redirección en la ruta vieja. La página nueva está
sin traducir (98 % en inglés) y la regla la descarta. Pero **3.12 todavía la tiene en
`library/functions.html`, traducida al 82 %**, y la regla la conserva.

O sea el contenido está en el corpus; lo que no está es su versión vigente. Eso no es un detalle
de cobertura, es una interacción con el resto del plan:

- una pregunta sobre `print` tiene gold en **una sola** página, y es una página
  `status="superseded"`;
- el filtro por versión de la Etapa 3, si restringe a 3.13, vuelve esa pregunta **incontestable**
  -- que es justo el tipo de caso donde un filtro de metadatos "mejora la precisión" y pierde la
  respuesta;
- y es un gancho de memoria mejor que el que teníamos: *"estoy en 3.13"* más una pregunta de
  builtins es un conflicto real entre lo que el usuario dijo y lo que el corpus puede probar.

Esto es la celda que justifica por qué el filtro por metadatos es una **rama etiquetada** de la
Etapa 3 y no un escalón de la escala acumulativa.
"""

# %%
MOVED = ("library/functions.html", "library/stdtypes.html", "library/exceptions.html",
         "library/constants.html")
moved_rows = []
for rel in MOVED:
    here = {d.metadata.version: d for d in docs if d.metadata.rel == rel}
    moved_rows.append((rel.replace("library/", "").replace(".html", ""), {
        "en 3.13": "stub de redirección" if "3.13" not in here else f"{here['3.13'].metadata.tokens:,} tk",
        "en 3.12": f"{here['3.12'].metadata.tokens:,} tk, "
                   f"{here['3.12'].metadata.english_ratio:.0%} inglés" if "3.12" in here else "—",
        "anclas recuperadas": len(here["3.12"].metadata.anchors) if "3.12" in here else 0,
        "status del único gold": here["3.12"].metadata.status if "3.12" in here else "—"}))
print_table("las páginas que 3.13 movió a `builtins/` y 3.12 todavía tiene traducidas",
            ["en 3.13", "en 3.12", "anclas recuperadas", "status del único gold"],
            moved_rows, smoke=SMOKE_TEST, name_header="página")
print(verdict(all(r[1]["anclas recuperadas"] for r in moved_rows),
              "el contenido de los builtins entra al corpus por la puerta de 3.12: las preguntas "
              "sobre `print` o `len` son posibles, con gold de una sola versión y superseded",
              "ni por 3.12 entra: ahí sí ninguna pregunta puede ser sobre los builtins"))

# %% [markdown]
r"""
## 4 · Casi-duplicados: versiones, no copias

La Etapa 0 lo midió sobre el corpus entero con MinHash: 375 páginas de 3.13 tienen gemela en
3.12, con Jaccard de 5-gramas mediano **0,86**. Acá hace falta la *política*, no otra medición
masiva, así que la verificación corre sobre una muestra con Jaccard exacto -- distinto método,
por si el MinHash se estaba equivocando.

**La política: son versiones, no copias.** Dan respuestas distintas a la misma pregunta
("¿desde qué versión…?"), así que no se deduplica ninguna. Las dos llevan `canonical_id` = el
`doc_id` de 3.13, que es lo que permite, en la Etapa 2, que una pregunta sin versión tenga un
*conjunto* gold con las dos páginas, y en la Etapa 3, que el filtro de versión sea una rama
medible en vez de un truco que parece arruinar la métrica.

No hay copias intercambiables dentro de una versión: los pares confundibles que la Etapa 0
encontró (`asynchat ~ asyncore`, contención de vocabulario 0,975) comparten vocabulario, no
texto -- su Jaccard de 5-gramas es 0,67. Son dos módulos distintos que dicen lo mismo con
palabras distintas, y la respuesta correcta a una pregunta sobre uno no es el otro.
"""


# %%
def shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    """Jaccard exacto sobre 5-gramas de palabras. Sin MinHash: la muestra es chica y el punto
    de esta celda es confirmar con otro método lo que la Etapa 0 estimó."""
    words = text.lower().split()
    return {tuple(words[i:i + k]) for i in range(max(len(words) - k + 1, 1))}


by_rel: dict[str, dict[str, Document]] = collections.defaultdict(dict)
for d in docs:
    by_rel[d.metadata.rel][d.metadata.version] = d

twins = [rel for rel, per in by_rel.items() if len(per) == len(VERSIONS)]
for rel, per in by_rel.items():
    canonical = per.get(CURRENT) or next(iter(per.values()))
    for d in per.values():
        d.metadata.canonical_id = canonical.doc_id
        d.metadata.duplicate_of = None     # política: ninguna página es copia de otra
        d.metadata.versions_of = sorted(per)

sample = sorted(twins)[:: max(len(twins) // 20, 1)][:20]
pairs = []
for rel in sample:
    a, b = by_rel[rel][VERSIONS[0]], by_rel[rel][VERSIONS[1]]
    sa, sb = shingles(a.text), shingles(b.text)
    pairs.append((len(sa & sb) / max(len(sa | sb), 1), rel))
pairs.sort()
jac = [j for j, _ in pairs]
if jac:
    print(f"gemelas (la misma página en las dos versiones): {len(twins)} de {len(by_rel)} rutas")
    print(f"Jaccard exacto de 5-gramas sobre {len(jac)} de ellas: "
          f"mín {jac[0]:.3f} · mediana {jac[len(jac) // 2]:.3f} · máx {jac[-1]:.3f}")
    # La gemela MENOS parecida es la interesante: es una página que 3.13 reescribió, así que la
    # pregunta sobre ella sí depende de la versión y su gold NO puede ser un conjunto.
    print(f"la gemela menos parecida: {pairs[0][1]} ({pairs[0][0]:.3f}) -- una página reescrita "
          f"entre versiones. Para la Etapa 2 son las que necesitan gold de una sola versión.")
    print(verdict(jac[len(jac) // 2] > 0.5,
                  "confirma el MinHash de la Etapa 0: son la misma página en dos versiones, y "
                  "una métrica por documento que cuente la otra versión como fallo no mide nada",
                  "la muestra NO confirma la medición de la Etapa 0: revisar antes de etiquetar"))
solo = {v: sorted(d.metadata.rel for d in docs
                  if d.metadata.version == v and len(by_rel[d.metadata.rel]) == 1)
        for v in VERSIONS}
for v, rels in solo.items():
    print(f"\nsolo en {v}: {len(rels)} páginas  {[r.split('/')[-1] for r in rels[:8]]}")

# %% [markdown]
r"""
## 5 · Metadatos que mienten

`version` tiene 100 % de cobertura, y eso es sospechoso: no lo leímos de la página, lo pusimos
nosotros según de qué zip salió. Un campo así es el caso típico de la trampa -- cobertura
perfecta, cero información verificada. La Etapa 3 va a filtrar por él, así que hay que
contrastarlo con el cuerpo.

**La prueba.** Una página de 3.12 no puede hablar de algo que llegó en 3.13: si una nota de
versión de una página de 3.12 nombra una versión mayor que 3.12, o la traducción se importó de
un build más nuevo o nuestra etiqueta está mal. Lo mismo al revés no es un error (3.13 sí puede
decir "nuevo en 3.4"), así que la prueba es de un solo lado.
"""


# %%
def branch(v: str) -> tuple[int, ...]:
    """Solo (mayor, menor). El primer intento comparaba la versión entera y marcaba
    `library/os.path` de 3.13 por decir "3.13.4": una nota de parche de su propia rama, no una
    contradicción. La etiqueta de versión es una rama, así que la comparación también."""
    parts = [int(x) for x in v.split(".") if x.isdigit()]
    return tuple(parts[:2])


disagree = []
for d in docs:
    own = branch(d.metadata.version)
    ahead = [v for v in d.metadata.versions_named if branch(v) > own]
    if ahead:
        disagree.append((d.doc_id, ahead))
checked = [d for d in docs if d.metadata.versions_named]
print(f"documentos con alguna nota de versión: {len(checked)} de {len(docs)}")
print(f"documentos cuya nota nombra una versión POSTERIOR a su propia etiqueta: {len(disagree)} "
      f"({100 * len(disagree) / max(len(checked), 1):.2f} % de los verificables)")
for doc_id, ahead in disagree[:5]:
    print(f"   {doc_id}: {ahead}")
print(verdict(len(disagree) / max(len(checked), 1) < 0.02,
              "la etiqueta de versión concuerda con el cuerpo: el filtro de la Etapa 3 se puede "
              "construir sobre ella",
              "la etiqueta de versión y el cuerpo se contradicen demasiado: el filtro de versión "
              "de la Etapa 3 arrastra ese error y hay que reportar la tasa junto al resultado"))

# `status` también lo pusimos nosotros, pero no por construcción: sale de la clase de página.
# La prueba es que el cuerpo lo diga. Las frases son las que usan los avisos de eliminación.
GONE = re.compile(r"no (forma parte|es parte)|se elimin|was removed|removed in|is deprecated"
                  r"|Obsoleto desde la versión|PEP 594", re.IGNORECASE)
dep = [d for d in docs if d.metadata.status == "deprecated"]
dep_silent = [d.doc_id for d in dep if not GONE.search(d.text)]
cur = [d for d in docs if d.metadata.status == "current"]
print(f"\nstatus=deprecated: {len(dep)} documentos, {len(dep_silent)} sin ninguna frase de "
      f"eliminación en el cuerpo {dep_silent[:5]}")
print(verdict(not dep_silent,
              "cada página marcada deprecated lo dice en su propio texto",
              "hay páginas marcadas deprecated que el cuerpo no respalda: revisar a mano"))

# %% [markdown]
r"""
## 6 · Qué cambió la limpieza

El paso 8, en dos partes: los contadores por regla, y un diff antes/después sobre diez
documentos escrito a disco para leerlo con los ojos.
"""

# %%
total_raw = sum(len(m["raw"]) for m in rows_meta.values())
total_clean = sum(len(m["clean"]) for m in rows_meta.values())
ALL_RULES = ["NFC", "marca ¶ -> salto de línea",
             "firma partida (pathlib. Path -> pathlib.Path)",
             "etiqueta de nota de versión al español", "espacios en blanco"]
print_table("la limpieza, regla por regla",
            ["bloques que tocó", "caracteres que sacó", "% del corpus crudo"],
            [(rule, {"bloques que tocó": CLEANING_HITS.get(rule, 0),
                     "caracteres que sacó": CLEANING_CHARS.get(rule, 0),
                     "% del corpus crudo": f"{100 * CLEANING_CHARS.get(rule, 0) / total_raw:.3f}"})
             for rule in sorted(ALL_RULES, key=lambda r: -CLEANING_CHARS.get(r, 0))],
            smoke=SMOKE_TEST, name_header="regla")
print("la unidad es el BLOQUE (el texto propio de una sección), no el documento: la limpieza "
      "corre por bloque para poder recalcular los offsets.")
print("un cero también es un resultado: NFC no tocó nada (Sphinx ya emite NFC) y los espacios "
      "tampoco (`corpus_html` los colapsa al extraer). Quedan igual, porque el día que cambie "
      "el extractor el contador lo va a decir.")
print("la etiqueta de nota AGREGA caracteres: 'Added in version' es más corto que 'Nuevo en la "
      "versión'. El número de versión es el dato y queda intacto; el largo no importa.")
print(f"corpus crudo {total_raw:,} caracteres -> limpio {total_clean:,} "
      f"({100 * (total_raw - total_clean) / total_raw:.2f} % menos)")

diff_dir = ART / "stage1_cleaning_diff"
diff_dir.mkdir(parents=True, exist_ok=True)
sample_docs = sorted(rows_meta.items(), key=lambda kv: -len(kv[1]["raw"]))[:5] + \
              sorted(rows_meta.items(), key=lambda kv: len(kv[1]["raw"]))[:5]
for doc_id, m in sample_docs:
    name = doc_id.replace(":", "_").replace("/", "_") + ".diff.txt"
    (diff_dir / name).write_text(
        f"{doc_id}\ncrudo {len(m['raw']):,} caracteres -> limpio {len(m['clean']):,}\n"
        f"{'=' * 100}\nCRUDO (primeros 3.000)\n{'=' * 100}\n{m['raw'][:3000]}\n\n"
        f"{'=' * 100}\nLIMPIO (primeros 3.000)\n{'=' * 100}\n{m['clean'][:3000]}\n",
        encoding="utf-8")
print(f"\ndiff antes/después de {len(sample_docs)} documentos en: {diff_dir}")
print("PENDIENTE (una persona): leer dos o tres y confirmar que la limpieza no inventó texto "
      "en ningún borde.")

# Lo que la regla NO arregló. El contador de arriba dice cuántas veces disparó; esto dice
# cuántas firmas partidas quedan, que es la pregunta que importa y la que descubrió que la
# primera versión de la regla cubría la mitad del problema.
leftover = 0
for d in docs:
    variants = set()
    for a in d.metadata.anchors:
        parts = a["anchor_id"].split(".")
        for i in range(len(parts)):
            suffix = ".".join(parts[i:])
            for m in re.finditer(r"\.", suffix):
                variants.add(f"{suffix[:m.start() + 1]} {suffix[m.start() + 1:]}")
    leftover += sum(d.text.count(v) for v in variants)
print(f"firmas partidas que quedan en el texto limpio: {leftover:,} "
      f"(antes de extender la regla a los sufijos cortos eran 3.364)")

# El assert que vale la pena: los números de versión son el dato de las preguntas temporales.
lost = [(d.doc_id, v) for d in docs for v in d.metadata.versions_named if v not in d.text]
if lost:
    raise RuntimeError(f"la limpieza se comió números de versión: {lost[:5]}")
print(f"\nverificado: los {sum(len(d.metadata.versions_named) for d in docs):,} números de "
      f"versión que las notas declaran siguen literales en el texto limpio.")

# %% [markdown]
r"""
## 7 · Escribir, con manifiesto, y validar

`documents_hash` es el hash del *contenido*, independiente del orden en que se escribieron las
filas: va en el manifiesto de cada índice, así un índice puede decir de qué corpus salió. El
manifiesto se escribe a mano en vez de con `build_or_load` a propósito: si esta etapa se
saltease con caché, los números de limpieza de la celda anterior no existirían, y son parte de
la entrega.
"""

# %%
docs.sort(key=lambda d: d.doc_id)
out = ART / "documents.jsonl"
n = store.write_jsonl(out, docs)
corpus_sha = documents_hash(docs)

spec = {
    "stage": "01_documents", "profile": PROFILE,
    "zips": {v: zips[v]["sha256"] for v in VERSIONS},
    "downloaded_on": DOWNLOADED_ON,
    "subset": {"directories": list(corpus_html.SUBSET),
               "english_cap": corpus_html.ENGLISH_CAP,
               "min_tokens": corpus_html.MIN_TOKENS,
               "stub_max_tokens": corpus_html.STUB_MAX_TOKENS},
    "cleaning_rules": sorted(CLEANING_HITS),
    "generator_for_token_counts": GENERATOR,
    "parser": store.sha256_file(corpus_html.__file__),
    "smoke_pages": sorted(SMOKE_PAGES) if SMOKE_TEST else None,
}
store.write_manifest(out, spec, documents=n, corpus_sha256=corpus_sha,
                     sections=sum(len(d.sections) for d in docs),
                     tokens=sum(d.metadata.tokens for d in docs),
                     cleaning_hits=CLEANING_HITS, cleaning_chars=CLEANING_CHARS)
print(f"escrito: {out}  ({n} documentos)")
print(f"corpus_sha256: {corpus_sha}")

validate_documents(out).show().raise_if_failed()

# %%
summary = {
    "profile": PROFILE, "corpus_sha256": corpus_sha,
    "documents": n, "by_version": dict(by_version),
    "sections": sum(len(d.sections) for d in docs),
    "tokens": sum(d.metadata.tokens for d in docs),
    "characters": {"raw": total_raw, "clean": total_clean},
    "cleaning_hits": CLEANING_HITS, "cleaning_chars": CLEANING_CHARS,
    "classes": {k: len(v) for k, v in sorted(classes.items())},
    "twins": len(twins), "twin_jaccard_sample": jac,
    "only_in": {v: len(rels) for v, rels in solo.items()},
    "metadata_disagreement": {"checked": len(checked), "version_ahead": len(disagree),
                              "deprecated_without_phrase": dep_silent},
    "status_counts": dict(collections.Counter(d.metadata.status for d in docs)),
    "zips": {v: zips[v]["sha256"] for v in VERSIONS}, "downloaded_on": DOWNLOADED_ON,
}
store.write_json(ART / "stage1_documents.json", summary)
run.note("documents", n)
run.note("corpus_sha256", corpus_sha)
run.note("cleaning_chars", CLEANING_CHARS)
run.note("metadata_version_disagreement", len(disagree))
run.finish()
