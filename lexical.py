"""La normalización léxica, en un solo lugar.

El enunciado de la Etapa 2 pone el chequeo de contaminación en línea y avisa, en el comentario
del fragmento: *«La MISMA normalización que usa tu canal léxico: un chequeo que tokeniza
distinto que BM25 mide un solapamiento distinto del que infla a BM25.»* Una copia del
tokenizador en el notebook 02 y otra en el 03 son dos cosas que empiezan iguales y terminan
distintas sin que nadie se entere: el chequeo diría que no hay contaminación mientras BM25
cobra las frases copiadas. Así que vive acá, y los dos lo importan.

`tokenize` devuelve una lista de tokens lista para `bm25s.BM25.index`, que acepta listas de
listas de cadenas: el índice de la Etapa 3 se construye con esta función, no con
`bm25s.tokenize`, para que no haya dos stemmers.
"""

from __future__ import annotations

import re

import Stemmer  # PyStemmer

_stemmer = Stemmer.Stemmer("spanish")
stem = _stemmer.stemWord

# La lista del enunciado, textual. Corta a propósito: una lista de stopwords más larga es otra
# decisión de recuperación, y esta etapa no la mide.
STOP = frozenset("de la que el en y los se del las un por con no una su para es al lo como más".split())

_WORD = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """minúsculas -> \\w+ -> sin stopwords -> stem español."""
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOP]


# El nombre con el que el enunciado la llama.
content_words = tokenize


def contamination(question: str, gold_text: str, n: int = 5) -> dict:
    """Cuánto de la pregunta ya está en el pasaje gold.

    `overlap` es la fracción de tokens de contenido de la pregunta que aparecen en el pasaje;
    `longest_shared_ngram` es la frase más larga compartida, en tokens. La bandera mira la
    frase, no el solapamiento: pedir «¿cómo ordeno una lista por clave?» sobre una página de
    ordenamiento comparte casi todas las palabras y eso no es copia, es el tema. Cinco tokens
    seguidos sí lo son.
    """
    q, g = tokenize(question), tokenize(gold_text)
    overlap = sum(w in set(g) for w in q) / max(len(q), 1)
    grams = lambda ws, k: {tuple(ws[i:i + k]) for i in range(len(ws) - k + 1)}
    shared = max((k for k in range(len(q), 0, -1) if grams(q, k) & grams(g, k)), default=0)
    return {"overlap": round(overlap, 2), "longest_shared_ngram": shared, "flag": shared >= n}
