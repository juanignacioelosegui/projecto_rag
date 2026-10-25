"""
Converts a `.py` with `# %%` cell markers into a real `.ipynb`.

    python tools/py_to_notebook.py                   # regenerate every stage that exists
    python tools/py_to_notebook.py --check            # fail if any .ipynb is stale or broken
    python tools/py_to_notebook.py --check --publish  # ...and if REPO_URL is still a placeholder
    python tools/py_to_notebook.py --only 00_corpus_audit
    python tools/py_to_notebook.py --source foo.py --destination foo.ipynb

WHY THIS STEP EXISTS
--------------------
The `.py` is the EDITABLE source: it diffs cleanly in git, greps normally, runs as a plain
script (`python 00_corpus_audit.py`) and carries no outputs. The `.ipynb` is the ARTIFACT
you open in Colab. Edit the `.py`, regenerate.

EXPECTED FORMAT
---------------
    # %%                 -> code cell
    # %% [markdown]      -> markdown cell, whose content is ONE triple-quoted string
                            (a raw string recommended), so the prose has no `#` per line.

Everything before the first `# %%` is discarded (it's the file's header, not a cell).
Code cells must be plain Python: no `!pip` / `%magic` lines (they make the `.py` invalid
as a script, and `--check` rejects them). Shell steps go through `subprocess`.
A line starting with `# %%` INSIDE a markdown string splits the cell there -- `--check`
catches that too, because the halves no longer parse.

COLAB BOOTSTRAP
---------------
Every generated notebook gets an extra first cell that, on Colab only, clones this repo,
`cd`s into it, pulls any published fixes, installs the requirements once per VM, and --
if you set USE_DRIVE -- keeps `artifacts/` on Google Drive so a disconnect doesn't cost
you your index. It's injected here so the clone URL lives in one place. **Set REPO_URL
before publishing** (`--check --publish` fails until you do).

Stdlib only: no dependency on jupytext or nbformat.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STAGE_STEMS = (
    "00_corpus_audit",
    "00b_corpus_audit_ours",   # nuestra Etapa 0: la resuelta es donde empieza, no donde termina
    "01_documents",
    "02_eval_set",
    "03_retrieval",
    "04_generation",   # nuestros nombres: el enunciado sugería 04_grounded_answers / 05_conversation
    "05_dialogue",
    "06_memory",       # sin Etapa 7: el plan de la Entrada 1 del registro la deja afuera
)
STAGES: dict[str, tuple[Path, Path]] = {s: (ROOT / f"{s}.py", ROOT / f"{s}.ipynb") for s in STAGE_STEMS}

CELL_MARKER = re.compile(r"^#\s*%%(.*)$")

# --- Colab bootstrap: EDIT THESE when you publish (the only place they appear) ----------
URL_PLACEHOLDER = "CHANGEME"
REPO_URL = "https://github.com/juanignacioelosegui/projecto_rag.git"
REPO_BRANCH = ""          # "" = the repository's default branch
REPO_DIR = "projecto_rag"
DRIVE_DIR = "projecto_rag"   # folder under MyDrive used when USE_DRIVE = True

BOOTSTRAP_TEMPLATE = '''# --- Bootstrap de Colab (no hace nada fuera de Colab) --------------------------------
# Colab arranca en /content con un entorno vacío: sin esta celda, el `import environment`
# siguiente falla con ModuleNotFoundError. La genera tools/py_to_notebook.py -- editala
# ahí, no acá.
import hashlib
import os
import subprocess
import sys

# El disco de Colab se borra cuando se desconecta el entorno de ejecución (~90 min inactivo,
# 12 h como máximo). Con USE_DRIVE = True, artifacts/ (documentos, embeddings, índices,
# resultados, cachés del LLM) vive en cambio en tu Google Drive, y al volver a correr se
# saltea todo lo que ya está construido.
USE_DRIVE = False
# Guardar también en Drive la caché de modelos de HuggingFace (~3 GB). Apagado por defecto:
# volver a descargar suele ser más rápido que leer archivos grandes a través del montaje de Drive.
MODELS_ON_DRIVE = False

# Un entorno de ejecución de Colab, desde el navegador o desde la extensión de Colab de VS Code
# (cuyo kernel puede no haber importado google.colab: la propia variable de entorno del runtime
# dice dónde estamos).
if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
    REPO_URL = "@REPO_URL@"
    REPO_BRANCH = "@REPO_BRANCH@"
    REPO_DIR = "@REPO_DIR@"
    DRIVE = "/content/drive/MyDrive/@DRIVE_DIR@"
    quiet = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}  # nunca esperar un pedido de contraseña

    def github_token():
        # Un repositorio privado: un token de GitHub como secreto de Colab llamado GITHUB_TOKEN.
        # Los secretos de Colab solo se pueden leer desde el navegador; una variable de entorno
        # funciona en cualquier lado.
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            try:
                from google.colab import userdata
                token = userdata.get("GITHUB_TOKEN")
            except Exception:
                token = None
        return token

    # Dónde ESTAMOS antes que qué hay en disco: `isdir(REPO_DIR)` es relativo, así que
    # chequearlo primero clonaría una copia anidada cada vez que se vuelve a correr esta celda
    # en la misma sesión.
    if os.path.basename(os.getcwd()) != REPO_DIR:
        if not os.path.isdir(REPO_DIR):
            cmd = ["git", "clone", "--depth", "1"] + (["--branch", REPO_BRANCH] if REPO_BRANCH else [])
            if subprocess.run(cmd + [REPO_URL, REPO_DIR], env=quiet).returncode != 0:
                token = github_token()
                if not token:
                    raise SystemExit("git clone failed. Is the repository public? For a private one, "
                                     "add a Colab secret named GITHUB_TOKEN (in browser Colab: VS Code's "
                                     "extension can't read Colab secrets).")
                authed = REPO_URL.replace("https://", "https://x-access-token:" + token + "@")
                # capture_output: un mensaje de error no tiene que imprimir el token en el notebook
                if subprocess.run(cmd + [authed, REPO_DIR], env=quiet, capture_output=True).returncode != 0:
                    raise SystemExit("git clone failed even with GITHUB_TOKEN: check that the token "
                                     "can read this repository (fine-grained: Contents: Read).")
                # El token NO se guarda en .git/config: un pull posterior lo vuelve a pasar (abajo).
                subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", REPO_URL])
        os.chdir(REPO_DIR)
    # Traer los arreglos publicados después de tu primer clon (no hace nada si no hay).
    pulled = subprocess.run(["git", "pull", "--ff-only", "-q"], capture_output=True, text=True, env=quiet)
    token = github_token() if pulled.returncode != 0 else None
    if token:   # un repositorio privado: volver a hacer pull con el token, sin guardarlo
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        authed = REPO_URL.replace("https://", "https://x-access-token:" + token + "@")
        pulled = subprocess.run(["git", "pull", "--ff-only", "-q", authed, branch], capture_output=True,
                                text=True, env=quiet)
    if pulled.returncode != 0:
        message = (pulled.stderr or pulled.stdout).strip()
        print("note: git pull skipped:", (message.replace(token, "***") if token else message)[:200])
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    # Instalar una vez por VM: una marca con clave en el contenido de los archivos de requirements.
    reqs = [f for f in ("requirements.txt", "requirements-group.txt") if os.path.isfile(f)]
    digest = hashlib.sha256(b"".join(open(f, "rb").read() for f in reqs)).hexdigest()[:12]
    stamp = "/content/.lab_pip_" + digest
    if not os.path.exists(stamp):
        pip = [sys.executable, "-m", "pip", "install", "-q"]
        for f in reqs:
            pip += ["-r", f]
        subprocess.run(pip, check=True)  # UNA sola resolución: una versión del grupo que choca con una del núcleo falla ruidosamente
        open(stamp, "w").close()

    if MODELS_ON_DRIVE and not USE_DRIVE:
        print("note: MODELS_ON_DRIVE needs USE_DRIVE = True; the model cache stays on the VM disk.")
    if USE_DRIVE:
        # ANTES de `import environment`: un directorio artifacts/ real creado primero le ganaría
        # en silencio al link, y todo terminaría en el disco efímero.
        if not os.path.isdir("/content/drive/MyDrive"):   # VS Code: ya montado por su comando
            # Se avisa ANTES de la llamada: en la extensión de Colab de VS Code, drive.mount() puede colgarse en vez de fallar.
            print("Mounting Drive... (in VS Code, if this hangs: interrupt, run 'Colab: Mount Google Drive "
                  "to Server...' from Ctrl+Shift+P, then re-run this cell)")
            try:
                from google.colab import drive
                drive.mount("/content/drive")
            except Exception as e:  # drive.mount() desde una celda no funciona en la extensión de Colab de VS Code
                raise SystemExit("Drive is not mounted (" + type(e).__name__ + "). In VS Code: Ctrl+Shift+P > "
                                 "'Colab: Mount Google Drive to Server...', then re-run this cell.")
        os.makedirs(DRIVE + "/artifacts", exist_ok=True)
        if os.path.islink("artifacts"):
            print("artifacts ->", os.readlink("artifacts"))
        elif os.path.isdir("artifacts"):
            print("WARNING: artifacts/ is already a real directory, so it is NOT on Drive.")
            print("Move its contents into", DRIVE + "/artifacts", ", delete it, then re-run this cell.")
        else:
            os.symlink(DRIVE + "/artifacts", "artifacts")
            print("artifacts ->", DRIVE + "/artifacts")
        os.environ["LAB_USE_DRIVE"] = "1"
        if MODELS_ON_DRIVE:
            os.makedirs(DRIVE + "/hf_cache", exist_ok=True)
            os.environ.setdefault("HF_HOME", DRIVE + "/hf_cache")

    print("Colab bootstrap OK. Working directory:", os.getcwd())
'''


def bootstrap_cell() -> str:
    return (BOOTSTRAP_TEMPLATE.replace("@REPO_URL@", REPO_URL).replace("@REPO_BRANCH@", REPO_BRANCH)
            .replace("@REPO_DIR@", REPO_DIR).replace("@DRIVE_DIR@", DRIVE_DIR)).rstrip("\n")


def _to_lines(text: str) -> list[str]:
    """ipynb stores `source` as a list of lines, each ending in \\n except the last."""
    if not text:
        return []
    lines = text.splitlines()
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def _trim(block: list[str]) -> str:
    while block and not block[0].strip():
        block.pop(0)
    while block and not block[-1].strip():
        block.pop()
    return "\n".join(block)


def _text_from_string_literal(content: str, where: str) -> str:
    try:
        body = ast.parse(content).body
    except SyntaxError as e:
        raise SystemExit(f"{where}: markdown cell doesn't parse as Python ({e}). If the prose "
                         "contains a line starting with `# %%`, that line split the cell.") from e
    if len(body) == 1 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[0].value.value.strip("\n")
    raise SystemExit(f'{where}: a `# %% [markdown]` cell must contain exactly ONE string literal '
                     f'(r"""..."""). It starts with: {content[:80]!r}')


def split_into_cells(source: str, where: str = "<source>") -> list[tuple[str, str]]:
    """[(kind, content), ...] with kind in {'code', 'markdown'}."""
    cells: list[tuple[str, str]] = []
    kind: str | None = None
    buffer: list[str] = []
    start_line = 0

    def close():
        if kind is None:
            return
        content = _trim(list(buffer))
        if not content:
            return
        if kind == "markdown":
            content = _text_from_string_literal(content, f"{where}:{start_line}")
        cells.append((kind, content))

    for n, line in enumerate(source.splitlines(), start=1):
        m = CELL_MARKER.match(line)
        if m:
            close()
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
            buffer, start_line = [], n
        elif kind is not None:
            buffer.append(line)
    close()
    return cells


def check_code_cells(cells: list[tuple[str, str]], where: str) -> list[str]:
    """Every code cell must compile as plain Python (catches `!` / `%` magics too)."""
    problems = []
    for i, (kind, content) in enumerate(cells, start=1):
        if kind != "code":
            continue
        try:
            compile(content, f"{where} cell {i}", "exec")
        except SyntaxError as e:
            problems.append(f"{where} cell {i}: not plain Python ({e.msg}, line {e.lineno}): "
                            f"{(e.text or '').strip()[:60]!r}")
    return problems


def build_notebook(cells: list[tuple[str, str]]) -> dict:
    output = []
    for i, (kind, content) in enumerate(cells, start=1):
        cell = {"id": f"cell-{i:03d}", "cell_type": kind, "metadata": {}, "source": _to_lines(content)}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        output.append(cell)
    return {
        "cells": output,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(source: Path) -> tuple[str, list[tuple[str, str]], list[str]]:
    """(notebook JSON text, cells incl. bootstrap, problems)."""
    cells = split_into_cells(source.read_text(encoding="utf-8"), source.name)
    problems = [] if cells else [f"no `# %%` cells found in {source.name}"]
    cells = [("code", bootstrap_cell())] + cells
    problems += check_code_cells(cells, source.name)
    text = json.dumps(build_notebook(cells), ensure_ascii=False, indent=1) + "\n"
    return text, cells, problems


def convert_one(source: Path, destination: Path, check: bool) -> bool:
    if not source.is_file():
        print(f"missing source: {source}")
        return False
    text, cells, problems = render(source)
    for p in problems:
        print(f"ERROR {p}")
    if problems:
        return False
    if check:
        if not destination.is_file():
            print(f"MISSING {destination.name}: run without --check.")
            return False
        if destination.read_bytes() != text.encode("utf-8"):
            print(f"STALE {destination.name} (differs from {source.name}): run without --check.")
            return False
        print(f"OK {destination.name} ({len(cells)} cells)")
        return True
    with open(destination, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    n_md = sum(1 for k, _ in cells if k == "markdown")
    print(f"written {destination.name}: {len(cells)} cells ({len(cells) - n_md} code, {n_md} markdown)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="don't write; fail if any .ipynb is stale/broken")
    ap.add_argument("--publish", action="store_true", help="with --check: also fail on a placeholder REPO_URL")
    ap.add_argument("--only", help="one stage stem, e.g. 00_corpus_audit")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--destination", type=Path)
    args = ap.parse_args()

    if args.source is not None:
        return 0 if convert_one(args.source, args.destination or args.source.with_suffix(".ipynb"),
                                args.check) else 1

    if args.only is not None and args.only not in STAGES:
        print(f"unknown stage {args.only!r}; known: {', '.join(STAGES)}")
        return 2
    pairs = STAGES.items() if args.only is None else [(args.only, STAGES[args.only])]

    all_ok, found = True, False
    for _, (source, destination) in pairs:
        if not source.is_file():
            continue  # a stage that hasn't been written yet
        found = True
        all_ok = convert_one(source, destination, args.check) and all_ok
    if not found:
        print("no stage .py files found.")
        return 1

    if URL_PLACEHOLDER in REPO_URL:
        if args.check and args.publish:
            print(f"ERROR REPO_URL is still the {URL_PLACEHOLDER} placeholder: every notebook's "
                  "bootstrap would fail on `git clone`. Set it in tools/py_to_notebook.py.")
            all_ok = False
        elif not args.check:
            print(f"\nnote: REPO_URL is still the {URL_PLACEHOLDER} placeholder -- set it before "
                  "publishing (`--check --publish` fails until you do).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
