"""sync_shared.py — copia os modulos COMPARTILHADOS do repo de treino para `core/`.

Fonte unica (repo de treino, `ptbr_lora/core/`): text_norm, text_blocks, adapter_scale,
reference_prep. Para nao ter 2 copias divergindo (foi assim que a UI ficou com um
`text_norm` diferente do usado no treino/avaliacao), edite SO la e rode:

  python scripts/sync_shared.py            # copia
  python scripts/sync_shared.py --check    # so verifica (exit 1 se divergir) — use em CI

Localizacao do repo de treino: --src, ou $PTBR_TRAINING_REPO, ou a pasta irma
`../Breeze-tts2-ptbr-lora-training`.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import sys
from pathlib import Path

SHARED = ["text_norm.py", "text_blocks.py", "adapter_scale.py", "reference_prep.py"]
HERE = Path(__file__).resolve().parents[1]


def find_src(arg: str | None) -> Path:
    cands = [arg, os.environ.get("PTBR_TRAINING_REPO"),
             str(HERE.parent / "Breeze-tts2-ptbr-lora-training"),
             str(HERE.parents[1] / "Breeze-tts2-ptbr-lora-training")]
    for c in cands:
        if c and (Path(c) / "ptbr_lora" / "core" / "text_norm.py").is_file():
            return Path(c) / "ptbr_lora" / "core"
    sys.exit("[sync] repo de treino nao encontrado (use --src ou PTBR_TRAINING_REPO)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="raiz do repo de treino")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    src = find_src(a.src)
    dst = HERE / "core"
    bad = 0
    for name in SHARED:
        s, d = src / name, dst / name
        same = d.is_file() and filecmp.cmp(s, d, shallow=False)
        if same:
            print(f"[sync] {name}: ok")
            continue
        if a.check:
            print(f"[sync] {name}: DIVERGE")
            bad += 1
        else:
            shutil.copyfile(s, d)
            print(f"[sync] {name}: copiado de {s}")
    if a.check and bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
