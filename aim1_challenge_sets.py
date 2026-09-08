#!/usr/bin/env python3
"""E1a - AIM 1, dependency-aware challenge sets (0 new fits).

One launcher for the three analyses that share E1a's population and estimator.
They were three top-level scripts - `aim1_e1a_challenge_sets.py`,
`aim1_e1a_vs_acomplete.py` and `aim1_e1_why_D_improves.py` - and are now
subcommands; the bodies moved unchanged to `src/oceanpath/aim1/cli/` and each
keeps its own flags, so every previously valid invocation still works with the
subcommand inserted.

    challenge-sets   the E1a restriction panel itself - Results.md Table 1.2
    vs-acomplete     set D against the A-complete comparator - Table 1.3
    why-d            why D beats A-complete, analyses A-D - Table 1.5

WHY THESE THREE AND NOT E1a-S. The composition-standardized analysis answers a
different question with a different estimator (direct standardization over
common strata, not restriction), and the study reports it as its own experiment,
E1a-S. It stays a separate launcher, `aim1_s_composition_standardized.py`.

Usage:
    python aim1_challenge_sets.py challenge-sets report [--model 1a] [--seed 42]
    python aim1_challenge_sets.py vs-acomplete [--condition pb_cap8192] [--compare]
    python aim1_challenge_sets.py why-d [--arm 1a_pb_cap8192] [--encoder univ1]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from oceanpath.aim1.cli import e1a_acomplete, e1a_challenge, e1a_whyd  # noqa: E402

SUBCOMMANDS = {
    "challenge-sets": (e1a_challenge, "E1a restriction panel (Table 1.2)"),
    "vs-acomplete": (e1a_acomplete, "D vs A-complete (Table 1.3)"),
    "why-d": (e1a_whyd, "why D beats A-complete, analyses A-D (Table 1.5)"),
}


def _usage(stream=sys.stdout) -> None:
    print(__doc__.strip(), file=stream)
    print("\nsubcommands:", file=stream)
    for name, (_, blurb) in SUBCOMMANDS.items():
        print(f"  {name:16s} {blurb}", file=stream)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _usage()
        raise SystemExit(0)
    name = sys.argv[1]
    if name not in SUBCOMMANDS:
        print(f"unknown subcommand {name!r}\n", file=sys.stderr)
        _usage(sys.stderr)
        raise SystemExit(2)
    module, _ = SUBCOMMANDS[name]
    # Hand the sub-parser a clean argv so its own flags parse exactly as before.
    sys.argv = [f"{Path(sys.argv[0]).name} {name}", *sys.argv[2:]]
    module.main()


if __name__ == "__main__":
    main()
