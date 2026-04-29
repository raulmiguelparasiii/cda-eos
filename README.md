# CDA-EOS

Curvature-Driven Association Equation of State (CDA-EOS) for liquid water.

This repository contains a framework paper and preliminary numerical prototype for a Helmholtz-energy construction that combines:

1. the IAPWS-95 Helmholtz reference equation of state,
2. Ruppeiner thermodynamic curvature as a correlation-strength field,
3. a SAFT-style four-site hydrogen-bond association term,
4. a curvature-regulated association strength inside a single total Helmholtz free-energy structure.

The current paper presents CDA-EOS as a proposed EOS construction and preliminary numerical proof-of-concept, not as a final reference-quality water EOS.

## Repository layout

```text
paper/
  main.tex                    LaTeX source for the paper
  CDA-EOS Paras.pdf           compiled paper PDF
  LLM_USAGE.md                disclosure of AI-assisted drafting and numerical work
  figures/                    figures used by the paper

scripts/
  cda_eos_numerical_prototype.py
                              reproducible CoolProp-based numerical prototype

notebooks/
  CDA_EOS_numerical_prototype_colab.ipynb
                              Colab notebook wrapper for running the prototype

requirements.txt              Python dependencies for the numerical prototype
```

## Numerical prototype

The prototype uses CoolProp as a practical direct-property backend for real water fields. It computes base fields, a direct-metric Ruppeiner curvature field, a curvature-derived network factor, a SAFT-style association correction, a pressure-level TMD diagnostic, and a prototype heat-capacity consistency check.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python scripts/cda_eos_numerical_prototype.py --out outputs_cda_eos
```

The script writes JSON reports, CSV tables, NumPy fields, and PNG plots under the output folder.

## Status

The current numerical checks are preliminary. They test basic consistency and stability on a finite grid, including:

- base CoolProp/IAPWS-layer liquid-side fields,
- direct-metric curvature mapping,
- nonzero CDA association-pressure correction,
- mechanical stability checks,
- TMD diagnostic shift,
- prototype \(C_p\) positivity and derivative-identity consistency.

A full fitted benchmark validation against accepted water property datasets remains future work.
