#!/usr/bin/env python3
"""Generate matched per-experiment configs for the CORRECTED ablation (post code-review).
All share model/data/training; only cond/proj/cond_mode + loss switches + out dir differ.

Corrected per review:
- lambda_time = 0 everywhere (L_time was unfair across projectors / a null-space shortcut; redesign later).
- SCM ablation = ONE SCPGA backbone (proj=identity), vary cond_mode point<->secant (NOT base-vs-scpga arch swap).
- SHMM ablation = same rank-Kg projectors (dct/v1/v2) under cond_mode=secant_full (NOT identity full-rank).
"""
import os
import copy
import yaml

BASE = os.path.expanduser("~/ScoliCMF/configs/sc_pixel.yaml")
OUT = os.path.expanduser("~/ScoliCMF/configs")

# name, cond, proj, cond_mode, lambda_st, lambda_comp, lambda_roll, lambda_time
EXPS = [
    # --- Bridge + composition ladder (cond=base; cond_mode unused) ---
    ("s2_base",    "base",  None,       "static",      0.0,  0.0, 0.0, 0.0),
    ("s3_st",      "base",  None,       "static",      0.05, 0.0, 0.0, 0.0),
    ("s4_comp",    "base",  None,       "static",      0.05, 0.1, 0.1, 0.0),
    # --- SCM ablation: same SCPGA backbone, proj=identity, vary cond_mode (fair point-vs-secant) ---
    ("scm_static", "scpga", "identity", "static",      0.05, 0.1, 0.1, 0.0),
    ("scm_point",  "scpga", "identity", "point",       0.05, 0.1, 0.1, 0.0),
    ("scm_secant", "scpga", "identity", "secant_full", 0.05, 0.1, 0.1, 0.0),
    # --- SHMM ablation: cond_mode=secant_full, SAME rank-Kg projectors (patient-specific vs generic) ---
    ("shmm_dct",   "scpga", "dct",      "secant_full", 0.05, 0.1, 0.1, 0.0),
    ("shmm_v1",    "scpga", "v1",       "secant_full", 0.05, 0.1, 0.1, 0.0),
    ("shmm_v2",    "scpga", "v2",       "secant_full", 0.05, 0.1, 0.1, 0.0),
]


def main():
    base = yaml.safe_load(open(BASE))
    for name, cond, proj, cmode, st, comp, roll, tm in EXPS:
        c = copy.deepcopy(base)
        c["model"]["cond"] = cond
        c["model"]["dyn_off"] = False
        c["model"]["cond_mode"] = cmode
        if proj is not None:
            c["model"]["proj"] = proj
        c["meanflow"].update(lambda_st=st, lambda_comp=comp, lambda_roll=roll, lambda_time=tm)
        c["meanflow"]["lambda_tokdiv"] = 0.1 if cond == "scpga" else 0.0
        c["project"]["image_save_path"] = f"runs/{name}/images"
        c["project"]["checkpoint_path"] = f"runs/{name}/ckpts"
        c["project"]["log_file"] = f"runs/{name}/log.txt"
        c["experiment"] = name
        with open(os.path.join(OUT, f"{name}.yaml"), "w") as f:
            f.write(f"# {name}: cond={cond} proj={proj} cond_mode={cmode} | "
                    f"ST={st} comp={comp} roll={roll} time={tm}\n")
            yaml.safe_dump(c, f, sort_keys=False, allow_unicode=True)
        print(f"wrote {name} (cond={cond} proj={proj} cond_mode={cmode})")


if __name__ == "__main__":
    main()
