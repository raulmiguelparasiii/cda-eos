#!/usr/bin/env python3
"""
CDA-EOS numerical prototype.

This script reproduces the preliminary numerical checks used in the CDA-EOS
paper. It is intentionally a prototype, not a reference EOS implementation.

Outputs:
- base_fields_grid.npz / base_stability_report.json
- metric_curvature_fields.npz / metric_curvature_report.json
- association_sweep.csv / association_sweep_summary.json
- cp_validation_fields.npz / cp_validation_report.json
- PNG heatmaps and diagnostic plots

Run:
    python scripts/cda_eos_numerical_prototype.py --out outputs_cda_eos
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from CoolProp.CoolProp import PropsSI, PhaseSI

R_GAS = 8.31446261815324       # J/(mol K)
M_WATER = 0.018015268          # kg/mol
M_SITES = 4


def json_write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def heatmap(path: Path, Ts: np.ndarray, rhos: np.ndarray, Z: np.ndarray,
            title: str, cbar_label: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")
    plt.imshow(
        np.ma.masked_invalid(Z),
        origin="lower",
        aspect="auto",
        extent=[float(rhos[0]), float(rhos[-1]), float(Ts[0]), float(Ts[-1])],
        cmap=cmap,
    )
    cb = plt.colorbar()
    if cbar_label:
        cb.set_label(cbar_label)
    plt.xlabel(r"$\rho_{\rm mass}$ [kg m$^{-3}$]")
    plt.ylabel("T [K]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def is_bad_state(T: float, rho_mass: float) -> bool:
    """Conservative single-phase liquid-side mask."""
    try:
        ph = str(PhaseSI("T", float(T), "Dmass", float(rho_mass), "Water")).lower()
        if any(s in ph for s in ["two", "twophase", "gas", "vapor"]):
            return True
    except Exception:
        pass

    try:
        rhoL = float(PropsSI("Dmass", "T", float(T), "Q", 0, "Water"))
        if rho_mass < rhoL - 1e-4 * rhoL:
            return True
    except Exception:
        pass
    return False


def build_base_grid(Ts: np.ndarray, rhos: np.ndarray, out: Path) -> Dict[str, np.ndarray]:
    shape = (len(Ts), len(rhos))
    fields = {k: np.full(shape, np.nan) for k in [
        "p", "cp", "cv", "alpha_p", "kappa_T", "dp_drho_T", "s", "u"
    ]}
    invalid = np.zeros(shape, dtype=bool)

    for i, T in enumerate(Ts):
        for j, rho in enumerate(rhos):
            try:
                if is_bad_state(T, rho):
                    invalid[i, j] = True
                    continue
                fields["p"][i, j] = float(PropsSI("P", "T", float(T), "Dmass", float(rho), "Water"))
                fields["cp"][i, j] = float(PropsSI("Cpmass", "T", float(T), "Dmass", float(rho), "Water"))
                fields["cv"][i, j] = float(PropsSI("Cvmass", "T", float(T), "Dmass", float(rho), "Water"))
                fields["s"][i, j] = float(PropsSI("Smass", "T", float(T), "Dmass", float(rho), "Water"))
                fields["u"][i, j] = float(PropsSI("Umass", "T", float(T), "Dmass", float(rho), "Water"))
                fields["alpha_p"][i, j] = float(PropsSI("ISOBARIC_EXPANSION_COEFFICIENT", "T", float(T), "Dmass", float(rho), "Water"))
                fields["kappa_T"][i, j] = float(PropsSI("ISOTHERMAL_COMPRESSIBILITY", "T", float(T), "Dmass", float(rho), "Water"))
                fields["dp_drho_T"][i, j] = float(PropsSI("d(P)/d(Dmass)|T", "T", float(T), "Dmass", float(rho), "Water"))
                if not all(np.isfinite(fields[k][i, j]) for k in fields):
                    invalid[i, j] = True
            except Exception:
                invalid[i, j] = True

    valid = ~invalid
    def frac_bad(cond):
        denom = max(int(np.sum(valid)), 1)
        return float(np.sum(cond & valid) / denom)

    report = {
        "valid_cells": int(np.sum(valid)),
        "invalid_cells": int(np.sum(invalid)),
        "frac_invalid": float(np.mean(invalid)),
        "min_dp_drho_T_valid": float(np.nanmin(fields["dp_drho_T"][valid])),
        "min_kappa_T_valid": float(np.nanmin(fields["kappa_T"][valid])),
        "min_cp_valid": float(np.nanmin(fields["cp"][valid])),
        "frac_dp_drho_T_le_0_valid": frac_bad(fields["dp_drho_T"] <= 0),
        "frac_kappa_T_le_0_valid": frac_bad(fields["kappa_T"] <= 0),
        "frac_cp_le_0_valid": frac_bad(fields["cp"] <= 0),
    }

    fields["invalid"] = invalid
    np.savez_compressed(out / "base_fields_grid.npz", Ts=Ts, rhos=rhos, **fields)
    json_write(out / "base_stability_report.json", report)

    heatmap(out / "base_pressure.png", Ts, rhos, fields["p"], "Base pressure", "Pa")
    heatmap(out / "base_cp.png", Ts, rhos, fields["cp"], "Base Cp", "J kg$^{-1}$ K$^{-1}$")
    heatmap(out / "base_alpha_p.png", Ts, rhos, fields["alpha_p"], "Base alpha_p", "K$^{-1}$")
    return fields


def tmd_from_alpha(Ts: np.ndarray, rhos: np.ndarray, alpha: np.ndarray, invalid: np.ndarray) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for j, rho in enumerate(rhos):
        a = alpha[:, j]
        bad = invalid[:, j] | ~np.isfinite(a)
        for i in range(len(Ts) - 1):
            if bad[i] or bad[i + 1]:
                continue
            if a[i] == 0:
                out[float(rho)] = float(Ts[i])
                break
            if a[i] * a[i + 1] < 0:
                T0, T1 = Ts[i], Ts[i + 1]
                out[float(rho)] = float(T0 - a[i] * (T1 - T0) / (a[i + 1] - a[i]))
                break
    return out


def direct_metric_curvature(Ts: np.ndarray, rhos: np.ndarray, out: Path) -> Tuple[np.ndarray, np.ndarray]:
    nT, nrho = len(Ts), len(rhos)
    vms = M_WATER / rhos
    gTT = np.full((nT, nrho), np.nan)
    gvv = np.full((nT, nrho), np.nan)
    invalid = np.zeros((nT, nrho), dtype=bool)

    for i, T in enumerate(Ts):
        for j, rho in enumerate(rhos):
            try:
                if is_bad_state(T, rho):
                    invalid[i, j] = True
                    continue
                cv_m = float(PropsSI("Cvmass", "T", float(T), "Dmass", float(rho), "Water")) * M_WATER
                dp_drho_mass = float(PropsSI("d(P)/d(Dmass)|T", "T", float(T), "Dmass", float(rho), "Water"))
                v_m = M_WATER / rho
                dp_dv_m = dp_drho_mass * (-M_WATER / (v_m * v_m))
                gTT[i, j] = cv_m / (R_GAS * T * T)
                gvv[i, j] = -dp_dv_m / (R_GAS * T)
                if not (np.isfinite(gTT[i, j]) and np.isfinite(gvv[i, j]) and gTT[i, j] > 0 and gvv[i, j] > 0):
                    invalid[i, j] = True
            except Exception:
                invalid[i, j] = True

    valid = (~invalid) & np.isfinite(gTT) & np.isfinite(gvv) & (gTT > 0) & (gvv > 0)
    A, B = gTT, gvv
    dA_dT, dA_dv = np.gradient(A, Ts, vms, edge_order=2)
    dB_dT, dB_dv = np.gradient(B, Ts, vms, edge_order=2)

    invA, invB = 1.0 / A, 1.0 / B
    Gamma = np.full((2, 2, 2, nT, nrho), np.nan)
    Gamma[0, 0, 0] = 0.5 * invA * dA_dT
    Gamma[0, 0, 1] = Gamma[0, 1, 0] = 0.5 * invA * dA_dv
    Gamma[0, 1, 1] = -0.5 * invA * dB_dT
    Gamma[1, 0, 0] = -0.5 * invB * dA_dv
    Gamma[1, 0, 1] = Gamma[1, 1, 0] = 0.5 * invB * dB_dT
    Gamma[1, 1, 1] = 0.5 * invB * dB_dv

    dG = {}
    for k in range(2):
        for i in range(2):
            for j in range(2):
                dT, dv = np.gradient(Gamma[k, i, j], Ts, vms, edge_order=2)
                dG[(k, i, j, 0)] = dT
                dG[(k, i, j, 1)] = dv

    Ric = np.full((2, 2, nT, nrho), np.nan)
    for i in range(2):
        for j in range(2):
            term1 = sum(dG[(k, i, j, k)] for k in range(2))
            term2 = sum(dG[(k, i, k, j)] for k in range(2))
            term3 = np.zeros((nT, nrho))
            term4 = np.zeros((nT, nrho))
            for k in range(2):
                for l in range(2):
                    term3 += Gamma[k, i, j] * Gamma[l, k, l]
                    term4 += Gamma[l, i, k] * Gamma[k, j, l]
            Ric[i, j] = term1 - term2 + term3 - term4

    Rfield = invA * Ric[0, 0] + invB * Ric[1, 1]

    robust = valid.copy()
    robust[0, :] = robust[-1, :] = False
    robust[:, 0] = robust[:, -1] = False
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            shifted = np.roll(np.roll(invalid, di, axis=0), dj, axis=1)
            if di > 0: shifted[:di, :] = False
            if di < 0: shifted[di:, :] = False
            if dj > 0: shifted[:, :dj] = False
            if dj < 0: shifted[:, dj:] = False
            robust &= ~shifted

    Rfield = np.where(robust & np.isfinite(Rfield), Rfield, np.nan)
    vals = np.abs(Rfield[np.isfinite(Rfield)])
    R0 = float(np.nanmedian(vals)) if vals.size else 1e-6
    if not np.isfinite(R0) or R0 <= 0:
        R0 = 1e-6
    Reps = max(1e-6 * R0, 1e-30)
    nfield = 1.0 - np.exp(-np.sqrt(Rfield * Rfield + Reps * Reps) / R0)

    report = {
        "method": "direct diagonal Ruppeiner metric in (T, v_m)",
        "valid_metric_cells": int(np.sum(valid)),
        "robust_R_cells": int(np.sum(np.isfinite(Rfield))),
        "frac_robust_R": float(np.mean(np.isfinite(Rfield))),
        "R_min": float(np.nanmin(Rfield)),
        "R_max": float(np.nanmax(Rfield)),
        "R_median": float(np.nanmedian(Rfield)),
        "absR_median_R0": R0,
        "R_epsilon": Reps,
        "n_min": float(np.nanmin(nfield)),
        "n_max": float(np.nanmax(nfield)),
        "n_median": float(np.nanmedian(nfield)),
    }
    np.savez_compressed(out / "metric_curvature_fields.npz", Ts=Ts, rho_mass=rhos, v_molar=vms,
                        gTT=gTT, gvv=gvv, valid=valid, robust_valid=robust, R=Rfield, n=nfield)
    json_write(out / "metric_curvature_report.json", report)
    heatmap(out / "metric_R.png", Ts, rhos, Rfield, "Direct-metric Ruppeiner curvature", "m$^3$/mol")
    heatmap(out / "metric_n.png", Ts, rhos, nfield, "Network factor n")
    return Rfield, nfield


def interpolate_n(Ts: np.ndarray, rhos: np.ndarray, Ts_c: np.ndarray, rhos_c: np.ndarray, n_c: np.ndarray) -> np.ndarray:
    TT, RR = np.meshgrid(Ts, rhos, indexing="ij")
    pts, vals = [], []
    for i, T in enumerate(Ts_c):
        for j, rho in enumerate(rhos_c):
            if np.isfinite(n_c[i, j]):
                pts.append((float(T), float(rho)))
                vals.append(float(n_c[i, j]))
    pts = np.array(pts)
    vals = np.array(vals)
    n_lin = griddata(pts, vals, (TT, RR), method="linear")
    n_near = griddata(pts, vals, (TT, RR), method="nearest")
    n_grid = np.where(np.isfinite(n_lin), n_lin, n_near)
    return np.clip(gaussian_filter(n_grid, sigma=1.5), 0.0, 1.0)


def association_and_cp(Ts: np.ndarray, rhos: np.ndarray, base: Dict[str, np.ndarray], n_grid: np.ndarray, out: Path) -> None:
    rho_mol = rhos / M_WATER
    base_tmd = tmd_from_alpha(Ts, rhos, base["alpha_p"], base["invalid"])

    def trial(K0: float, epsK: float, alpha0: float, lam: float):
        Tmat, rhomat = np.meshgrid(Ts, rho_mol, indexing="ij")
        Delta0 = K0 * (np.exp(epsK / Tmat) - 1.0)
        Delta = Delta0 * (alpha0 + (1.0 - alpha0) * n_grid)
        A = rhomat * M_SITES * Delta
        X = np.where(A < 1e-12, 1.0 - A, (-1.0 + np.sqrt(1.0 + 4.0 * A)) / (2.0 * A))
        X = np.clip(X, 1e-12, 1.0)
        a_assoc = R_GAS * Tmat * M_SITES * (np.log(X) - X / 2.0 + 0.5)
        da_drho = np.gradient(a_assoc, rho_mol, axis=1, edge_order=2)
        p_assoc = rho_mol[None, :] ** 2 * da_drho
        P_total = base["p"] + lam * p_assoc
        dP_dT = np.gradient(P_total, Ts, axis=0, edge_order=2)
        dP_drho = np.gradient(P_total, rho_mol, axis=1, edge_order=2)
        alpha_total = (1.0 / rho_mol[None, :]) * dP_dT / dP_drho
        kappa_total = 1.0 / (rho_mol[None, :] * dP_drho)
        valid = (~base["invalid"]) & np.isfinite(P_total) & np.isfinite(alpha_total) & np.isfinite(kappa_total) & (dP_drho > 0) & (kappa_total > 0)
        denom = max(int(np.sum((~base["invalid"]) & np.isfinite(P_total))), 1)
        bad = ((dP_drho <= 0) | (kappa_total <= 0)) & (~base["invalid"]) & np.isfinite(P_total)
        tmd_total = tmd_from_alpha(Ts, rhos, alpha_total, base["invalid"] | (~valid))
        common = sorted(set(base_tmd).intersection(tmd_total))
        shifts = [tmd_total[r] - base_tmd[r] for r in common]
        p_corr = lam * p_assoc
        result = {
            "K0_m3_per_mol": K0, "epsK_K": epsK, "alpha0": alpha0, "lambda": lam,
            "valid_cells": int(np.sum(valid)),
            "bad_mechanical_cells": int(np.sum(bad)),
            "bad_mechanical_frac_valid": float(np.sum(bad) / denom),
            "min_dP_drho_molar_valid": float(np.nanmin(dP_drho[valid])),
            "min_kappa_total_valid": float(np.nanmin(kappa_total[valid])),
            "p_correction_median_MPa": float(np.nanmedian(p_corr[valid]) / 1e6),
            "p_correction_p01_MPa": float(np.nanquantile(p_corr[valid], 0.01) / 1e6),
            "p_correction_p99_MPa": float(np.nanquantile(p_corr[valid], 0.99) / 1e6),
            "tmd_points_base": len(base_tmd), "tmd_points_total": len(tmd_total), "tmd_common_points": len(common),
            "tmd_shift_mean_K": float(np.mean(shifts)) if shifts else float("nan"),
            "tmd_shift_min_K": float(np.min(shifts)) if shifts else float("nan"),
            "tmd_shift_max_K": float(np.max(shifts)) if shifts else float("nan"),
        }
        fields = dict(Delta=Delta, X=X, a_assoc=a_assoc, p_assoc=p_assoc, P_total=P_total,
                      alpha_total=alpha_total, kappa_total=kappa_total, dP_drho=dP_drho,
                      valid=valid, tmd_total=tmd_total)
        return result, fields

    rows, cache = [], {}
    for K0 in [1e-10, 3e-10, 1e-9, 3e-9]:
        for epsK in [1000.0, 1400.0, 1800.0, 2200.0]:
            for alpha0 in [0.05, 0.10, 0.25, 0.50]:
                for lam in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]:
                    r, f = trial(K0, epsK, alpha0, lam)
                    rows.append(r)
                    cache[(K0, epsK, alpha0, lam)] = f

    with (out / "association_sweep.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    stable = [r for r in rows if r["bad_mechanical_frac_valid"] == 0 and r["tmd_common_points"] >= 5
              and abs(r["p_correction_p01_MPa"]) < 100 and abs(r["p_correction_p99_MPa"]) < 100]
    def score(r):
        return abs(r["tmd_shift_mean_K"]) - 0.01 * (abs(r["p_correction_p01_MPa"]) + abs(r["p_correction_p99_MPa"]))
    top = sorted(stable, key=score, reverse=True)[:10]
    chosen = top[0]
    chosen_key = (chosen["K0_m3_per_mol"], chosen["epsK_K"], chosen["alpha0"], chosen["lambda"])
    cf = cache[chosen_key]

    summary = {"n_grid_min": float(np.nanmin(n_grid)), "n_grid_max": float(np.nanmax(n_grid)),
               "n_grid_median": float(np.nanmedian(n_grid)), "sweep_count": len(rows),
               "stable_candidate_count": len(stable), "top_candidates": top, "chosen_candidate": chosen}
    json_write(out / "association_sweep_summary.json", summary)

    # Cp validation for chosen candidate.
    lam = chosen["lambda"]
    Tgrid = Ts[:, None]
    rho_mol_grid = rho_mol[None, :]
    h_base_mol = (base["u"] + base["p"] / rhos[None, :]) * M_WATER
    da_dT = np.gradient(cf["a_assoc"], Ts, axis=0, edge_order=2)
    s_assoc = -da_dT
    h_corr = lam * (cf["a_assoc"] + Tgrid * s_assoc + cf["p_assoc"] / rho_mol_grid)
    h_total = h_base_mol + h_corr
    h_T = np.gradient(h_total, Ts, axis=0, edge_order=2)
    h_rho = np.gradient(h_total, rho_mol, axis=1, edge_order=2)
    p_T = np.gradient(cf["P_total"], Ts, axis=0, edge_order=2)
    p_rho = np.gradient(cf["P_total"], rho_mol, axis=1, edge_order=2)
    Cp_total_mass = (h_T - h_rho * p_T / p_rho) / M_WATER

    hb_T = np.gradient(h_base_mol, Ts, axis=0, edge_order=2)
    hb_rho = np.gradient(h_base_mol, rho_mol, axis=1, edge_order=2)
    pb_T = np.gradient(base["p"], Ts, axis=0, edge_order=2)
    pb_rho = np.gradient(base["p"], rho_mol, axis=1, edge_order=2)
    Cp_base_identity = (hb_T - hb_rho * pb_T / pb_rho) / M_WATER
    base_err = (Cp_base_identity - base["cp"]) / base["cp"]

    robust = cf["valid"] & (~base["invalid"]) & np.isfinite(Cp_total_mass) & np.isfinite(base["cp"]) & (p_rho > 0)
    robust[0, :] = robust[-1, :] = False
    robust[:, 0] = robust[:, -1] = False
    def stat(arr):
        vals = arr[robust & np.isfinite(arr)]
        return {"min": float(np.min(vals)), "p01": float(np.quantile(vals, .01)),
                "median": float(np.median(vals)), "p99": float(np.quantile(vals, .99)), "max": float(np.max(vals))}
    cp_report = {
        "method": "Cp prototype from h_total and p_total finite derivatives",
        "chosen_candidate": chosen,
        "robust_cp_cells": int(np.sum(robust)),
        "robust_cp_fraction": float(np.mean(robust)),
        "Cp_total_mass_stats_J_per_kgK": stat(Cp_total_mass),
        "Cp_base_mass_stats_J_per_kgK": stat(base["cp"]),
        "Cp_delta_mass_stats_J_per_kgK": stat(Cp_total_mass - base["cp"]),
        "Cp_ratio_stats": stat(Cp_total_mass / base["cp"]),
        "Cp_relative_change_stats": stat((Cp_total_mass - base["cp"]) / base["cp"]),
        "base_Cp_identity_relative_error_stats": stat(base_err),
        "Cp_total_le_0_cells": int(np.sum((Cp_total_mass <= 0) & robust)),
        "Cp_total_le_0_frac_robust": float(np.sum((Cp_total_mass <= 0) & robust) / max(int(np.sum(robust)), 1)),
        "p_rho_le_0_cells": int(np.sum((p_rho <= 0) & robust)),
        "p_rho_le_0_frac_robust": float(np.sum((p_rho <= 0) & robust) / max(int(np.sum(robust)), 1)),
    }
    json_write(out / "cp_validation_report.json", cp_report)

    np.savez_compressed(out / "association_chosen_fields.npz", Ts=Ts, rho_mass=rhos, rho_mol=rho_mol, n_grid=n_grid,
                        P_base=base["p"], alpha_base=base["alpha_p"], kappa_base=base["kappa_T"],
                        invalid_base=base["invalid"], **cf)
    np.savez_compressed(out / "cp_validation_fields.npz", Ts=Ts, rho_mass=rhos, rho_mol=rho_mol, robust=robust,
                        Cp_total_mass=Cp_total_mass, cp_base_mass=base["cp"], Cp_delta_mass=Cp_total_mass-base["cp"],
                        Cp_ratio=Cp_total_mass/base["cp"], Cp_rel_change=(Cp_total_mass-base["cp"])/base["cp"],
                        base_identity_rel_error=base_err, h_corr_mol=h_corr, h_total_mol=h_total, p_rho=p_rho, p_T=p_T)

    heatmap(out / "assoc_pressure_correction_MPa.png", Ts, rhos, np.where(cf["valid"], chosen["lambda"]*cf["p_assoc"]/1e6, np.nan), "CDA pressure correction", "MPa")
    heatmap(out / "cp_total.png", Ts, rhos, np.where(robust, Cp_total_mass, np.nan), "CDA prototype Cp_total", "J kg$^{-1}$ K$^{-1}$")
    heatmap(out / "cp_ratio.png", Ts, rhos, np.where(robust, Cp_total_mass/base["cp"], np.nan), "CDA prototype Cp_total / Cp_base")

    # TMD comparison plot.
    tmd_total = cf["tmd_total"]
    plt.figure(figsize=(7, 5))
    if base_tmd:
        br = np.array(sorted(base_tmd)); bt = np.array([base_tmd[x] for x in br])
        plt.plot(br, bt, label="Base alpha_p=0")
    if tmd_total:
        cr = np.array(sorted(tmd_total)); ct = np.array([tmd_total[x] for x in cr])
        plt.plot(cr, ct, label="CDA corrected alpha_p=0")
    plt.xlabel(r"$\rho_{\rm mass}$ [kg m$^{-3}$]"); plt.ylabel("TMD diagnostic T [K]")
    plt.title("TMD diagnostic comparison"); plt.legend(); plt.tight_layout()
    plt.savefig(out / "tmd_comparison.png", dpi=180); plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs_cda_eos")
    parser.add_argument("--T-min", type=float, default=250.0)
    parser.add_argument("--T-max", type=float, default=350.0)
    parser.add_argument("--n-T", type=int, default=101)
    parser.add_argument("--rho-min", type=float, default=900.0)
    parser.add_argument("--rho-max", type=float, default=1100.0)
    parser.add_argument("--n-rho", type=int, default=81)
    parser.add_argument("--curv-n-T", type=int, default=61)
    parser.add_argument("--curv-n-rho", type=int, default=51)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Ts = np.linspace(args.T_min, args.T_max, args.n_T)
    rhos = np.linspace(args.rho_min, args.rho_max, args.n_rho)
    base = build_base_grid(Ts, rhos, out)

    Ts_c = np.linspace(args.T_min, args.T_max, args.curv_n_T)
    rhos_c = np.linspace(args.rho_min, args.rho_max, args.curv_n_rho)
    _, n_c = direct_metric_curvature(Ts_c, rhos_c, out)
    n_grid = interpolate_n(Ts, rhos, Ts_c, rhos_c, n_c)
    heatmap(out / "network_factor_interpolated.png", Ts, rhos, n_grid, "Interpolated network factor n")

    association_and_cp(Ts, rhos, base, n_grid, out)
    print(f"Wrote CDA-EOS numerical prototype outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
