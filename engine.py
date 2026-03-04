# ============================================================
# engine.py  —  BioSmart v5.8
# Physics engine: blend, efficiency factors, feature vector,
# warnings & recommendations.
#
# v5 fixes (P1–P5):
#   P1 — OLR: batch uses VS-density metric, continuous uses OLR
#   P2 — Water input: water_added_kg dilutes TS% correctly
#   P3 — _e_temp_base: corrected transition zone & high-T curve
#   P4 — _e_temp_fluct: added hyper-thermophilic branch (T>58C)
#   P5 — _e_mode: continuous CSTR gets +15% yield bonus
#
# v5.7 fix:
#   P6 — build_feature_vector: added eff_mode + overall_efficiency_adjusted
#        so XGBoost Stage 3 can directly see the continuous yield bonus.
#
# v5.8 fixes:
#   P7 — _e_OLR continuous branch: low OLR (<0.5) was wrongly
#        returning 0.72 (severe penalty). Literature shows low OLR
#        in CSTR gives the HIGHEST specific yield (more residence time).
#        Fixed: OLR<0.3->0.90, 0.3-0.5->0.92, 0.5-1.0->0.93
#   P8 — generate_warnings_and_recs: OLR warning was mode-blind.
#        Batch was shown "under-loaded + increase feed mass" even
#        when VS_density was 68 kg/m3 (digester too SMALL, not too large).
#        Fixed: batch uses VS_density check; continuous uses OLR check.
# ============================================================


import numpy as np


# ============================================================
def blend_feedstocks(components, db):
    """
    Compute mass-weighted blend of all feedstock properties.
    components : list of (feedstock_name, mass_kg)
    db         : dict from feedstock_database.csv
    Returns a dict of blended parameters.
    """
    total_mass = sum(m for _, m in components)
    total_TS   = sum(m * db[f]["TS_pct"]/100             for f, m in components)
    total_VS   = sum(m * db[f]["TS_pct"]/100 * db[f]["VS_TS"]/100 for f, m in components)

    def _blend_vs(key):
        """VS-weighted average of a % property."""
        return (
            sum(m * db[f]["TS_pct"]/100 * db[f]["VS_TS"]/100 * db[f][key]
                for f, m in components) / total_VS
        ) if total_VS > 0 else 0

    total_C = sum(
        m * db[f]["TS_pct"]/100 * db[f]["VS_TS"]/100 *
        (db[f]["CN"] / (db[f]["CN"] + 1))
        for f, m in components)
    total_N = sum(
        m * db[f]["TS_pct"]/100 * db[f]["VS_TS"]/100 *
        (1 / (db[f]["CN"] + 1))
        for f, m in components)
    CN = total_C / total_N if total_N > 0 else 999

    # ── Biochemical substrate composition (% of blended VS) ────
    fat_pct     = _blend_vs("fat_pct")
    protein_pct = _blend_vs("protein_pct")
    carb_pct    = _blend_vs("carb_pct")
    lignin_pct  = _blend_vs("Lignin_pct")

    # ── Buswell stoichiometric CH4 fraction (0-1 scale) ─────────
    # CH4 = 0.72*fat + 0.60*protein + 0.51*carb + 0.42*lignin
    # Coefficients from Buswell & Mueller (1952), Angelidaki (2004)
    ch4_stoich = (
        0.72 * (fat_pct/100) +
        0.60 * (protein_pct/100) +
        0.51 * (carb_pct/100) +
        0.42 * (lignin_pct/100)
    )

    return dict(
        total_mass_kg       = total_mass,
        total_TS_kg         = total_TS,
        total_VS_kg         = total_VS,
        TS_pct              = (total_TS / total_mass) * 100 if total_mass > 0 else 0,
        VS_TS_pct           = (total_VS / total_TS) * 100   if total_TS  > 0 else 0,
        blended_CN_ratio    = CN,
        blended_lignin_pct  = lignin_pct,
        blended_fat_pct     = fat_pct,
        blended_protein_pct = protein_pct,
        blended_carb_pct    = carb_pct,
        CH4_stoich_index    = ch4_stoich,
        fat_to_carb_ratio   = fat_pct / carb_pct if carb_pct > 0 else 0,
        protein_to_VS       = protein_pct / 100,
        base_yield          = _blend_vs("base_yield"),
        base_CH4            = _blend_vs("base_CH4"),
        H2S_risk            = max(db[f]["H2S_risk"] for f, _ in components),
        NH3_risk            = max(db[f]["NH3_risk"] for f, _ in components),
    )


# ──────────────────────────────────────────────────────────────────────
# EFFICIENCY FUNCTIONS  (v5 — science-corrected)
# ──────────────────────────────────────────────────────────────────────

def _e_CN(CN):
    """
    C/N ratio efficiency.
    Optimal 20-30. Outside range: VFA/ammonia inhibition.
    Literature: Mata-Alvarez 2000, Khalid 2011.
    """
    if   20 <= CN <= 30:  return 1.00
    elif 15 <= CN <  20:  return 0.88
    elif 30 < CN  <= 35:  return 0.90
    elif 35 < CN  <= 45:  return 0.80
    elif 10 <= CN <  15:  return 0.52
    elif 45 < CN  <= 60:  return 0.65
    elif  5 <= CN <  10:  return 0.28
    elif CN > 60:          return 0.40
    else:                  return 0.12


def _e_OLR(OLR, is_continuous, VS_density=None):
    """
    P1 / P7 FIX — Batch vs continuous OLR distinction.

    Continuous: OLR (kg VS/m3/day) controls microbial loading.
      Optimal range 1.0-3.0 kg VS/m3/day.

      P7 FIX — Low OLR in continuous CSTR is NOT a penalty:
        PubMed CSTR study: OLR=1 -> 0.48 L CH4/g VS (highest yield),
                           OLR=15 -> 0.10 L CH4/g VS (lowest yield).
        Scielo: best CSTR performance at OLR=0.7 gVS/L/day.
        Low OLR = more residence time = more complete VS conversion.

    Batch: OLR as a daily rate is MEANINGLESS — all feedstock loads
      at once. Use VS loading density (kg VS/m3 digester) instead.
      Typical wet AD batch: 1-20 kg VS/m3.
      Literature: Ward 2008, Mshandete 2008.
    """
    if not is_continuous:
        # ── Batch: VS loading density metric ─────────────────
        vsd = VS_density if VS_density is not None else 1.0
        if   vsd >= 8.0:   return 1.00   # well-loaded batch
        elif vsd >= 4.0:   return 0.97
        elif vsd >= 1.5:   return 0.93
        elif vsd >= 0.5:   return 0.88
        elif vsd >= 0.2:   return 0.82
        else:              return 0.72   # very dilute batch
    else:
        # ── Continuous: literature-corrected OLR curve ────────
        if   1.0 <= OLR <= 3.0:  return 1.00   # optimal
        elif 3.0 <  OLR <= 4.5:  return 0.92   # slightly overloaded
        elif 4.5 <  OLR <= 6.0:  return 0.72   # overloaded
        elif OLR > 6.0:           return 0.25   # severe overload — VFA crash
        elif 0.5 <= OLR <  1.0:  return 0.93   # slightly low — good yield [Scielo]
        elif 0.3 <= OLR <  0.5:  return 0.92   # low OLR — high specific yield [PubMed]
        else:                     return 0.90   # very low — lightly loaded CSTR,
                                                # VS fully converted, steady-state biology


def _e_temp_base(T):
    """
    v5.6: Smooth bimodal cubic-spline temperature efficiency curve.

    Biology-based zone definitions:
      < 25C  -> FAIL zone        (e_base < 0.35)
      25-28C -> PARTIAL_FAIL zone (0.35 <= e_base < 0.55)
      29-65C -> SUCCESS-capable   (e_base >= 0.55)
      > 65C  -> declining toward FAIL

    Key anchor points:
      T=29C -> 0.58  mesophilic range starts (NITRKL thesis)
      T=38C -> 1.00  mesophilic optimum (Angelidaki 2004)
      T=45C -> 0.65  transition valley (AIMS 2025)
      T=47C -> 0.78  thermophilic onset (Hindawi 2020)
      T=50C -> 0.92  thermophilic rising
      T=55C -> 1.05  thermophilic optimum
      T=65C -> 0.30  near-inhibition threshold
      T=70C -> 0.06  thermal kill zone
    """
    import numpy as _np
    from scipy.interpolate import CubicSpline as _CS
    _T = _np.array([10,15,20,25,28,29,30,35,38,40,42,45,47,50,55,58,60,63,65,70,75], dtype=float)
    _e = _np.array([0.03,0.08,0.20,0.35,0.50,0.58,0.70,0.88,1.00,0.95,0.85,
                    0.65,0.78,0.92,1.05,1.00,0.80,0.55,0.30,0.06,0.04], dtype=float)
    _cs = _CS(_T, _e, bc_type="not-a-knot")
    return float(_np.clip(_cs(float(T)), 0.03, 1.10))


def _e_temp_fluct(T, fluct):
    """
    P4 FIX — Added hyper-thermophilic branch for T > 58C.

    Three distinct regimes:
      Psychrophilic/mesophilic (<50C): relatively tolerant
      Thermophilic (50-58C):           moderately sensitive
      Hyper-thermophilic (>58C):       extremely sensitive

    Literature: McMahon 2001, Lindorfer 2008, Boe 2010.
    """
    if T > 58:
        thresholds = [(0.5, 1.00), (1.0, 0.72), (2.0, 0.28), (3.0, 0.06)]
        fallback   = 0.02
    elif 50 <= T <= 58:
        thresholds = [(1.0, 1.00), (1.5, 0.88), (2.0, 0.72), (3.0, 0.45), (5.0, 0.22)]
        fallback   = 0.08
    elif 35 <= T < 50:
        thresholds = [(1.5, 1.00), (2.0, 0.95), (4.0, 0.90), (6.0, 0.75), (9.0, 0.50)]
        fallback   = 0.35
    else:
        thresholds = [(3.0, 1.00), (6.0, 0.90), (9.0, 0.78), (12.0, 0.62)]
        fallback   = 0.40
    for lim, val in thresholds:
        if fluct <= lim:
            return val
    return fallback


def _e_HRT(HRT, lignin):
    """
    HRT efficiency. High-lignin substrates need longer digestion time.
    min_hrt = 15 + (lignin%/100)*55 days.
    """
    min_hrt = 15 + (lignin / 100) * 55
    if   HRT >= min_hrt * 1.2: return 1.00
    elif HRT >= min_hrt:        return 0.92
    elif HRT >= min_hrt * 0.7:  return 0.58
    elif HRT >= min_hrt * 0.5:  return 0.30
    else:                        return 0.10


def _e_TS(TS):
    """
    P2 — TS% uses water-corrected value (see compute_all_efficiencies).
    Optimal 8-12%. Below 4%: washout risk. Above 15%: mixing failure.
    Literature: Fantozzi 2013, Motte 2013.
    """
    if   8 <= TS <= 12:   return 1.00
    elif 12 < TS <= 15:   return 0.88
    elif 15 < TS <= 20:   return 0.70
    elif TS > 20:          return 0.45
    elif  5 <= TS <  8:   return 0.90
    elif  3 <= TS <  5:   return 0.75
    elif  1 <= TS <  3:   return 0.55
    else:                  return 0.35


def _e_ISR(ISR):
    """
    Inoculum-to-Substrate Ratio efficiency.
    Optimal ISR = 1.0-2.0 (VS basis).
    Literature: PMC11610627 2024, Xiao et al. 2022.
    """
    if   ISR >= 1.0:        return 1.00
    elif 0.5 <= ISR < 1.0:  return 0.78
    elif 0.3 <= ISR < 0.5:  return 0.52
    elif 0.1 <= ISR < 0.3:  return 0.30
    else:                    return 0.10


def _e_pressure(P):
    """
    Headspace pressure efficiency.
    Optimal <= 108 kPa. High backpressure inhibits methanogenesis.
    """
    if   P <= 108:  return 1.00
    elif P <= 120:  return 0.97
    elif P <= 140:  return 0.88
    elif P <= 160:  return 0.73
    elif P <= 180:  return 0.55
    else:            return 0.32


def _e_lignin(lig):
    """
    Lignin recalcitrance penalty. Linear degradation model.
    Literature: Zeeman 1994.
    """
    return max(0.10, 1 - 0.72 * (lig / 100))


def _e_mode(is_continuous):
    """
    P5 FIX — Continuous CSTR mode yield bonus (+15%).
    Continuous operation advantages:
      * Steady-state microbiology (no lag phase)
      * Better substrate-microbe contact
      * More uniform pH and VFA profile
    Literature: Stafford 1980, Wellinger 1992, Kaparaju 2009.
    """
    return 1.15 if is_continuous else 1.00


# ============================================================
# COMPUTE ALL EFFICIENCIES
# ============================================================
def compute_all_efficiencies(blend, params):
    """
    Computes all 9 physics efficiency factors + mode multiplier.

    v5 changes:
      * water_added_kg corrects TS% before _e_TS (P2)
      * _e_OLR receives is_continuous + VS_density (P1)
      * _e_temp_base and _e_temp_fluct use corrected curves (P3, P4)
      * _e_mode applied as separate multiplier, not in geom mean (P5)
    """
    T        = params["operating_temp_C"]
    fl       = params["temp_fluctuation_degC"]
    HRT      = params["HRT_days"]
    vol      = params["digester_volume_m3"]
    ISR      = params["ISR"]
    P        = params["operating_pressure_kPa"]
    is_cont  = params.get("is_continuous", False)
    water_kg = params.get("water_added_kg", 0)

    lig = blend["blended_lignin_pct"]

    # ── P2: Water-corrected TS% ──────────────────────────────
    if water_kg > 0:
        total_slurry_mass = blend["total_mass_kg"] + water_kg
        corrected_TS_pct  = (blend["total_TS_kg"] / total_slurry_mass) * 100
    else:
        corrected_TS_pct  = blend["TS_pct"]

    # ── OLR (continuous) and VS loading density (batch) ─────
    OLR        = (blend["total_VS_kg"] / HRT) / vol
    VS_density = blend["total_VS_kg"] / vol

    # ── 9 core efficiency factors ────────────────────────────
    e_cn  = _e_CN(blend["blended_CN_ratio"])
    e_olr = _e_OLR(OLR, is_cont, VS_density)
    e_tb  = _e_temp_base(T)
    e_tf  = _e_temp_fluct(T, fl)
    e_hrt = _e_HRT(HRT, lig)
    e_ts  = _e_TS(corrected_TS_pct)
    e_isr = _e_ISR(ISR)
    e_p   = _e_pressure(P)
    e_lig = _e_lignin(lig)

    overall = e_cn * e_olr * e_tb * e_tf * e_hrt * e_ts * e_isr * e_p * e_lig
    geom    = overall ** (1 / 9)

    # ── P5: Continuous mode multiplier (applied separately) ──
    e_mode_val       = _e_mode(is_cont)
    overall_adjusted = overall * e_mode_val

    return dict(
        eff_CN=e_cn, eff_OLR=e_olr, eff_temp_base=e_tb,
        eff_temp_fluctuation=e_tf, eff_HRT=e_hrt, eff_TS=e_ts,
        eff_ISR=e_isr, eff_pressure=e_p, eff_lignin=e_lig,
        overall_efficiency=overall,
        geom_mean_efficiency=geom,
        eff_mode=e_mode_val,
        overall_efficiency_adjusted=overall_adjusted,
        OLR=OLR,
        VS_density=VS_density,
        corrected_TS_pct=corrected_TS_pct,
    )


# ============================================================
# BUILD FEATURE VECTOR  (37 features for ML)
# ============================================================
def build_feature_vector(blend, params, effs):
    """
    Builds the complete 37-feature vector for model input.
    v5:   uses corrected_TS_pct from effs (P2 water correction).
    v5.7: adds eff_mode + overall_efficiency_adjusted (P6).
    """
    vol     = params["digester_volume_m3"]
    HRT     = params["HRT_days"]
    is_cont = params.get("is_continuous", False)

    return {
        # ── Raw operating parameters (20) ──────────────────
        "n_feedstock_components":    params.get("n_components", 1),
        "is_continuous":             1 if is_cont else 0,
        "water_added_kg":            params.get("water_added_kg", 0),
        "total_feed_mass_kg":        blend["total_mass_kg"],
        "digester_volume_m3":        vol,
        "HRT_days":                  HRT,
        "operating_temp_C":          params["operating_temp_C"],
        "temp_fluctuation_degC":     params["temp_fluctuation_degC"],
        "operating_pressure_kPa":    params["operating_pressure_kPa"],
        "feed_flow_rate_kg_per_day": blend["total_mass_kg"] / HRT if is_cont else 0,
        "ISR_dimensionless":         params["ISR"],
        "urea_added_kg":             params.get("urea_added_kg", 0),
        "blended_CN_ratio":          blend["blended_CN_ratio"],
        "TS_pct":                    effs["corrected_TS_pct"],
        "VS_TS_pct":                 blend["VS_TS_pct"],
        "total_VS_kg":               blend["total_VS_kg"],
        "blended_lignin_pct":        blend["blended_lignin_pct"],
        "OLR_kg_VS_m3_day":          effs["OLR"],
        "ammonia_risk_score":        blend["NH3_risk"],
        "H2S_risk_score":            blend["H2S_risk"],

        # ── Biochemical substrate composition (4) ───────────
        "blended_fat_pct":           blend["blended_fat_pct"],
        "blended_protein_pct":       blend["blended_protein_pct"],
        "blended_carb_pct":          blend["blended_carb_pct"],
        "CH4_stoich_index":          blend["CH4_stoich_index"],

        # ── Physics efficiency factors (13) ─────────────────
        "phys_eff_CN":               effs["eff_CN"],
        "phys_eff_OLR":              effs["eff_OLR"],
        "phys_eff_temp_base":        effs["eff_temp_base"],
        "phys_eff_temp_fluctuation": effs["eff_temp_fluctuation"],
        "phys_eff_HRT":              effs["eff_HRT"],
        "phys_eff_TS":               effs["eff_TS"],
        "phys_eff_ISR":              effs["eff_ISR"],
        "phys_eff_pressure":         effs["eff_pressure"],
        "phys_eff_lignin":           effs["eff_lignin"],
        "overall_efficiency":        effs["overall_efficiency"],
        "geom_mean_efficiency":      effs["geom_mean_efficiency"],
        # P6: continuous mode yield signal
        "eff_mode":                    effs["eff_mode"],
        "overall_efficiency_adjusted": effs["overall_efficiency_adjusted"],
    }


# ============================================================
# WARNINGS & RECOMMENDATIONS  (v5.8 — mode-aware OLR check)
# ============================================================
def generate_warnings_and_recs(blend, effs, OLR, HRT, temp, corrected_TS_pct=None):
    """
    Physics-based diagnostic engine.
    Returns (warnings_list, recommendations) — both plain-string lists.
    Called by app.py after compute_all_efficiencies().

    v5.8 P8: OLR warning section is now mode-aware.
      Batch mode: uses VS_density — prevents wrong "underloaded" warning
                  when digester is actually too small for the feedstock.
      Continuous:  uses OLR as before.
    """
    warnings_list = []
    recs          = []

    CN  = blend["blended_CN_ratio"]
    TS  = corrected_TS_pct if corrected_TS_pct is not None else blend["TS_pct"]
    lig = blend["blended_lignin_pct"]
    h2s = blend["H2S_risk"]
    nh3 = blend["NH3_risk"]

    # ── C/N Ratio ──────────────────────────────────────────
    if CN < 15:
        warnings_list.append(f"C/N ratio {CN:.1f} is too low — risk of ammonia inhibition.")
        recs.append("Add carbon-rich feedstock (straw, cardboard) to raise C/N to 20-30.")
    elif CN > 35:
        warnings_list.append(f"C/N ratio {CN:.1f} is too high — nitrogen deficiency likely.")
        recs.append("Add nitrogen-rich feedstock (manure, food waste) or supplement with urea.")

    # ── OLR / VS-density  (P8 FIX — mode-aware) ───────────
    # Derive mode from eff_mode: 1.15 = continuous, 1.0 = batch
    _is_cont    = (effs.get("eff_mode", 1.0) > 1.0)
    _vs_density = effs.get("VS_density", None)

    if _is_cont:
        # ── Continuous: OLR check ─────────────────────────────
        if OLR > 4.0:
            warnings_list.append(
                f"OLR {OLR:.2f} kg VS/m³/day exceeds recommended maximum (4.0) "
                f"— risk of VFA overload.")
            recs.append("Reduce feed mass or increase digester volume to lower OLR below 4.0.")
        elif OLR < 0.5:
            warnings_list.append(
                f"OLR {OLR:.2f} kg VS/m³/day is low — lightly loaded continuous digester.")
            recs.append(
                "For higher volumetric productivity, increase feed rate or use a smaller digester.")
    else:
        # ── Batch: VS loading density check ──────────────────
        if _vs_density is not None:
            if _vs_density > 20:
                warnings_list.append(
                    f"VS loading density {_vs_density:.1f} kg VS/m³ is too high for wet AD "
                    f"(practical max ~20 kg VS/m³). Feedstock mass exceeds digester capacity.")
                recs.append(
                    "Increase digester volume or reduce feedstock mass to achieve "
                    "VS density below 20 kg VS/m³.")
            elif _vs_density < 0.3:
                warnings_list.append(
                    f"VS loading density {_vs_density:.1f} kg VS/m³ is very low "
                    f"— batch digester severely under-loaded.")
                recs.append(
                    "Increase feedstock mass or use a smaller digester volume "
                    "to improve batch utilisation.")

    # ── Temperature ────────────────────────────────────────
    if temp < 20:
        warnings_list.append(f"Temperature {temp}°C is below psychrophilic range — very slow kinetics.")
        recs.append("Insulate digester or use a solar heater to reach >=25°C.")
    elif 20 <= temp < 30:
        recs.append("Consider mesophilic operation (35-40°C) for 30-50% higher biogas yield.")
    elif 30 <= temp <= 45:
        pass
    elif 45 < temp <= 60:
        recs.append("Thermophilic operation (50-60°C) — maintain temperature stability within ±2°C. "
                    "Fluctuations > ±2°C can disrupt Methanothermobacter activity.")
        fluct_input = effs.get("fluct_input", None)
        if fluct_input is not None and fluct_input > 2.0:
            warnings_list.append(
                f"Temperature fluctuation (±{fluct_input:.1f}°C) exceeds thermophilic tolerance (±2°C). "
                f"Methanothermobacter activity will be disrupted — expect PARTIAL_FAIL conditions.")
    elif temp > 60:
        warnings_list.append(f"Temperature {temp}°C exceeds thermophilic range — microbial die-off risk.")
        recs.append("Reduce operating temperature to 50-60°C for thermophilic or 35-40°C for mesophilic.")

    # ── HRT ────────────────────────────────────────────────
    if HRT < 10:
        warnings_list.append(f"HRT {HRT} days is too short — washout of slow-growing methanogens likely.")
        recs.append("Increase HRT to at least 15-20 days to retain methanogenic biomass.")
    elif HRT > 60:
        warnings_list.append(f"HRT {HRT} days is very long — digester may be oversized.")
        recs.append("Consider reducing HRT to 20-40 days to free capacity without yield loss.")

    # ── TS% ────────────────────────────────────────────────
    if TS > 15:
        warnings_list.append(f"Total solids {TS:.1f}% is high — pumping and mixing issues expected.")
        recs.append("Dilute feed with water or effluent recycling to bring TS% below 12%.")
    elif TS < 2:
        warnings_list.append(f"Total solids {TS:.1f}% is very low — poor volumetric efficiency.")
        recs.append("Increase feedstock concentration to improve volumetric biogas yield.")

    # ── ISR ────────────────────────────────────────────────
    _isr_val = effs.get("ISR_input", None)
    if _isr_val is not None:
        if _isr_val < 0.3:
            warnings_list.append(
                f"Critical ISR ({_isr_val:.2f}) — severe VFA accumulation and "
                f"acidification almost certain. Methanogenesis will likely collapse.")
            recs.append(
                "Drastically increase inoculum. Target ISR >= 1.0 (VS basis). "
                "Pre-seed the digester with active digestate.")
        elif _isr_val < 1.0:
            warnings_list.append(
                f"Low ISR ({_isr_val:.2f}) — below recommended minimum of 1.0. "
                f"Risk of VFA accumulation reducing methane yield by ~20%.")
            recs.append(
                "Increase inoculum to achieve ISR >= 1.0 (VS basis) "
                "for stable methanogenesis and to prevent VFA build-up.")

    # ── Lignin ─────────────────────────────────────────────
    if lig > 20:
        warnings_list.append(f"Blended lignin {lig:.1f}% is high — biodegradability severely limited.")
        recs.append("Pre-treat lignocellulosic feedstocks (steam explosion, NaOH soaking).")

    # ── H2S Risk ───────────────────────────────────────────
    if h2s >= 3:
        warnings_list.append("High H2S risk feedstock detected — risk of pipeline corrosion and engine damage.")
        recs.append("Add iron dosing (FeCl2) or use biogas scrubbing to control H2S below 200 ppm.")
    elif h2s == 2:
        recs.append("Moderate H2S risk — monitor gas quality and consider iron dosing if required.")

    # ── Ammonia Risk ───────────────────────────────────────
    if nh3 >= 3:
        warnings_list.append("High ammonia inhibition risk — TAN may exceed 3,000 mg/L at mesophilic temp.")
        recs.append("Dilute high-protein feedstocks or lower operating temperature to reduce free ammonia.")
    elif nh3 == 2:
        recs.append("Moderate ammonia risk — monitor TAN and maintain C/N ratio above 20.")

    # ── Efficiency-based fallback ────────────────────────────
    worst_eff = min(
        effs["eff_CN"], effs["eff_OLR"], effs["eff_temp_base"],
        effs["eff_temp_fluctuation"], effs["eff_HRT"],
        effs["eff_TS"], effs["eff_ISR"], effs["eff_pressure"],
        effs["eff_lignin"]
    )
    if worst_eff < 0.5 and not warnings_list:
        warnings_list.append("One or more operating parameters are significantly outside optimal range.")
        recs.append("Review all parameters — the lowest efficiency factor is dragging overall yield.")

    return warnings_list, recs
