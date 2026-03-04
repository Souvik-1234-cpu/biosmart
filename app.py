# ============================================================
# app.py  —  BioSmart v4  (v5.8 fix applied)
# Flask backend: physics engine + 4-stage ML chain pipeline
# Run:  python app.py
#
# v5.8 FIX — overflow return jsonify():
#   * 3 wrong key names corrected:
#       "primary_cause"     -> "primary_failure_cause"
#       "geom_efficiency"   -> "geom_mean_efficiency"
#       "status_confidence" -> "status_probabilities"
#   * 11 missing fields added (efficiency_breakdown, blend_summary,
#     CO2_pct, overall_efficiency, blended_CN_ratio, TS_pct, VS_TS_pct,
#     total_VS_kg, OLR_kg_VS_m3_day, H2S_risk, ammonia_risk)
#   * Recommendation now gives specific volume/mass numbers
#   * These mismatches caused JS TypeError:
#     "Cannot convert undefined or null to object"
#     displayed as "Network error — could not reach server"
# ============================================================


import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template


# wrappers import is REQUIRED here even though the classes are not
# used directly — joblib.load() needs them registered in the
# current module's namespace to deserialize ml_pipeline.pkl
from wrappers import CatBoostClassifierWrapper, CatBoostRegressorWrapper  # noqa: F401


import engine
from engine import (
    blend_feedstocks,
    build_feature_vector,
    compute_all_efficiencies,
    generate_warnings_and_recs,
)


# ── App setup ────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR  = os.path.join(BASE_DIR, "templates")
DB_PATH       = os.path.join(BASE_DIR, "feedstock_database.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "ml_pipeline.pkl")


# Explicit template_folder so Flask resolves correctly on all platforms
app = Flask(__name__, template_folder=TEMPLATE_DIR)


# ── Startup sanity checks ────────────────────────────────────
_tmpl = os.path.join(TEMPLATE_DIR, "index.html")
if not os.path.isdir(TEMPLATE_DIR):
    raise RuntimeError(
        f"\n\n[BioSmart] 'templates' folder not found:\n  {TEMPLATE_DIR}\n"
        "Create the folder and place index.html inside it.\n"
    )
if not os.path.isfile(_tmpl):
    raise RuntimeError(
        f"\n\n[BioSmart] index.html not found:\n  {_tmpl}\n"
        "Move index.html into the 'templates/' subfolder.\n"
    )
print(f"✓ templates/index.html  found")


# ── Load feedstock database ──────────────────────────────────
_db_df       = pd.read_csv(DB_PATH)
FEEDSTOCK_DB = _db_df.set_index("feedstock_name").to_dict("index")
print(f"✓ Feedstock DB loaded: {len(FEEDSTOCK_DB)} feedstocks")


# ── Load ML pipeline ─────────────────────────────────────────
ML = None
if os.path.exists(MODEL_PATH):
    ML = joblib.load(MODEL_PATH)
    names = ML.get("model_names", {})
    print(f"✓ ML pipeline loaded — models: {names}")
    # ── v5.3: Set ALL models to CPU for inference ────────────────
    _model_keys = ["status_model", "cause_model", "yield_model", "ch4_model"]
    for _mk in _model_keys:
        _m = ML.get(_mk)
        if _m is None:
            continue
        if hasattr(_m, "get_booster"):
            try:
                _m.get_booster().set_param("device", "cpu")
                _m.get_booster().set_param("predictor", "cpu_predictor")
            except Exception:
                pass
        elif hasattr(_m, "set_params"):
            try:
                _m.set_params(task_type="CPU")
            except Exception:
                pass
        elif hasattr(_m, "set_param"):
            try:
                _m.set_param("device", "cpu")
                _m.set_param("predictor", "cpu_predictor")
            except Exception:
                pass
    print("✓ All models set to CPU inference mode (XGBoost + CatBoost)")
else:
    print("⚠  ml_pipeline.pkl not found — physics fallback active")



# ────────────────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")



@app.route("/api/feedstocks")
def api_feedstocks():
    """Return feedstock list grouped by category for the frontend dropdown."""
    rows = [
        {"name": name, "category": props["category"]}
        for name, props in FEEDSTOCK_DB.items()
    ]
    return jsonify(rows)



@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        body = request.get_json(force=True)


        # ── Parse feedstock components ───────────────────────
        components = [
            (item["name"], float(item["mass_kg"]))
            for item in body["feedstocks"]
        ]


        # ── Bundle all operating params into a single dict ───
        params = {
            "digester_volume_m3":      float(body["digester_volume_m3"]),
            "HRT_days":                float(body["HRT_days"]),
            "operating_temp_C":        float(body["operating_temp_C"]),
            "temp_fluctuation_degC":   float(body["temp_fluctuation_degC"]),
            "operating_pressure_kPa":  float(body["operating_pressure_kPa"]),
            "ISR":                     float(body["ISR"]),
            "urea_added_kg":           float(body.get("urea_added_kg", 0)),
            "is_continuous":           bool(body.get("is_continuous", True)),
            "n_components":            len(body["feedstocks"]),
            "water_added_kg":          float(body.get("water_added_kg", 0)),
        }


        # ── Blend feedstocks ─────────────────────────────────
        blend = blend_feedstocks(components, FEEDSTOCK_DB)


        # ── v5.1: Physical overflow check ─────────────────────────────
        # If total slurry mass > 90% of digester volume (in litres),
        # the digester is physically overfilled — hard FAIL, no biogas.
        _total_slurry_kg = sum(m for _, m in components) + float(body.get("water_added_kg", 0))
        _vol_litres      = float(body["digester_volume_m3"]) * 1000 * 0.90  # 90% working volume
        if _total_slurry_kg > _vol_litres:
            # v5.8 FIX: include ALL fields that the frontend JS expects.
            # Original response had 3 wrong key names and 11 missing fields,
            # causing JS TypeError displayed as "Network error — Cannot convert
            # undefined or null to object".
            return jsonify({
                # ── Core prediction fields ─────────────────────────────
                "run_status":              "FAIL",
                "primary_failure_cause":   "Physical_Overflow",
                "status_probabilities":    {"FAIL": 1.0, "PARTIAL_FAIL": 0.0, "SUCCESS": 0.0},
                "total_biogas_L":          0,
                "biogas_yield_L_per_kgVS": 0,
                "CH4_pct":                 0,
                "CO2_pct":                 0,
                # ── Efficiency fields ──────────────────────────────────
                "geom_mean_efficiency":    0,
                "overall_efficiency":      0,
                "efficiency_breakdown": {
                    "CN Ratio": 0, "OLR": 0, "Temp Base": 0, "Temp Fluct": 0,
                    "HRT": 0, "TS%": 0, "ISR": 0, "Pressure": 0, "Lignin": 0, "Mode": 0,
                },
                # ── Blend summary fields ───────────────────────────────
                "blended_CN_ratio":    round(blend["blended_CN_ratio"], 2),
                "TS_pct":              round(blend["TS_pct"], 2),
                "VS_TS_pct":           round(blend["VS_TS_pct"], 2),
                "total_VS_kg":         round(blend["total_VS_kg"], 2),
                "OLR_kg_VS_m3_day":    0,
                "H2S_risk":            blend["H2S_risk"],
                "ammonia_risk":        blend["NH3_risk"],
                "blend_summary": {
                    "n_components":     len(components),
                    "total_mass_kg":    round(_total_slurry_kg, 1),
                    "blended_CN_ratio": round(blend["blended_CN_ratio"], 2),
                    "TS_pct":           round(blend["TS_pct"], 2),
                },
                "model_names": {},
                # ── Diagnostics ────────────────────────────────────────
                "warnings": [
                    f"OVERFLOW: {_total_slurry_kg:.0f} kg of feed+water exceeds "
                    f"the digester working capacity of {_vol_litres:.0f} L "
                    f"({float(body['digester_volume_m3']):.1f} m³ × 90%)."
                ],
                "recommendations": [
                    f"Increase digester volume to at least "
                    f"{_total_slurry_kg / 900:.1f} m³, OR reduce "
                    f"feedstock mass to below {_vol_litres * 0.8:.0f} kg."
                ],
            })


        # ── Optional urea C/N correction ──────────────────────
        if params["urea_added_kg"] > 0:
            C_frac  = blend["blended_CN_ratio"] / (blend["blended_CN_ratio"] + 1)
            N_curr  = blend["total_VS_kg"] / (blend["blended_CN_ratio"] + 1)
            N_new   = N_curr + params["urea_added_kg"] * 0.46
            blend["blended_CN_ratio"] = (blend["total_VS_kg"] * C_frac) / N_new


        # ── Compute efficiencies (OLR computed inside engine) ─
        effs = compute_all_efficiencies(blend, params)
        OLR  = effs["OLR"]


        # ── Build 33-feature vector for ML ───────────────────
        features = build_feature_vector(blend, params, effs)


        # ── Warnings & recommendations (physics-based) ────────
        effs["ISR_input"]   = float(body["ISR"])
        effs["fluct_input"] = float(body.get("temp_fluctuation_degC", params["temp_fluctuation_degC"]))
        warnings_list, recs = generate_warnings_and_recs(
            blend, effs, OLR,
            params["HRT_days"],
            params["operating_temp_C"],
            corrected_TS_pct=effs.get("corrected_TS_pct")
        )


        # ── Predict via ML or physics fallback ────────────────
        if ML is not None:
            result = _run_chain_prediction(features, blend, effs)
        else:
            result = _physics_predict(blend, effs)


        # ── v5.2: Physics-based status correction ─────────────────────────
        _phys_yield_raw = blend["base_yield"] * effs["overall_efficiency_adjusted"] * blend["total_VS_kg"]
        _base_max_yield = blend["base_yield"] * blend["total_VS_kg"]
        if result.get("run_status","") == "SUCCESS" and _base_max_yield > 0:
            _yield_ratio = _phys_yield_raw / _base_max_yield
            if _yield_ratio < 0.40:
                result["run_status"]      = "PARTIAL_FAIL"
                result["primary_cause"]   = "Low_Yield_Despite_Parameters"
                _worst_eff = min(
                    {k: effs[k] for k in [
                        "eff_CN","eff_OLR","eff_temp_base","eff_temp_fluctuation",
                        "eff_HRT","eff_TS","eff_ISR","eff_pressure","eff_lignin"
                    ]}.items(), key=lambda x: x[1]
                )
                warnings_list.append(
                    f"Status downgraded: physics yield is only {_yield_ratio*100:.0f}% of theoretical "
                    f"(bottleneck: {_worst_eff[0].replace('eff_','')} = {_worst_eff[1]:.2f})."
                )


        # ── P6: CH4% cap ────────────────────────────────────────────
        _status = result.get("run_status", "SUCCESS")
        if _status == "FAIL":
            _ch4 = round(min(float(result["CH4_pct"]), 33.0), 1)
            result["CH4_pct"] = _ch4
            result["CO2_pct"] = round(max(20.0, 100.0 - _ch4 - 3.5), 1)
        elif _status == "PARTIAL_FAIL":
            _ch4 = round(min(float(result["CH4_pct"]), 49.0), 1)
            result["CH4_pct"] = _ch4
            result["CO2_pct"] = round(max(20.0, 100.0 - _ch4 - 2.5), 1)


        # ── Build full response ───────────────────────────────
        return jsonify({
            **result,

            "geom_mean_efficiency": round(effs["geom_mean_efficiency"], 4),
            "overall_efficiency":   round(effs["overall_efficiency"],   4),
            "efficiency_breakdown": {
                "CN Ratio":   round(effs["eff_CN"],               3),
                "OLR":        round(effs["eff_OLR"],              3),
                "Temp Base":  round(effs["eff_temp_base"],        3),
                "Temp Fluct": round(effs["eff_temp_fluctuation"], 3),
                "HRT":        round(effs["eff_HRT"],              3),
                "TS%":        round(effs["eff_TS"],               3),
                "ISR":        round(effs["eff_ISR"],              3),
                "Pressure":   round(effs["eff_pressure"],         3),
                "Lignin":     round(effs["eff_lignin"],           3),
                "Mode":       round(effs["eff_mode"],             3),
            },

            "blended_CN_ratio":        round(blend["blended_CN_ratio"], 2),
            "TS_pct":                  round(effs["corrected_TS_pct"],  2),
            "VS_TS_pct":               round(blend["VS_TS_pct"],        2),
            "total_VS_kg":             round(blend["total_VS_kg"],      2),
            "OLR_kg_VS_m3_day":        round(OLR,                       3),
            "H2S_risk":                blend["H2S_risk"],
            "ammonia_risk":            blend["NH3_risk"],
            "biogas_yield_L_per_kgVS": round(
                result.get("total_biogas_L", 0) / max(blend["total_VS_kg"], 0.001), 1
            ),

            "warnings":        warnings_list,
            "recommendations": recs,
            "blend_summary": {
                "n_components":     len(components),
                "total_mass_kg":    round(blend["total_mass_kg"], 1),
                "blended_CN_ratio": round(blend["blended_CN_ratio"], 2),
                "TS_pct":           round(blend["TS_pct"], 2),
            },
        })


    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400




# ────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ────────────────────────────────────────────────────────────


def _run_chain_prediction(features, blend, effs=None):
    """
    Runs the 4-stage ML chain pipeline.
    """
    base_cols = ML["base_features"]

    raw_df   = pd.DataFrame([{c: features.get(c, 0) for c in base_cols}])
    X_scaled = pd.DataFrame(
        ML["scaler"].transform(raw_df), columns=base_cols)

    # Stage 1 — Run Status
    pred_status_code = int(ML["status_model"].predict(X_scaled.values)[0])
    pred_status_text = ML["status_encoder"].inverse_transform(
        [pred_status_code])[0]

    status_proba = {}
    if hasattr(ML["status_model"], "predict_proba"):
        proba = ML["status_model"].predict_proba(X_scaled.values)[0]
        status_proba = {
            cls: round(float(p), 3)
            for cls, p in zip(ML["status_encoder"].classes_, proba)
        }

    # Stage 2 — Failure Cause
    if pred_status_text in ["FAIL", "PARTIAL_FAIL"]:
        code = int(ML["cause_model"].predict(X_scaled.values)[0])
        pred_cause = ML["cause_encoder_fail"].inverse_transform([code])[0]
    else:
        pred_cause = "N/A — System Stable"

    # Stage 3 — Total Biogas Yield
    X_yield = X_scaled.copy()
    X_yield["chain_status"] = pred_status_code
    _raw_yield = float(ML["yield_model"].predict(X_yield[ML["yield_features"]].values)[0])
    if effs is not None:
        _phys_yield = blend["base_yield"] * effs["overall_efficiency_adjusted"] * blend["total_VS_kg"]
        _phys_ceil  = _phys_yield * 1.12
        if _raw_yield > 0:
            total_biogas = min(max(_raw_yield, _phys_yield * 0.85), _phys_ceil)
        else:
            total_biogas = max(0.0, _phys_yield)
    elif _raw_yield > 0:
        total_biogas = _raw_yield
    else:
        total_biogas = 0.0

    # Stage 4 — CH4 %
    X_ch4 = X_yield.copy()
    X_ch4["chain_yield"] = total_biogas
    ch4 = float(ML["ch4_model"].predict(X_ch4[ML["ch4_features"]].values)[0])
    co2 = round(max(20.0, 100.0 - ch4 - 2.5), 1)

    return {
        "run_status":            pred_status_text,
        "status_probabilities":  status_proba,
        "primary_failure_cause": pred_cause,
        "total_biogas_L":        round(total_biogas, 1),
        "CH4_pct":               round(ch4, 1),
        "CO2_pct":               co2,
    }



def _physics_predict(blend, effs):
    """Fallback when ml_pipeline.pkl is not available."""
    overall    = effs["overall_efficiency_adjusted"]
    geom       = effs["geom_mean_efficiency"]
    yield_pred = max(0.0, blend["base_yield"] * overall)
    total_L    = yield_pred * blend["total_VS_kg"]
    ch4        = float(blend["base_CH4"])
    co2        = round(max(20.0, 100.0 - ch4 - 2.5), 1)

    if geom >= 0.80:
        status = "SUCCESS"
    elif geom >= 0.67:
        status = "PARTIAL_FAIL"
    else:
        status = "FAIL"

    eff_map = {
        "CN Ratio":  effs["eff_CN"],
        "OLR":       effs["eff_OLR"],
        "Temp Base": effs["eff_temp_base"],
        "Temp Fluct":effs["eff_temp_fluctuation"],
        "HRT":       effs["eff_HRT"],
        "TS%":       effs["eff_TS"],
        "ISR":       effs["eff_ISR"],
        "Pressure":  effs["eff_pressure"],
        "Lignin":    effs["eff_lignin"],
    }
    cause = min(eff_map, key=eff_map.get) if status != "SUCCESS" else "N/A — System Stable"

    if status == "FAIL":
        ch4 = min(ch4, 33.0)
    elif status == "PARTIAL_FAIL":
        ch4 = min(ch4, 49.0)
    co2 = round(max(20.0, 100.0 - ch4 - 2.5), 1)

    return {
        "run_status":            status,
        "status_probabilities":  {},
        "primary_failure_cause": cause,
        "total_biogas_L":        round(total_L, 1),
        "CH4_pct":               round(ch4, 1),
        "CO2_pct":               co2,
    }



# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, port=5000)
