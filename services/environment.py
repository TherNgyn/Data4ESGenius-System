from __future__ import annotations

import io
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "config" / "environment_weights.json"
ML_CONFIG_PATH = ROOT / "config" / "ml_config.json"
MODEL_ROOT = ROOT / "models"
PROCESSED_DATA_PATH = ROOT / "data" / "processed" / "environment_records.csv"
MONTHS = {
    "JAN": "Jan", "FEB": "Feb", "MAR": "Mar", "APR": "Apr",
    "MAY": "May", "JUN": "Jun", "JUL": "Jul", "AUG": "Aug",
    "SEP": "Sep", "OCT": "Oct", "NOV": "Nov", "DEC": "Dec",
}
MONTH_ORDER = {month: index for index, month in enumerate(MONTHS.values())}
PERIOD_COLUMNS = {"YTD": "YTD", **{f"Q{i}": f"Q{i}" for i in range(1, 5)}}
RECORD_COLUMNS = [
    "record_id", "indicator", "category", "value", "unit", "reporting_period", "year",
    "month", "quarter", "location", "data_source", "data_type", "raw_record",
]


def load_weights():
    with WEIGHTS_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def load_ml_config():
    with ML_CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def _read_csv(path_or_file):
    if hasattr(path_or_file, "read"):
        content = path_or_file.read()
        path_or_file.seek(0)
        return pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    try:
        return pd.read_csv(path_or_file, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path_or_file, encoding="cp1252")


def load_extracted_files(root=ROOT):
    frames = {}
    for folder in ("Emission", "Energy_GHG", "Waste", "Water"):
        for path in sorted((Path(root) / folder).glob("*.csv")):
            frames[path.name] = _read_csv(path)
    return frames


def load_uploaded_files(files):
    frames = {}
    for uploaded in files:
        name = getattr(uploaded, "name", "uploaded.csv")
        suffix = Path(name).suffix.lower()
        if suffix == ".csv":
            frames[name] = _read_csv(uploaded)
        elif suffix in {".xls", ".xlsx"}:
            uploaded.seek(0)
            sheets = pd.read_excel(uploaded, sheet_name=None)
            frames.update({f"{name} · {sheet}": frame for sheet, frame in sheets.items()})
        else:
            raise ValueError(f"Unsupported file format: {name}")
    return frames


def _category_for(source, indicator):
    name = f"{source} {indicator}".casefold()
    if "greenhouse" in name or "scope" in name or "ghg" in name:
        return "GHG Emissions"
    if "energy" in name or "coal" in name or "electricity" in name or "biomass" in name:
        return "Energy Management"
    if "water" in name:
        return "Water Management"
    if "waste" in name or "landfill" in name or "recycle" in name:
        return "Waste Management"
    if any(token in name for token in ("dust", "nox", "sox", "emission")):
        return "Air Quality Emissions"
    return "Unmapped"


def _location(indicator):
    match = re.search(r"\((DV\s*\+\s*YB|DV|YB)\)", str(indicator), re.IGNORECASE)
    return re.sub(r"\s+", "", match.group(1).upper()) if match else None


def _period(column):
    normalized = re.sub(r"[^A-Z0-9]", "", str(column).upper())
    if normalized in MONTHS:
        month = MONTHS[normalized]
        return month, month, None, "Monthly"
    if normalized in PERIOD_COLUMNS:
        period = PERIOD_COLUMNS[normalized]
        if period == "YTD":
            return period, None, None, "YTD"
        return period, None, period, "Quarterly"
    return None


def _base_record(indicator, value, unit, period, year, month, quarter, location, source, data_type, raw):
    record_values = json.dumps(
        [source, raw.get("_source_row"), indicator, period, year, month, quarter, float(value)],
        ensure_ascii=False,
        default=str,
    )
    return {
        "record_id": hashlib.sha256(record_values.encode("utf-8")).hexdigest()[:16],
        "indicator": str(indicator).strip(),
        "category": _category_for(source, indicator),
        "value": float(value),
        "unit": str(unit).strip() if pd.notna(unit) else None,
        "reporting_period": period,
        "year": year,
        "month": month,
        "quarter": quarter,
        "location": location,
        "data_source": source,
        "data_type": data_type,
        "raw_record": raw,
    }


def _normalize_wide(frame, source):
    records = []
    if frame.shape[1] < 3:
        return records
    columns = list(frame.columns)
    for source_row, row in frame.iterrows():
        indicator = row.iloc[0]
        unit = row.iloc[1]
        if pd.isna(indicator) or str(indicator).strip().lower() in {"", "nan"}:
            continue
        raw_record = row.to_dict()
        raw_record["_source_row"] = int(source_row)
        for index, column in enumerate(columns[2:], start=2):
            period_info = _period(column)
            if period_info is None:
                continue
            value = pd.to_numeric(row.iloc[index], errors="coerce")
            if pd.isna(value):
                continue
            period, month, quarter, data_type = period_info
            records.append(_base_record(
                indicator, value, unit, period, None, month, quarter,
                _location(indicator), source, data_type, raw_record,
            ))
    return records


def _normalize_emissions(frame, source):
    records = []
    stem = Path(source.split(" · ")[0]).stem.casefold()
    metric = "Dust" if "dust" in stem else "NOx" if "nox" in stem else "SOx" if "sox" in stem else None
    if metric is None:
        return records
    total_column = "dust_emission_total" if metric == "Dust" else "emission_total"
    value_column = "dust_emission_actual_o2" if metric == "Dust" else "emission_actual_o2"
    for source_row, row in frame.iterrows():
        month = str(row.get("report_month", "")).strip().title()
        if month not in MONTH_ORDER:
            continue
        sample_date = pd.to_datetime(row.get("sampling_date", row.get("sampling_month_year")), errors="coerce", dayfirst=True)
        year = int(sample_date.year) if not pd.isna(sample_date) else None
        raw_record = row.to_dict()
        raw_record["_source_row"] = int(source_row)
        for column, suffix, unit in ((total_column, "total emissions", "kg"), (value_column, "concentration", "mg/m3")):
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.isna(value):
                continue
            indicator = f"{metric} {suffix}"
            records.append(_base_record(
                indicator, value, unit, month, year, month, None, None,
                source, "Monthly", raw_record,
            ))
    return records


def _normalize_waste(frame, source):
    records = []
    metric_columns = {
        "waste_generated": "generated",
        "reuse_recycle": "reused/recycled",
        "incineration": "incinerated",
        "landfill": "landfilled",
    }
    for source_row, row in frame.iterrows():
        month = str(row.get("report_month", "")).strip().title()
        if month not in MONTH_ORDER:
            continue
        waste_type = "Hazardous" if "hazardous" in str(row.get("section", "")).casefold() else "Non-hazardous"
        item = str(row.get("waste_category_en", "")).strip()
        raw_record = row.to_dict()
        raw_record["_source_row"] = int(source_row)
        for column, label in metric_columns.items():
            value = pd.to_numeric(row.get(column), errors="coerce")
            if pd.isna(value):
                continue
            records.append(_base_record(
                f"{waste_type} waste {label}: {item}", value, "ton", month,
                None, month, None, None, source, "Monthly", raw_record,
            ))
    return records


def normalize_frames(frames):
    records = []
    duplicate_rows = sum(int(frame.duplicated().sum()) for frame in frames.values())
    for source, frame in frames.items():
        columns = {str(column).strip().casefold() for column in frame.columns}
        if {"report_month", "waste_generated"}.issubset(columns):
            records.extend(_normalize_waste(frame, source))
        elif "report_month" in columns and ("emission_total" in columns or "dust_emission_total" in columns):
            records.extend(_normalize_emissions(frame, source))
        else:
            records.extend(_normalize_wide(frame, source))
    normalized = pd.DataFrame(records, columns=RECORD_COLUMNS)
    diagnostics = {
        "duplicate_rows": duplicate_rows,
        "unmapped_rows": int((normalized["category"] == "Unmapped").sum()) if not normalized.empty else 0,
        "source_files": len(frames),
        "source_missing_cells": sum(int(frame.isna().sum().sum()) for frame in frames.values()),
        "negative_values": int((normalized["value"] < 0).sum()) if not normalized.empty else 0,
        "zero_values": int((normalized["value"] == 0).sum()) if not normalized.empty else 0,
    }
    return normalized, diagnostics


def assess_coverage(records, weights=None):
    weights = weights or load_weights()
    present_names = records.loc[records["value"].notna(), "indicator"].astype(str).str.casefold().tolist() if not records.empty else []
    groups = {}
    weighted_coverage = 0.0
    for group, details in weights["groups"].items():
        indicator_coverage = {}
        for indicator, spec in details["indicators"].items():
            aliases = [alias.casefold() for alias in spec["aliases"]]
            available = any(alias in name for alias in aliases for name in present_names)
            indicator_coverage[indicator] = {
                "weight": spec["weight"],
                "available": available,
                "score": None,
                "status": "Data available; benchmark required" if available else "No data",
            }
        coverage = sum(item["weight"] for item in indicator_coverage.values() if item["available"])
        groups[group] = {
            "weight": details["weight"],
            "coverage": coverage,
            "score": None,
            "indicators": indicator_coverage,
        }
        weighted_coverage += details["weight"] * coverage
    return {
        "score": None,
        "score_status": "Unavailable: no approved normalization benchmarks are configured",
        "coverage": weighted_coverage,
        "groups": groups,
    }


def calculate_environment_assessment(records, weights=None, minimum_environment_coverage=0.50):
    weights = weights or load_weights()
    base = assess_coverage(records, weights)
    group_results = {}
    evaluated_group_weight = 0.0
    environment_weighted_sum = 0.0

    for group_name, details in weights["groups"].items():
        indicator_results = {}
        available_weight = 0.0
        scored_weight = 0.0
        weighted_score_sum = 0.0

        for indicator_name, spec in details["indicators"].items():
            aliases = [alias.casefold() for alias in spec["aliases"]]
            matching_indicator = None
            matching_records = pd.DataFrame()
            for indicator in sorted(records["indicator"].dropna().unique(), key=str.lower):
                indicator_text = str(indicator).casefold()
                if any(alias in indicator_text for alias in aliases):
                    matching_indicator = indicator
                    matching_records = records[records["indicator"] == indicator].copy()
                    break

            indicator_weight = spec["weight"]
            indicator_result = {
                "weight": indicator_weight,
                "available": not matching_records.empty,
                "score": None,
                "status": "No data",
                "method": None,
                "actual_value": None,
                "previous_value": None,
                "change_pct": None,
            }

            if matching_records.empty:
                indicator_results[indicator_name] = indicator_result
                continue

            trend = monthly_trend(matching_records, matching_indicator)
            trend = trend.dropna(subset=["value"])
            if len(trend) < 2:
                indicator_result["status"] = "Available but no previous period"
                indicator_results[indicator_name] = indicator_result
                continue

            previous_value = float(trend.iloc[-2]["value"])
            actual_value = float(trend.iloc[-1]["value"])
            if previous_value == 0:
                indicator_result["status"] = "Previous period is zero; previous-period ratio unavailable"
                indicator_results[indicator_name] = indicator_result
                continue

            change_pct = (actual_value - previous_value) / abs(previous_value) * 100
            indicator_text = f"{indicator_name} {matching_indicator}".casefold()
            higher_is_better = any(token in indicator_text for token in (
                "renewable", "reduction", "recycling", "reuse", "recovery"
            ))
            if higher_is_better:
                relative_change = (actual_value - previous_value) / abs(previous_value)
            else:
                relative_change = (previous_value - actual_value) / abs(previous_value)

            score = int(round(max(0.0, min(100.0, 50.0 + 50.0 * relative_change))))
            indicator_result.update({
                "available": True,
                "score": score,
                "status": "Temporary score",
                "method": "Previous-period comparison",
                "actual_value": actual_value,
                "previous_value": previous_value,
                "change_pct": change_pct,
            })
            available_weight += indicator_weight
            scored_weight += indicator_weight
            weighted_score_sum += indicator_weight * score
            indicator_results[indicator_name] = indicator_result

        group_score = None
        redistribution_note = None
        if scored_weight > 0:
            group_score = weighted_score_sum / scored_weight
            if scored_weight < sum(spec["weight"] for spec in details["indicators"].values()) - 1e-9:
                redistribution_note = (
                    f"Temporary group weight redistributed over {scored_weight:.0%} of scored indicator weight. "
                    "This redistribution is disclosed and is not a benchmark-based final score."
                )

        group_results[group_name] = {
            "weight": details["weight"],
            "coverage": available_weight,
            "scored_weight": scored_weight,
            "score": group_score,
            "status": "Temporary score" if group_score is not None else "Missing or insufficient previous-period data",
            "redistribution_note": redistribution_note,
            "indicators": indicator_results,
        }

        if group_score is not None:
            evaluated_group_weight += details["weight"]
            environment_weighted_sum += details["weight"] * group_score

    if evaluated_group_weight >= minimum_environment_coverage and environment_weighted_sum > 0:
        environment_score = environment_weighted_sum / evaluated_group_weight
        environment_status = "Temporary score based on previous-period comparison"
    else:
        environment_score = None
        environment_status = "Unavailable: insufficient coverage for temporary previous-period scoring"

    indicator_statuses = []
    for group in group_results.values():
        for indicator_name, indicator in group["indicators"].items():
            if indicator["score"] is not None:
                indicator_statuses.append({
                    "indicator": indicator_name,
                    "score": indicator["score"],
                    "status": indicator["status"],
                })

    return {
        "score": round(environment_score, 1) if environment_score is not None else None,
        "score_status": environment_status,
        "coverage": base["coverage"],
        "evaluated_environment_weight": evaluated_group_weight,
        "redistribution_disclosed": evaluated_group_weight < 1.0,
        "method": "Previous-period comparison; 50 is unchanged, higher is better or lower is better according to indicator direction",
        "groups": group_results,
        "indicator_results": indicator_statuses,
    }


def monthly_trend(records, indicator):
    subset = records[(records["indicator"] == indicator) & (records["data_type"] == "Monthly")].copy()
    if subset.empty:
        return subset
    subset["month_order"] = subset["month"].map(MONTH_ORDER)
    aggregation = "mean" if subset["unit"].astype(str).str.contains("mg/m3", case=False).all() else "sum"
    return subset.groupby(["year", "month", "month_order"], dropna=False, as_index=False).agg(
        value=("value", aggregation), unit=("unit", "first")
    ).sort_values(["year", "month_order"], na_position="last")


def z_score_anomalies(trend, threshold=2.5):
    if trend.empty or len(trend) < 8:
        return trend.iloc[0:0].assign(z_score=pd.Series(dtype=float))
    values = trend["value"].astype(float)
    deviation = values.std(ddof=0)
    if not deviation or np.isnan(deviation):
        return trend.iloc[0:0].assign(z_score=pd.Series(dtype=float))
    z_scores = (values - values.mean()) / deviation
    anomalies = trend.loc[z_scores.abs() >= threshold].copy()
    anomalies["z_score"] = z_scores.loc[anomalies.index]
    anomalies["deviation_from_mean"] = anomalies["value"] - values.mean()
    anomalies["severity"] = anomalies["z_score"].abs().map(
        lambda value: "High" if value >= 3.5 else "Review"
    )
    return anomalies


def persist_processed_records(records, output_path=PROCESSED_DATA_PATH):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    export = records.copy()
    if "raw_record" in export:
        export["raw_record"] = export["raw_record"].map(
            lambda value: json.dumps(value, ensure_ascii=False, default=str)
        )
    export.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _indicator_key(indicator):
    return hashlib.sha256(indicator.encode("utf-8")).hexdigest()[:16]


def _metadata_path(indicator, model_root):
    return Path(model_root) / "metadata" / f"{_indicator_key(indicator)}.json"


def _read_indicator_metadata(indicator, model_root):
    path = _metadata_path(indicator, model_root)
    if not path.exists():
        return {"indicator": indicator, "active_version": None, "versions": []}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def monthly_training_candidates(records, minimum_observations=None):
    config = load_ml_config()
    minimum_observations = minimum_observations or config["anomaly_min_observations"]
    indicators = sorted(records.loc[records["data_type"] == "Monthly", "indicator"].dropna().unique())
    candidates = []
    for indicator in indicators:
        trend = monthly_trend(records, indicator)
        values = pd.to_numeric(trend["value"], errors="coerce").dropna()
        reason = None
        if len(values) < minimum_observations:
            reason = f"Requires at least {minimum_observations} monthly observations; found {len(values)}"
        elif (values < 0).any():
            reason = "Negative physical measurements require source review"
        elif values.nunique() < 3:
            reason = "At least three distinct values are required"
        candidates.append({
            "indicator": indicator,
            "observations": len(values),
            "eligible": reason is None,
            "reason": reason or "Eligible for unsupervised training",
            "first_year": int(trend["year"].dropna().min()) if trend["year"].notna().any() else None,
            "last_year": int(trend["year"].dropna().max()) if trend["year"].notna().any() else None,
        })
    return candidates


def train_anomaly_model(records, indicator, contamination=None, model_root=MODEL_ROOT, training_root=None):
    config = load_ml_config()
    contamination = contamination or config["isolation_forest_contamination"]
    model_root = Path(model_root)
    training_root = Path(training_root) if training_root else ROOT / "data" / "training"
    trend = monthly_trend(records, indicator).dropna(subset=["value"]).reset_index(drop=True)
    values = pd.to_numeric(trend["value"], errors="coerce").to_numpy(dtype=float)
    if len(values) < config["anomaly_min_observations"]:
        raise ValueError(f"At least {config['anomaly_min_observations']} monthly observations are required; found {len(values)}.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Training data contains missing, non-finite or negative physical measurements.")
    if np.unique(values).size < 3:
        raise ValueError("At least three distinct observed values are required for Isolation Forest.")
    if not 0.01 <= contamination <= 0.25:
        raise ValueError("Contamination must be between 0.01 and 0.25.")

    train_end = int(len(values) * config["train_fraction"])
    validation_end = int(len(values) * (config["train_fraction"] + config["validation_fraction"]))
    if train_end < 10 or validation_end <= train_end or validation_end >= len(values):
        raise ValueError("Not enough ordered observations for chronological train, validation and test partitions.")

    transformed = np.log1p(values).reshape(-1, 1)
    model = make_pipeline(
        StandardScaler(),
        IsolationForest(
            n_estimators=config["isolation_forest_estimators"],
            contamination=contamination,
            random_state=config["random_state"],
        ),
    )
    model.fit(transformed[:train_end])
    validation_predictions = model.predict(transformed[train_end:validation_end])
    test_predictions = model.predict(transformed[validation_end:])
    baseline_values = values[:train_end]
    baseline_std = float(baseline_values.std(ddof=0))
    if baseline_std <= 0:
        raise ValueError("Training-period values have no variance; a Z-score baseline cannot be fitted.")

    metadata = _read_indicator_metadata(indicator, model_root)
    next_version = max((int(item["version"].lstrip("v")) for item in metadata["versions"]), default=0) + 1
    version = f"v{next_version:03d}"
    indicator_key = _indicator_key(indicator)
    artifact_relative = Path("anomaly_detection") / indicator_key / f"{version}.joblib"
    artifact_path = model_root / artifact_relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    trend["split"] = ["train"] * train_end + ["validation"] * (validation_end - train_end) + ["test"] * (len(values) - validation_end)
    training_root.mkdir(parents=True, exist_ok=True)
    dataset_path = training_root / f"{indicator_key}_{version}.csv"
    trend.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    dataset_hash = hashlib.sha256(np.asarray(values, dtype="float64").tobytes()).hexdigest()
    artifact = {
        "model": model,
        "indicator": indicator,
        "version": version,
        "zscore_mean": float(baseline_values.mean()),
        "zscore_std": baseline_std,
        "zscore_threshold": config["zscore_threshold"],
    }
    joblib.dump(artifact, artifact_path)

    is_first_version = metadata["active_version"] is None
    version_metadata = {
        "version": version,
        "model_name": "IsolationForest with StandardScaler",
        "target_indicator": indicator,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_hash[:16],
        "training_period": [
            f"{int(trend.iloc[0]['year'])}-{trend.iloc[0]['month']}" if pd.notna(trend.iloc[0]["year"]) else str(trend.iloc[0]["month"]),
            f"{int(trend.iloc[-1]['year'])}-{trend.iloc[-1]['month']}" if pd.notna(trend.iloc[-1]["year"]) else str(trend.iloc[-1]["month"]),
        ],
        "number_of_records": len(values),
        "train_records": train_end,
        "validation_records": validation_end - train_end,
        "test_records": len(values) - validation_end,
        "features": ["log1p(value)"],
        "hyperparameters": {
            "n_estimators": config["isolation_forest_estimators"],
            "contamination": contamination,
            "random_state": config["random_state"],
        },
        "evaluation": {
            "validation_anomaly_rate": float(np.mean(validation_predictions == -1)),
            "test_anomaly_rate": float(np.mean(test_predictions == -1)),
            "evaluation_note": "Unsupervised holdout alert rates; no labels are available, so accuracy metrics are not applicable.",
        },
        "artifact": str(artifact_relative),
        "training_dataset": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "status": "Active" if is_first_version else "Candidate",
    }
    metadata["versions"].append(version_metadata)
    if is_first_version:
        metadata["active_version"] = version
    metadata_path = _metadata_path(indicator, model_root)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return version_metadata


def list_anomaly_models(model_root=MODEL_ROOT):
    metadata_dir = Path(model_root) / "metadata"
    if not metadata_dir.exists():
        return []
    models = []
    for path in sorted(metadata_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        active = next((item for item in data["versions"] if item["version"] == data["active_version"]), None)
        latest = data["versions"][-1] if data["versions"] else None
        if latest:
            models.append({"indicator": data["indicator"], "active_version": data["active_version"],
                           "latest_version": latest["version"], "status": latest["status"],
                           "records": latest["number_of_records"], "trained_at": latest["trained_at"],
                           "test_anomaly_rate": latest["evaluation"]["test_anomaly_rate"],
                           "active_artifact": active["artifact"] if active else None})
    return models


def anomaly_model_versions(indicator, model_root=MODEL_ROOT):
    metadata = _read_indicator_metadata(indicator, model_root)
    return metadata["versions"]


def load_active_anomaly_model(indicator, model_root=MODEL_ROOT):
    metadata = _read_indicator_metadata(indicator, model_root)
    active = next((item for item in metadata["versions"] if item["version"] == metadata["active_version"]), None)
    if active is None:
        return None
    artifact_path = Path(model_root) / active["artifact"]
    artifact = joblib.load(artifact_path)
    if artifact.get("indicator") != indicator or artifact.get("version") != active["version"]:
        raise ValueError("Loaded model artifact metadata does not match the requested indicator/version.")
    return artifact


def promote_anomaly_model(indicator, version, model_root=MODEL_ROOT):
    metadata = _read_indicator_metadata(indicator, model_root)
    if not any(item["version"] == version for item in metadata["versions"]):
        raise ValueError(f"Model version {version} does not exist for {indicator}.")
    for item in metadata["versions"]:
        item["status"] = "Active" if item["version"] == version else "Archived"
    metadata["active_version"] = version
    _metadata_path(indicator, model_root).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return next(item for item in metadata["versions"] if item["version"] == version)


def infer_anomalies(trend, artifact):
    if trend.empty:
        return trend.copy()
    values = pd.to_numeric(trend["value"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Inference data contains missing, non-finite or negative physical measurements.")
    transformed = np.log1p(values).reshape(-1, 1)
    model_predictions = artifact["model"].predict(transformed)
    model_scores = -artifact["model"].decision_function(transformed)
    z_scores = (values - artifact["zscore_mean"]) / artifact["zscore_std"]
    result = trend.copy()
    result["isolation_forest_score"] = model_scores
    result["z_score"] = z_scores
    result["deviation_from_training_mean"] = values - artifact["zscore_mean"]
    result["anomaly_method"] = [
        ", ".join(method for method, flagged in (
            ("Isolation Forest", model_predictions[index] == -1),
            ("Z-score", abs(z_scores[index]) >= artifact["zscore_threshold"]),
        ) if flagged)
        for index in range(len(values))
    ]
    return result[result["anomaly_method"] != ""].assign(status="Review")


def forecast_readiness(records, indicator, minimum_observations=None):
    minimum_observations = minimum_observations or load_ml_config()["forecast_min_observations"]
    trend = monthly_trend(records, indicator).dropna(subset=["value"]).copy()
    if len(trend) < minimum_observations:
        return False, f"At least {minimum_observations} monthly observations are required; found {len(trend)}."
    if trend["year"].isna().any():
        return False, "Reporting year is missing; a time-ordered forecast cannot be validated."
    periods = pd.PeriodIndex(year=trend["year"].astype(int), month=trend["month"].map(MONTH_ORDER) + 1, freq="M")
    if len(periods.unique()) != len(periods) or not periods.is_monotonic_increasing:
        return False, "The monthly series contains duplicate or unordered periods."
    if len(periods) > 1:
        ordinals = np.asarray([period.ordinal for period in periods], dtype=int)
        if not np.all(np.diff(ordinals) == 1):
            return False, "The monthly series has gaps and is not continuous."
    return True, "Sufficient continuous monthly history for a chronological forecasting evaluation."


def _forecast_metadata_path(indicator, model_root):
    return Path(model_root) / "forecasting" / "metadata" / f"{_indicator_key(indicator)}.json"


def _forecast_series(records, indicator):
    trend = monthly_trend(records, indicator).dropna(subset=["value"]).copy()
    if trend.empty or trend["year"].isna().any():
        return trend, None
    trend["period"] = pd.PeriodIndex(
        year=trend["year"].astype(int),
        month=trend["month"].map(MONTH_ORDER) + 1,
        freq="M",
    )
    trend = trend.sort_values("period").drop_duplicates("period")
    periods = trend["period"]
    if len(periods) > 1:
        ordinals = np.asarray([period.ordinal for period in periods], dtype=int)
        if not np.all(np.diff(ordinals) == 1):
            return trend, None
    return trend, trend.set_index("period")["value"].astype(float)


def _forecast_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mask = np.abs(actual) >= 1e-12
    mape = float(np.mean(np.abs(errors[mask] / actual[mask])) * 100) if mask.any() else None
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def train_forecast_model(records, indicator, model_root=MODEL_ROOT, forecast_training_root=None):
    config = load_ml_config()
    ready, reason = forecast_readiness(records, indicator, config["forecast_min_observations"])
    if not ready:
        raise ValueError(reason)

    trend, series = _forecast_series(records, indicator)
    if series is None or len(series) < config["forecast_min_observations"]:
        raise ValueError(reason)
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Forecasting data contains missing, non-finite or negative physical measurements.")

    train_end = int(len(values) * config["train_fraction"])
    validation_end = int(len(values) * (config["train_fraction"] + config["validation_fraction"]))
    if train_end < 12 or validation_end <= train_end or validation_end >= len(values):
        raise ValueError("Not enough ordered observations for chronological SARIMA train, validation and test partitions.")

    order = tuple(config["sarima_order"])
    seasonal_order = tuple(config["sarima_seasonal_order"])
    train_values = values[:train_end]
    validation_values = values[train_end:validation_end]
    test_values = values[validation_end:]

    train_result = SARIMAX(train_values, order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    validation_forecast = train_result.get_forecast(steps=len(validation_values)).predicted_mean
    validation_metrics = _forecast_metrics(validation_values, validation_forecast)

    evaluation_result = SARIMAX(values[:validation_end], order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    test_forecast = evaluation_result.get_forecast(steps=len(test_values)).predicted_mean
    test_metrics = _forecast_metrics(test_values, test_forecast)

    full_result = SARIMAX(values, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    model_root = Path(model_root)
    metadata_path = _forecast_metadata_path(indicator, model_root)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {"indicator": indicator, "active_version": None, "versions": []}

    next_version = max((int(item["version"].lstrip("v")) for item in metadata["versions"]), default=0) + 1
    version = f"v{next_version:03d}"
    artifact_relative = Path("forecasting") / _indicator_key(indicator) / f"{version}.joblib"
    artifact_path = model_root / artifact_relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(forecast_training_root) if forecast_training_root else ROOT / "data" / "training"
    dataset_root.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_root / f"{_indicator_key(indicator)}_forecast_{version}.csv"
    trend.to_csv(dataset_path, index=False, encoding="utf-8-sig")

    joblib.dump({
        "result": full_result,
        "indicator": indicator,
        "version": version,
        "unit": trend["unit"].iloc[0] if "unit" in trend else None,
        "periods": trend["period"].astype(str).tolist(),
        "order": order,
        "seasonal_order": seasonal_order,
    }, artifact_path)

    is_first_version = metadata["active_version"] is None
    version_metadata = {
        "version": version,
        "model_name": "SARIMA / SARIMAX",
        "target_indicator": indicator,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": hashlib.sha256(np.asarray(values, dtype="float64").tobytes()).hexdigest()[:16],
        "training_period": [str(series.index[0]), str(series.index[-1])],
        "number_of_records": len(values),
        "train_records": train_end,
        "validation_records": validation_end - train_end,
        "test_records": len(values) - validation_end,
        "features": ["monthly_value"],
        "hyperparameters": {"order": list(order), "seasonal_order": list(seasonal_order)},
        "evaluation": {"validation": validation_metrics, "test": test_metrics},
        "artifact": str(artifact_relative),
        "training_dataset": str(dataset_path.relative_to(ROOT)) if dataset_path.is_relative_to(ROOT) else str(dataset_path),
        "status": "Active" if is_first_version else "Candidate",
    }
    metadata["versions"].append(version_metadata)
    if is_first_version:
        metadata["active_version"] = version
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return version_metadata


def list_forecast_models(model_root=MODEL_ROOT):
    metadata_dir = Path(model_root) / "forecasting" / "metadata"
    if not metadata_dir.exists():
        return []
    models = []
    for path in sorted(metadata_dir.glob("*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        active = next((item for item in metadata["versions"] if item["version"] == metadata["active_version"]), None)
        latest = metadata["versions"][-1] if metadata["versions"] else None
        if latest:
            models.append({
                "indicator": metadata["indicator"],
                "active_version": metadata["active_version"],
                "latest_version": latest["version"],
                "status": latest["status"],
                "records": latest["number_of_records"],
                "trained_at": latest["trained_at"],
                "test_metrics": latest["evaluation"]["test"],
            })
    return models


def load_active_forecast_model(indicator, model_root=MODEL_ROOT):
    metadata_path = _forecast_metadata_path(indicator, model_root)
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    active = next((item for item in metadata["versions"] if item["version"] == metadata["active_version"]), None)
    if active is None:
        return None
    artifact = joblib.load(Path(model_root) / active["artifact"])
    if artifact.get("indicator") != indicator or artifact.get("version") != active["version"]:
        raise ValueError("Forecast artifact metadata does not match the requested indicator/version.")
    return artifact


def infer_forecast(indicator, horizon=None, model_root=MODEL_ROOT):
    artifact = load_active_forecast_model(indicator, model_root)
    if artifact is None:
        return None
    horizon = horizon or load_ml_config()["forecast_horizon"]
    forecast = artifact["result"].get_forecast(steps=horizon)
    predicted = np.asarray(forecast.predicted_mean, dtype=float)
    confidence = np.asarray(forecast.conf_int(alpha=0.05), dtype=float)
    if confidence.ndim != 2 or confidence.shape[1] < 2:
        raise ValueError("SARIMA confidence interval has an unsupported shape.")
    last_period = pd.Period(artifact["periods"][-1], freq="M")
    periods = pd.period_range(last_period + 1, periods=horizon, freq="M")
    result = pd.DataFrame({
        "period": periods.astype(str),
        "forecast": predicted,
        "lower_95": confidence[:, 0],
        "upper_95": confidence[:, 1],
        "unit": artifact["unit"],
        "model": artifact["version"],
    })
    return result