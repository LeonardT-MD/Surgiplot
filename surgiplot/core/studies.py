from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import isnan
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping
import json
import re


PROVENANCE_FIELDS = [
    "subject_id",
    "anatomical_target",
    "acquisition_modality",
    "operator",
    "session_id",
    "source_context",
    "reference_anatomy",
    "source_file",
    "notes",
]


RUN_SKIP_KEYS = {
    "metric_family",
    "reference_anatomy",
    "warnings",
    "entry_role_order",
    "ordered_entry_labels",
    "action",
    "table_node",
    "subject_id",
    "anatomical_target",
    "added_column_count",
    "row_count",
    "column_count",
    "dataset_kind",
    "file_kind",
    "path",
    "status",
    "appended_metric_family",
}


def normalize_target_key(anatomical_target: str) -> str:
    text = str(anatomical_target or "").strip().upper()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "TARGET"


def _stringify(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def flatten_metric_payload(payload: Mapping[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for key, value in payload.items():
        if key in RUN_SKIP_KEYS:
            continue
        result[str(key)] = _stringify(value)
    return result


@dataclass
class AcquisitionProvenance:
    subject_id: str
    anatomical_target: str
    acquisition_modality: str = ""
    operator: str = ""
    session_id: str = ""
    source_context: str = ""
    reference_anatomy: str = ""
    source_file: str = ""
    notes: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {
            "subject_id": str(self.subject_id).strip(),
            "anatomical_target": str(self.anatomical_target).strip(),
            "acquisition_modality": str(self.acquisition_modality).strip(),
            "operator": str(self.operator).strip(),
            "session_id": str(self.session_id).strip(),
            "source_context": str(self.source_context).strip(),
            "reference_anatomy": str(self.reference_anatomy).strip(),
            "source_file": str(self.source_file).strip(),
            "notes": str(self.notes).strip(),
        }

    @property
    def target_key(self) -> str:
        return normalize_target_key(self.anatomical_target)


@dataclass
class MetricRunRecord:
    metric_family: str
    provenance: AcquisitionProvenance
    result_fields: Dict[str, str] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def as_row(self) -> Dict[str, str]:
        row = {
            "Subject ID": self.provenance.subject_id,
            "Anatomical Target": self.provenance.anatomical_target,
            "Target Key": self.provenance.target_key,
            "Metric Family": str(self.metric_family).strip(),
            "Acquisition Modality": self.provenance.acquisition_modality,
            "Operator": self.provenance.operator,
            "Session ID": self.provenance.session_id,
            "Source Context": self.provenance.source_context,
            "Reference Anatomy": self.provenance.reference_anatomy,
            "Source File": self.provenance.source_file,
            "Notes": self.provenance.notes,
            "Created At (UTC)": self.created_at_utc,
        }
        row.update(self.result_fields)
        return row


def build_metric_run_record(
    subject_id: str,
    anatomical_target: str,
    metric_family: str,
    payload: Mapping[str, Any],
    *,
    acquisition_modality: str = "",
    operator: str = "",
    session_id: str = "",
    source_context: str = "",
    reference_anatomy: str = "",
    source_file: str = "",
    notes: str = "",
) -> MetricRunRecord:
    provenance = AcquisitionProvenance(
        subject_id=subject_id,
        anatomical_target=anatomical_target,
        acquisition_modality=acquisition_modality,
        operator=operator,
        session_id=session_id,
        source_context=source_context,
        reference_anatomy=reference_anatomy,
        source_file=source_file,
        notes=notes,
    )
    return MetricRunRecord(
        metric_family=metric_family,
        provenance=provenance,
        result_fields=flatten_metric_payload(payload),
    )


def comparison_rows(records: Iterable[MetricRunRecord]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for record in records:
        row = record.as_row()
        row["Comparison Group"] = " | ".join(
            [
                row.get("Subject ID", "").strip(),
                row.get("Anatomical Target", "").strip(),
                row.get("Metric Family", "").strip(),
            ]
        )
        rows.append(row)
    return rows


def repeatability_rows(records: Iterable[MetricRunRecord]) -> List[Dict[str, str]]:
    grouped: Dict[tuple[str, str, str], List[MetricRunRecord]] = {}
    for record in records:
        key = (
            record.provenance.subject_id.strip(),
            record.provenance.anatomical_target.strip(),
            record.metric_family.strip(),
        )
        grouped.setdefault(key, []).append(record)

    rows: List[Dict[str, str]] = []
    for (subject_id, anatomical_target, metric_family), group_records in grouped.items():
        field_names = sorted(
            {
                field
                for record in group_records
                for field in record.result_fields.keys()
            }
        )
        modalities = sorted(
            {
                record.provenance.acquisition_modality.strip()
                for record in group_records
                if record.provenance.acquisition_modality.strip()
            }
        )
        operators = sorted(
            {
                record.provenance.operator.strip()
                for record in group_records
                if record.provenance.operator.strip()
            }
        )
        sessions = sorted(
            {
                record.provenance.session_id.strip()
                for record in group_records
                if record.provenance.session_id.strip()
            }
        )
        for field_name in field_names:
            numeric_values: List[float] = []
            for record in group_records:
                raw = record.result_fields.get(field_name, "")
                try:
                    value = float(raw)
                except Exception:
                    continue
                if isnan(value):
                    continue
                numeric_values.append(value)
            if not numeric_values:
                continue
            n = len(numeric_values)
            field_mean = mean(numeric_values)
            field_sd = pstdev(numeric_values) if n > 1 else 0.0
            field_min = min(numeric_values)
            field_max = max(numeric_values)
            cv_percent = (field_sd / field_mean * 100.0) if abs(field_mean) > 1e-12 else 0.0
            rows.append(
                {
                    "Subject ID": subject_id,
                    "Anatomical Target": anatomical_target,
                    "Target Key": normalize_target_key(anatomical_target),
                    "Metric Family": metric_family,
                    "Field": field_name,
                    "Replicate Count": str(n),
                    "Mean": f"{field_mean:.6f}",
                    "SD": f"{field_sd:.6f}",
                    "CV_percent": f"{cv_percent:.6f}",
                    "Min": f"{field_min:.6f}",
                    "Max": f"{field_max:.6f}",
                    "Acquisition Modalities": ", ".join(modalities),
                    "Operators": ", ".join(operators),
                    "Sessions": ", ".join(sessions),
                }
            )
    return rows
