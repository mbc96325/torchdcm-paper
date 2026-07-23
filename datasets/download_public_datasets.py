from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
REGISTRY_PATH = ROOT / "open_choice_benchmark_registry.csv"
RAW_DIR = ROOT / "raw"
MANIFEST_PATH = ROOT / "dataset_manifest.json"
R_EXPORT_SCRIPT = ROOT / "export_r_package_datasets.R"

BIOGEME_DIRECT_DATASETS = {
    "biogeme_airline_itinerary": "https://transp-or.epfl.ch/data/airline.dat",
    "biogeme_netherlands_mode": "https://transp-or.epfl.ch/data/netherlands.dat",
    "biogeme_switzerland_mode": "https://transp-or.epfl.ch/data/optima.dat",
    "biogeme_parking_spain": "https://transp-or.epfl.ch/data/parking.dat",
    "biogeme_telephone": "https://transp-or.epfl.ch/data/telephone.dat",
    "lpmc_london": "https://transp-or.epfl.ch/data/lpmc.dat",
}

LARGE_SURVEY_DATASETS = {
    "nhts_2017": {
        "source_url": "https://nhts.ornl.gov/",
        "status": "manual_large_pending",
        "title": "U.S. National Household Travel Survey 2017",
        "scope": "U.S. household/person/trip travel diary survey.",
        "download_note": "Use the NHTS official site to obtain household, person, trip, vehicle, and codebook files.",
    },
    "nhts_2022": {
        "source_url": "https://nhts.ornl.gov/",
        "status": "manual_large_pending",
        "title": "U.S. National Household Travel Survey 2022",
        "scope": "Most recent NHTS-style U.S. household/person/trip survey candidate.",
        "download_note": "Use the NHTS official site; confirm release file names and terms before adding raw archives.",
    },
    "uk_national_travel_survey": {
        "source_url": "https://www.gov.uk/government/collections/national-travel-survey-statistics",
        "status": "manual_large_pending",
        "title": "England National Travel Survey",
        "scope": "Household interview plus 7-day travel diary for residents of England.",
        "download_note": "Use GOV.UK tables for aggregate checks and UK Data Service or official access path for microdata.",
    },
    "psrc_household_travel_survey": {
        "source_url": "https://www.psrc.org/our-work/household-travel-survey-program",
        "status": "manual_large_pending",
        "title": "Puget Sound Regional Council Household Travel Survey",
        "scope": "Central Puget Sound household travel surveys with recent 2017, 2019, 2021, and 2023 waves.",
        "download_note": "Use PSRC Data Portal for recent waves and official zip/codebook files for older waves.",
    },
    "mtsa_us_metro_archive": {
        "source_url": "https://www.nrel.gov/transportation/secure-transportation-data/tsdc-metropolitan-travel-survey-archive",
        "status": "manual_large_pending",
        "title": "U.S. Metropolitan Travel Survey Archive",
        "scope": "Collection of household travel surveys from U.S. public agencies and metropolitan areas.",
        "download_note": "Select individual city/year surveys before downloading; preserve documentation and metadata with each survey.",
    },
    "germany_mid": {
        "source_url": "https://www.mobilitaet-in-deutschland.de/",
        "status": "manual_large_pending",
        "title": "Mobility in Germany (MiD)",
        "scope": "Nationwide German household travel behavior survey with 2002, 2008, 2017, and 2023 waves.",
        "download_note": "Access data through the MobilityData-Campus/BASt process and record usage terms before raw import.",
    },
    "covid_future_panel_survey": {
        "source_url": "https://arxiv.org/abs/2208.12618",
        "status": "manual_large_pending",
        "title": "COVID Future Panel Survey",
        "scope": "U.S. longitudinal survey on travel-related behavior and attitudes before/during/after COVID-19.",
        "download_note": "Find and verify the official public data repository before adding raw files.",
    },
}


@dataclass
class DatasetRecord:
    dataset_id: str
    status: str
    source_family: str = ""
    dataset_name: str = ""
    access_method: str = ""
    upstream_url: str = ""
    license_or_terms: str = ""
    files: list[dict[str, Any]] = field(default_factory=list)
    rows: int | None = None
    columns: int | None = None
    message: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "source_family": self.source_family,
            "dataset_name": self.dataset_name,
            "access_method": self.access_method,
            "upstream_url": self.upstream_url,
            "license_or_terms": self.license_or_terms,
            "rows": self.rows,
            "columns": self.columns,
            "files": self.files,
            "message": self.message,
            "notes": self.notes,
        }


def read_registry() -> dict[str, dict[str, str]]:
    with REGISTRY_PATH.open(newline="", encoding="utf-8") as f:
        return {row["dataset_id"]: row for row in csv.DictReader(f)}


def enrich(record: DatasetRecord, registry: dict[str, dict[str, str]]) -> DatasetRecord:
    row = registry.get(record.dataset_id, {})
    record.source_family = row.get("source_family", record.source_family)
    record.dataset_name = row.get("dataset_name", record.dataset_name)
    record.access_method = row.get("access_method", record.access_method)
    record.upstream_url = row.get("upstream_url", record.upstream_url)
    record.license_or_terms = row.get("license_or_terms", record.license_or_terms)
    return record


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: Path, root: Path = ROOT) -> dict[str, Any]:
    info = {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        info["rows"] = int(len(frame))
        info["columns"] = int(len(frame.columns))
    elif path.suffix.lower() in {".tsv", ".dat"}:
        frame = pd.read_csv(path, sep="\t")
        info["rows"] = int(len(frame))
        info["columns"] = int(len(frame.columns))
    return info


def read_delimited(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        pass
    for sep in ("\t", ";", ",", r"\s+"):
        try:
            return pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            continue
    raise ValueError(f"Could not parse delimited data file: {path}")


def write_metadata(dataset_id: str, payload: dict[str, Any]) -> None:
    out_dir = RAW_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def record_from_files(dataset_id: str, status: str, files: list[Path], message: str = "", notes: dict[str, Any] | None = None) -> DatasetRecord:
    described = [describe_file(path) for path in files if path.exists()]
    rows = described[0].get("rows") if described else None
    columns = described[0].get("columns") if described else None
    record = DatasetRecord(
        dataset_id=dataset_id,
        status=status,
        files=described,
        rows=rows,
        columns=columns,
        message=message,
        notes=notes or {},
    )
    write_metadata(dataset_id, record.to_dict())
    return record


def export_biogeme_swissmetro() -> DatasetRecord:
    dataset_id = "biogeme_swissmetro"
    out_dir = RAW_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import biogeme.data.swissmetro as swissmetro
    except ImportError as exc:
        return DatasetRecord(dataset_id=dataset_id, status="missing", message=f"Biogeme import failed: {exc}")

    source = Path(swissmetro.__file__).resolve().parent / "data" / "swissmetro.dat"
    if not source.exists():
        return DatasetRecord(dataset_id=dataset_id, status="missing", message=f"Swissmetro source file not found: {source}")
    raw_tsv = out_dir / "data.tsv"
    normalized_csv = out_dir / "data.csv"
    shutil.copyfile(source, raw_tsv)
    pd.read_csv(raw_tsv, sep="\t").to_csv(normalized_csv, index=False)
    return record_from_files(
        dataset_id,
        "downloaded",
        [normalized_csv, raw_tsv],
        notes={"python_package": "biogeme.data.swissmetro", "source_file": str(source)},
    )


def export_biogeme_optima() -> DatasetRecord:
    dataset_id = "biogeme_optima"
    out_dir = RAW_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from biogeme.data.optima import read_data
    except ImportError as exc:
        return DatasetRecord(dataset_id=dataset_id, status="missing", message=f"Biogeme import failed: {exc}")

    try:
        database = read_data()
        frame = database.dataframe.copy().reset_index(drop=True)
    except Exception as exc:  # pragma: no cover - depends on external package data
        return DatasetRecord(dataset_id=dataset_id, status="failed", message=f"Optima loader failed: {exc}")

    data_path = out_dir / "data.csv"
    frame.to_csv(data_path, index=False)
    return record_from_files(
        dataset_id,
        "downloaded",
        [data_path],
        notes={"python_package": "biogeme.data.optima", "source_loader": "read_data().dataframe"},
    )


def export_biogeme_mdcev() -> DatasetRecord:
    dataset_id = "biogeme_mdcev"
    out_dir = RAW_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import biogeme.data.mdcev_data as mdcev_data
        from biogeme.data.mdcev_data import read_data
    except ImportError as exc:
        return DatasetRecord(dataset_id=dataset_id, status="missing", message=f"Biogeme import failed: {exc}")

    try:
        database = read_data()
        frame = database.dataframe.copy().reset_index(drop=True)
    except Exception as exc:  # pragma: no cover - depends on external package data
        source = Path(mdcev_data.__file__).resolve().parent / "data" / "mdcev.csv"
        if not source.exists():
            return DatasetRecord(dataset_id=dataset_id, status="failed", message=f"MDCEV loader failed and raw CSV was not found: {exc}")
        raw_path = out_dir / "data.csv"
        shutil.copyfile(source, raw_path)
        return record_from_files(
            dataset_id,
            "downloaded",
            [raw_path],
            message=f"Biogeme read_data() failed, so the package raw CSV was copied directly: {exc}",
            notes={"python_package": "biogeme.data.mdcev_data", "source_file": str(source)},
        )

    data_path = out_dir / "data.csv"
    frame.to_csv(data_path, index=False)
    return record_from_files(
        dataset_id,
        "downloaded",
        [data_path],
        notes={"python_package": "biogeme.data.mdcev_data", "source_loader": "read_data().dataframe"},
    )


def download_direct_biogeme_dataset(dataset_id: str, url: str) -> DatasetRecord:
    out_dir = RAW_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "data.dat"
    csv_path = out_dir / "data.csv"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "torchdcm-benchmark-dataset-downloader"})
        with urllib.request.urlopen(request, timeout=120) as response:
            raw_path.write_bytes(response.read())
    except Exception as exc:
        return DatasetRecord(dataset_id=dataset_id, status="failed", message=f"Download failed from {url}: {exc}")

    files = [raw_path]
    message = ""
    try:
        frame = read_delimited(raw_path)
        frame.to_csv(csv_path, index=False)
        files.insert(0, csv_path)
    except Exception as exc:
        message = f"Raw file downloaded, but automatic CSV normalization failed: {exc}"
    return record_from_files(
        dataset_id,
        "downloaded",
        files,
        message=message,
        notes={"source_url": url, "source_page": "https://biogeme.epfl.ch/#data"},
    )


def run_r_exports() -> list[DatasetRecord]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return [DatasetRecord(dataset_id="r_package_exports", status="missing", message="Rscript not found.")]
    env = os.environ.copy()
    r_user_lib = str(Path.home() / "R" / "site-library")
    existing = env.get("R_LIBS_USER")
    env["R_LIBS_USER"] = r_user_lib if not existing else f"{r_user_lib}:{existing}"
    proc = subprocess.run(
        [rscript, str(R_EXPORT_SCRIPT), str(RAW_DIR)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return [DatasetRecord(dataset_id="r_package_exports", status="failed", message=(proc.stderr or proc.stdout).strip())]

    manifest = RAW_DIR / "_r_export_manifest.csv"
    if not manifest.exists():
        return [DatasetRecord(dataset_id="r_package_exports", status="failed", message="R export manifest was not created.")]
    r_rows = pd.read_csv(manifest).fillna("")
    records: list[DatasetRecord] = []
    for _, row in r_rows.iterrows():
        dataset_id = str(row["dataset_id"])
        status = str(row["status"])
        data_path = RAW_DIR / dataset_id / "data.csv"
        if status == "downloaded" and data_path.exists():
            records.append(
                record_from_files(
                    dataset_id,
                    "downloaded",
                    [data_path],
                    notes={"r_package": str(row["source_package"]), "r_object": str(row["object_name"])},
                )
            )
        else:
            records.append(
                DatasetRecord(
                    dataset_id=dataset_id,
                    status=status,
                    message=str(row.get("message", "")),
                    notes={"r_package": str(row.get("source_package", "")), "r_object": str(row.get("object_name", ""))},
                )
            )
    return records


def create_mirror_record(dataset_id: str, source_dataset_id: str) -> DatasetRecord:
    out_dir = RAW_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    note = (
        f"This dataset is represented by canonical raw data from {source_dataset_id}. "
        "The upstream xlogit examples use the same classic choice-data family; "
        "estimator-specific preprocessing should live in benchmarks/."
    )
    (out_dir / "README.md").write_text(note + "\n", encoding="utf-8")
    record = DatasetRecord(
        dataset_id=dataset_id,
        status="mirrored",
        files=[describe_file(out_dir / "README.md")],
        message=note,
        notes={"canonical_dataset_id": source_dataset_id},
    )
    write_metadata(dataset_id, record.to_dict())
    return record


def create_large_manual_record(dataset_id: str, config: dict[str, str]) -> DatasetRecord:
    out_dir = RAW_DIR / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)
    text = f"""# {config["title"]} Manual Large Download

Source: {config["source_url"]}

Scope: {config["scope"]}

Download note: {config["download_note"]}

Keep this dataset out of the estimator-parity benchmark until we define:

- household/person/trip table selection;
- mode-choice observation unit;
- choice-set construction;
- level-of-service enrichment;
- survey weights and filtering protocol.

After those choices are fixed, place immutable raw archives in this folder,
record the exact source version, and rerun
`python datasets/download_public_datasets.py --manifest-only`.
"""
    readme = out_dir / "MANUAL_DOWNLOAD.md"
    readme.write_text(text, encoding="utf-8")
    record = DatasetRecord(
        dataset_id=dataset_id,
        status=config["status"],
        files=[describe_file(readme)],
        message="Large public survey. Download deferred until benchmark preprocessing protocol is fixed.",
        notes={key: value for key, value in config.items() if key != "status"},
    )
    write_metadata(dataset_id, record.to_dict())
    return record


def rebuild_records_from_disk(registry: dict[str, dict[str, str]]) -> list[DatasetRecord]:
    records = []
    for dataset_id in registry:
        out_dir = RAW_DIR / dataset_id
        if not out_dir.exists():
            records.append(DatasetRecord(dataset_id=dataset_id, status="missing", message="Dataset directory not found."))
            continue
        files = sorted(path for path in out_dir.iterdir() if path.is_file() and path.name != "metadata.json")
        if not files:
            records.append(DatasetRecord(dataset_id=dataset_id, status="missing", message="No data or note files found."))
            continue
        status = "downloaded" if any(path.name == "data.csv" for path in files) else "documented"
        records.append(record_from_files(dataset_id, status, files))
    return [enrich(record, registry) for record in records]


def write_manifest(records: list[DatasetRecord], registry: dict[str, dict[str, str]]) -> None:
    by_id = {record.dataset_id: record for record in records}
    for dataset_id in registry:
        if dataset_id not in by_id:
            by_id[dataset_id] = DatasetRecord(dataset_id=dataset_id, status="missing", message="No downloader/exporter result.")
    enriched = [enrich(by_id[dataset_id], registry) for dataset_id in registry]
    extras = [enrich(record, registry) for dataset_id, record in sorted(by_id.items()) if dataset_id not in registry]
    payload = {
        "registry": str(REGISTRY_PATH.relative_to(REPO_ROOT)),
        "raw_dir": str(RAW_DIR.relative_to(REPO_ROOT)),
        "datasets": [record.to_dict() for record in [*enriched, *extras]],
        "summary": {
            "downloaded": sum(record.status == "downloaded" for record in [*enriched, *extras]),
            "mirrored": sum(record.status == "mirrored" for record in [*enriched, *extras]),
            "manual_large_pending": sum(record.status == "manual_large_pending" for record in [*enriched, *extras]),
            "missing_or_failed": sum(record.status in {"missing", "failed"} for record in [*enriched, *extras]),
        },
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download/export public benchmark datasets for TorchDCM validation.")
    parser.add_argument("--manifest-only", action="store_true", help="Only rebuild the manifest from existing raw files.")
    args = parser.parse_args(argv)

    registry = read_registry()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        records = rebuild_records_from_disk(registry)
    else:
        records = [
            export_biogeme_swissmetro(),
            export_biogeme_optima(),
            export_biogeme_mdcev(),
            *[
                download_direct_biogeme_dataset(dataset_id, url)
                for dataset_id, url in BIOGEME_DIRECT_DATASETS.items()
            ],
            *run_r_exports(),
            create_mirror_record("xlogit_electricity", "mlogit_electricity"),
            create_mirror_record("xlogit_fishing", "mlogit_fishing"),
            *[
                create_large_manual_record(dataset_id, config)
                for dataset_id, config in LARGE_SURVEY_DATASETS.items()
            ],
        ]
    write_manifest(records, registry)

    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    for item in payload["datasets"]:
        print(f"{item['dataset_id']}: {item['status']} rows={item['rows']} cols={item['columns']}")
        if item["message"] and item["status"] != "downloaded":
            print(f"  {item['message']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
