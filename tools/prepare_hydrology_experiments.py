"""Prepare MADE2 and North Loup hydrology data for NN-based discovery."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import savemat


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"D:\OneDrive - HHU\ML codes\Laplace-SR\public_version\dataset")
DEFAULT_MADE2_XLS = SOURCE_ROOT / "MADE2.xls"
DEFAULT_NORTH_LOUP_XLSX = SOURCE_ROOT / "North Loup River_data.xlsx"


@dataclass(frozen=True)
class Snapshot:
    time: float
    x: np.ndarray
    c: np.ndarray
    label: str


def _finite_positive_pairs(x: np.ndarray, c: np.ndarray, *, keep_zero: bool = True) -> tuple[np.ndarray, np.ndarray]:
    x_vals = np.asarray(x, dtype=float).reshape(-1)
    c_vals = np.asarray(c, dtype=float).reshape(-1)
    mask = np.isfinite(x_vals) & np.isfinite(c_vals)
    mask &= c_vals >= 0.0 if keep_zero else c_vals > 0.0
    x_vals = x_vals[mask]
    c_vals = c_vals[mask]
    if x_vals.size == 0:
        return x_vals, c_vals
    order = np.argsort(x_vals)
    x_vals = x_vals[order]
    c_vals = c_vals[order]
    unique_x, inverse = np.unique(x_vals, return_inverse=True)
    if unique_x.size != x_vals.size:
        summed = np.zeros_like(unique_x, dtype=float)
        counts = np.zeros_like(unique_x, dtype=float)
        np.add.at(summed, inverse, c_vals)
        np.add.at(counts, inverse, 1.0)
        c_vals = summed / np.maximum(counts, 1.0)
        x_vals = unique_x
    return x_vals, c_vals


class _OleWorkbookReader:
    """Tiny BIFF/OLE reader for the simple legacy MADE2 .xls workbook."""

    FREE = 0xFFFFFFFF
    END = 0xFFFFFFFE

    def __init__(self, path: Path):
        self.path = path
        self.data = path.read_bytes()

    def _parse_ole_stream(self, stream_name: str = "Workbook") -> bytes:
        data = self.data
        header = data[:512]
        if header[:8] != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            raise ValueError(f"{self.path} is not an OLE compound document")
        sector_size = 1 << struct.unpack_from("<H", header, 30)[0]
        mini_size = 1 << struct.unpack_from("<H", header, 32)[0]
        first_dir = struct.unpack_from("<I", header, 48)[0]
        mini_cutoff = struct.unpack_from("<I", header, 56)[0]
        first_mini_fat = struct.unpack_from("<I", header, 60)[0]
        num_fat = struct.unpack_from("<I", header, 44)[0]
        difat = list(struct.unpack_from("<109I", header, 76))
        fat_sectors = [value for value in difat if value not in {self.FREE, self.END}][:num_fat]

        def sector_bytes(sector: int) -> bytes:
            offset = 512 + sector * sector_size
            return data[offset : offset + sector_size]

        fat: list[int] = []
        for sector in fat_sectors:
            fat.extend(struct.unpack("<" + "I" * (sector_size // 4), sector_bytes(sector)))

        def chain(start: int) -> list[int]:
            out: list[int] = []
            current = start
            seen: set[int] = set()
            while current not in {self.FREE, self.END} and current < len(fat) and current not in seen:
                seen.add(current)
                out.append(current)
                current = fat[current]
            return out

        def stream_from_chain(start: int, size: int | None = None) -> bytes:
            payload = b"".join(sector_bytes(sector) for sector in chain(start))
            return payload if size is None else payload[:size]

        directory = stream_from_chain(first_dir)
        entries: list[dict[str, Any]] = []
        for offset in range(0, len(directory), 128):
            entry = directory[offset : offset + 128]
            if len(entry) < 128:
                break
            name_len = struct.unpack_from("<H", entry, 64)[0]
            if name_len < 2:
                continue
            name = entry[: name_len - 2].decode("utf-16le", errors="ignore")
            entries.append(
                {
                    "name": name,
                    "type": int(entry[66]),
                    "start": struct.unpack_from("<I", entry, 116)[0],
                    "size": struct.unpack_from("<Q", entry, 120)[0],
                }
            )
        root = next(item for item in entries if item["name"] == "Root Entry")
        mini_fat: list[int] = []
        if first_mini_fat not in {self.FREE, self.END}:
            for sector in chain(first_mini_fat):
                mini_fat.extend(struct.unpack("<" + "I" * (sector_size // 4), sector_bytes(sector)))
        mini_stream = stream_from_chain(root["start"], root["size"]) if root["start"] not in {self.FREE, self.END} else b""

        def mini_chain(start: int) -> list[int]:
            out: list[int] = []
            current = start
            seen: set[int] = set()
            while current not in {self.FREE, self.END} and current < len(mini_fat) and current not in seen:
                seen.add(current)
                out.append(current)
                current = mini_fat[current]
            return out

        def read_entry(entry: dict[str, Any]) -> bytes:
            if entry["size"] < mini_cutoff and entry["type"] != 5 and mini_stream:
                payload = b"".join(
                    mini_stream[sector * mini_size : (sector + 1) * mini_size]
                    for sector in mini_chain(entry["start"])
                )
                return payload[: entry["size"]]
            return stream_from_chain(entry["start"], entry["size"])

        workbook = next((item for item in entries if item["name"] in {stream_name, "Book"}), None)
        if workbook is None:
            raise ValueError(f"Could not locate Workbook stream in {self.path}")
        return read_entry(workbook)

    @staticmethod
    def _rk_value(rk: int) -> float:
        mult100 = rk & 0x01
        is_int = rk & 0x02
        value_bits = rk & 0xFFFFFFFC
        if is_int:
            value = value_bits >> 2
            if value & (1 << 29):
                value -= 1 << 30
            out = float(value)
        else:
            out = struct.unpack("<d", struct.pack("<II", 0, value_bits))[0]
        return out / 100.0 if mult100 else out

    @staticmethod
    def _decode_xl_string(record: bytes, offset: int) -> tuple[str, int]:
        char_count = struct.unpack_from("<H", record, offset)[0]
        offset += 2
        flags = record[offset]
        offset += 1
        is_utf16 = bool(flags & 0x01)
        rich_count = 0
        ext_size = 0
        if flags & 0x08:
            rich_count = struct.unpack_from("<H", record, offset)[0]
            offset += 2
        if flags & 0x04:
            ext_size = struct.unpack_from("<I", record, offset)[0]
            offset += 4
        byte_count = char_count * (2 if is_utf16 else 1)
        raw = record[offset : offset + byte_count]
        offset += byte_count + rich_count * 4 + ext_size
        return raw.decode("utf-16le" if is_utf16 else "latin1", errors="ignore"), offset

    def cells(self) -> dict[tuple[int, int], Any]:
        workbook = self._parse_ole_stream()
        strings: list[str] = []
        cells: dict[tuple[int, int], Any] = {}
        position = 0
        while position + 4 <= len(workbook):
            record_type, length = struct.unpack_from("<HH", workbook, position)
            record = workbook[position + 4 : position + 4 + length]
            position += 4 + length
            if record_type == 0x00FC:
                if len(record) < 8:
                    continue
                _, unique_count = struct.unpack_from("<II", record, 0)
                offset = 8
                strings = []
                for _ in range(unique_count):
                    if offset >= len(record):
                        break
                    text, offset = self._decode_xl_string(record, offset)
                    strings.append(text)
            elif record_type == 0x00FD and len(record) >= 10:
                row, col, _, index = struct.unpack_from("<HHHI", record, 0)
                cells[(int(row), int(col))] = strings[index] if index < len(strings) else ""
            elif record_type == 0x0203 and len(record) >= 14:
                row, col, _ = struct.unpack_from("<HHH", record, 0)
                cells[(int(row), int(col))] = float(struct.unpack_from("<d", record, 6)[0])
            elif record_type == 0x027E and len(record) >= 10:
                row, col, _, rk = struct.unpack_from("<HHHI", record, 0)
                cells[(int(row), int(col))] = self._rk_value(rk)
            elif record_type == 0x00BD and len(record) >= 6:
                row, first_col = struct.unpack_from("<HH", record, 0)
                last_col = struct.unpack_from("<H", record, len(record) - 2)[0]
                offset = 4
                for col in range(first_col, last_col + 1):
                    if offset + 6 > len(record):
                        break
                    _, rk = struct.unpack_from("<HI", record, offset)
                    offset += 6
                    cells[(int(row), int(col))] = self._rk_value(rk)
        return cells


def extract_made2(path: Path) -> list[Snapshot]:
    cells = _OleWorkbookReader(path).cells()
    max_row = max(row for row, _ in cells)
    snapshots: list[Snapshot] = []
    for col in sorted({col for row, col in cells if row == 0}):
        header = cells.get((0, col))
        if not isinstance(header, str) or not header.endswith("d-Y"):
            continue
        match = re.match(r"([0-9.]+)d-Y", header)
        if match is None:
            continue
        time = float(match.group(1))
        x_values: list[float] = []
        c_values: list[float] = []
        for row in range(3, max_row + 1):
            x_value = cells.get((row, col))
            c_value = cells.get((row, col + 1))
            if isinstance(x_value, (int, float)) and isinstance(c_value, (int, float)):
                x_values.append(float(x_value))
                c_values.append(float(c_value))
        x, c = _finite_positive_pairs(np.array(x_values), np.array(c_values), keep_zero=True)
        if x.size:
            snapshots.append(Snapshot(time=time, x=x, c=c, label=f"{time:g}d"))
    snapshots.sort(key=lambda item: item.time)
    return snapshots


def extract_north_loup(path: Path) -> list[Snapshot]:
    frame = pd.read_excel(path, sheet_name="1", header=None)
    snapshots: list[Snapshot] = []
    for col in range(frame.shape[1] - 2):
        label = frame.iat[1, col]
        if not isinstance(label, str) or "Digitize-X" not in label:
            continue
        match = re.search(r"([0-9.]+)h", label)
        if match is None:
            continue
        time = float(match.group(1))
        x_feet = pd.to_numeric(frame.iloc[7:, col], errors="coerce").to_numpy(float)
        c_raw = pd.to_numeric(frame.iloc[7:, col + 1], errors="coerce").to_numpy(float)
        x_meters, c = _finite_positive_pairs(0.3048 * x_feet, c_raw, keep_zero=True)
        if x_meters.size:
            snapshots.append(Snapshot(time=time, x=x_meters, c=c, label=f"{time:g}h"))
    snapshots.sort(key=lambda item: item.time)
    return snapshots


def _interpolate_snapshots(
    snapshots: list[Snapshot],
    *,
    grid_points: int,
    shift_to_zero: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not snapshots:
        raise ValueError("No snapshots were extracted")
    x_min = min(float(np.min(item.x)) for item in snapshots)
    x_max = max(float(np.max(item.x)) for item in snapshots)
    offset = x_min if shift_to_zero else 0.0
    lower = 0.0 if shift_to_zero else x_min
    upper = x_max - offset
    if upper <= lower:
        raise ValueError("Invalid spatial range after interpolation setup")
    x_grid = np.linspace(lower, upper, int(grid_points), dtype=float)
    c_grid = np.zeros((len(snapshots), x_grid.size), dtype=float)
    for row, item in enumerate(snapshots):
        x_values = item.x - offset
        c_grid[row, :] = np.interp(x_grid, x_values, item.c, left=0.0, right=0.0)
    return np.array([item.time for item in snapshots], dtype=float), x_grid, np.clip(c_grid, 0.0, None), offset


def _mass_normalize(c_grid: np.ndarray, dx: float) -> np.ndarray:
    out = np.asarray(c_grid, dtype=float).copy()
    for row in range(out.shape[0]):
        mass = float(np.sum(np.clip(out[row, :], 0.0, None)) * dx)
        if mass > 0.0 and math.isfinite(mass):
            out[row, :] = out[row, :] / mass
    return out


def _recommended_counts(total_samples: int) -> tuple[int, int]:
    train_points = max(1, int(0.75 * total_samples))
    val_points = max(1, total_samples - train_points)
    return train_points, val_points


def _write_variant(
    *,
    experiment: str,
    variant: str,
    snapshots: list[Snapshot],
    times: np.ndarray,
    x_grid: np.ndarray,
    c_grid: np.ndarray,
    x_offset: float,
    output_root: Path,
    time_unit: str,
    source_path: Path,
) -> dict[str, Any]:
    out_dir = output_root / experiment / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    dx = float(np.median(np.diff(x_grid)))
    c_values = _mass_normalize(c_grid, dx) if variant == "massnorm" else c_grid.copy()
    holdout_index = int(len(times) - 1)
    train_mask = np.ones(times.shape, dtype=bool)
    train_mask[holdout_index] = False
    train_t = times[train_mask]
    train_c = c_values[train_mask, :]
    all_mat = out_dir / f"{experiment}_{variant}_all.mat"
    train_mat = out_dir / f"{experiment}_{variant}_train.mat"
    full_npz = out_dir / f"{experiment}_{variant}_full.npz"
    metadata_json = out_dir / f"{experiment}_{variant}_metadata.json"
    base_payload = {
        "x": x_grid.reshape(1, -1),
        "t": times.reshape(1, -1),
        "c": c_values,
        "Exact": c_values,
        "original_x_offset": np.array([[x_offset]], dtype=float),
    }
    train_payload = dict(base_payload)
    train_payload.update({"t": train_t.reshape(1, -1), "c": train_c, "Exact": train_c})
    savemat(all_mat, base_payload)
    savemat(train_mat, train_payload)
    total_train_samples = int(train_c.size)
    train_points, val_points = _recommended_counts(total_train_samples)
    train_span = float(train_t[-1] - train_t[0])
    discovery_t_step = train_span / 100.0 if train_span > 0.0 else 1.0
    fit_x_margin = 0.05 * float(x_grid[-1] - x_grid[0])
    fit_t_margin = 0.05 * train_span
    metadata = {
        "experiment": experiment,
        "variant": variant,
        "source_path": str(source_path),
        "snapshot_labels": [item.label for item in snapshots],
        "snapshot_times": [float(value) for value in times],
        "train_times": [float(value) for value in train_t],
        "holdout_time": float(times[holdout_index]),
        "time_unit": time_unit,
        "x_unit": "m_shifted" if x_offset != 0.0 else "m",
        "original_x_offset": float(x_offset),
        "grid_shape_all": list(c_values.shape),
        "grid_shape_train": list(train_c.shape),
        "x_min": float(x_grid[0]),
        "x_max": float(x_grid[-1]),
        "x_step": dx,
        "discovery_x_max_exclusive": float(x_grid[-1] + 0.5 * dx),
        "discovery_t_min": float(train_t[0]),
        "discovery_t_max_exclusive": float(train_t[-1] + 0.5 * discovery_t_step),
        "discovery_t_step": float(discovery_t_step),
        "fit_x_min": float(x_grid[0] + fit_x_margin),
        "fit_x_max": float(x_grid[-1] - fit_x_margin),
        "fit_t_min": float(train_t[0] + fit_t_margin),
        "fit_t_max": float(train_t[-1]),
        "laplace_s_min": float(max(1.0e-4, 6.0 / max(float(train_t[-1]), np.finfo(float).eps))),
        "laplace_s_max": float(6.0 / max(float(train_t[0]), np.finfo(float).eps)),
        "train_points": train_points,
        "val_points": val_points,
        "train_mat": str(train_mat),
        "all_mat": str(all_mat),
        "full_npz": str(full_npz),
        "metadata_json": str(metadata_json),
    }
    np.savez_compressed(
        full_npz,
        x=x_grid,
        t=times,
        c=c_values,
        train_mask=train_mask,
        holdout_index=np.array(holdout_index),
        metadata=np.array(json.dumps(metadata), dtype=object),
    )
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def prepare_experiment(
    *,
    experiment: str,
    snapshots: list[Snapshot],
    output_root: Path,
    grid_points: int,
    shift_to_zero: bool,
    time_unit: str,
    source_path: Path,
) -> list[dict[str, Any]]:
    times, x_grid, c_grid, x_offset = _interpolate_snapshots(
        snapshots,
        grid_points=grid_points,
        shift_to_zero=shift_to_zero,
    )
    return [
        _write_variant(
            experiment=experiment,
            variant=variant,
            snapshots=snapshots,
            times=times,
            x_grid=x_grid,
            c_grid=c_grid,
            x_offset=x_offset,
            output_root=output_root,
            time_unit=time_unit,
            source_path=source_path,
        )
        for variant in ("raw", "massnorm")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--made2-xls", type=Path, default=DEFAULT_MADE2_XLS)
    parser.add_argument("--north-loup-xlsx", type=Path, default=DEFAULT_NORTH_LOUP_XLSX)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "hydrology_experiments")
    parser.add_argument("--made2-grid-points", type=int, default=220)
    parser.add_argument("--north-loup-grid-points", type=int, default=240)
    parser.add_argument(
        "--experiment",
        choices=("all", "made2", "north_loup"),
        default="all",
        help="select one experiment or prepare both",
    )
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    if args.experiment in {"all", "made2"}:
        made2 = extract_made2(args.made2_xls)
        summaries.extend(
            prepare_experiment(
                experiment="made2",
                snapshots=made2,
                output_root=args.output_root,
                grid_points=args.made2_grid_points,
                shift_to_zero=True,
                time_unit="d",
                source_path=args.made2_xls,
            )
        )
    if args.experiment in {"all", "north_loup"}:
        north_loup = extract_north_loup(args.north_loup_xlsx)
        summaries.extend(
            prepare_experiment(
                experiment="north_loup",
                snapshots=north_loup,
                output_root=args.output_root,
                grid_points=args.north_loup_grid_points,
                shift_to_zero=False,
                time_unit="h",
                source_path=args.north_loup_xlsx,
            )
        )

    summary_path = args.output_root / "hydrology_preparation_summary.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"prepared {len(summaries)} dataset variants")
    for item in summaries:
        print(
            f"{item['experiment']} {item['variant']}: snapshots={len(item['snapshot_times'])} "
            f"train={item['grid_shape_train']} holdout={item['holdout_time']}{item['time_unit']} "
            f"mat={item['train_mat']}"
        )
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
