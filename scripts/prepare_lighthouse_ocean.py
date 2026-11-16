"""Build the Lighthouse Ocean split and past-only features from audited merged logs."""

import argparse
import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BASE_FEATURES = {
    **{f"velocity_{axis}": "m/s" for axis in ("east", "north", "up")},
    **{f"linear_acceleration_{axis}": "m/s^2" for axis in ("x", "y", "z")},
}
HORIZONS = ((250_000, "250ms"), (1_000_000, "1s"), (2_000_000, "2s"))
CADENCE_US = 50_000
GAP_US = 250_000
FEATURES = {
    name: (unit, f"Current merged telemetry column {name}")
    for name, unit in BASE_FEATURES.items()
}
for _, suffix in HORIZONS:
    for axis in ("east", "north", "up"):
        FEATURES[f"gps_velocity_residual_{axis}_{suffix}"] = (
            "m/s", f"GPS position-derived velocity minus mean reported GPS velocity over {suffix}",
        )
        FEATURES[f"gps_velocity_change_{axis}_{suffix}"] = (
            "m/s", f"Change in reported GPS velocity over {suffix}",
        )
    FEATURES[f"altitude_change_residual_{suffix}"] = (
        "m", f"GPS altitude change minus pressure altitude change over {suffix}",
    )
    for axis in ("x", "y", "z"):
        FEATURES[f"imu_acceleration_std_{axis}_{suffix}"] = (
            "m/s^2", f"Standard deviation of IMU acceleration over the preceding {suffix}",
        )
INTEGRATED_FEATURES = {
    f"gps_displacement_error_{axis}_{scope}": (
        "m", f"Cumulative {axis} GPS position change minus integrated reported GPS velocity, {description}",
    )
    for axis in ("east", "north", "horizontal")
    for scope, description in (
        ("segment", "since the most recent gap over 250 ms"),
        ("observed", "summing observed intervals and skipping gaps over 250 ms"),
    )
}


def select_flights(inventory, pilot):
    available = inventory.loc[~inventory["flight_id"].isin(pilot["flight_id"])].copy()
    available["logged_at"] = available["flight_id"].map(
        lambda name: datetime.strptime(name, "log_%d_%m_%Y_%H_%M_%S")
    )
    selected = []
    for _, group in available.groupby(["scenario", "condition"]):
        group = group.sort_values("logged_at")
        if len(group) < 32:
            raise ValueError("Need 32 non-pilot candidate flights per stratum")
        selected.extend([
            group.iloc[:16].assign(split="train"),
            group.iloc[-12:-8].assign(split="validation"),
            group.iloc[-4:].assign(split="test"),
        ])
    result = pd.concat(selected).sort_values("flight_id")
    if len(result) != 144 or not result["flight_id"].is_unique or not result["sha256"].is_unique:
        raise ValueError("Expected 144 unique audited flights")
    return result


def extract_features(data, assignment, integrated=False):
    data = data.sort_values("time_usec", kind="stable").reset_index(drop=True)
    time = data["time_usec"].to_numpy(dtype=np.int64)
    if len(time) != assignment.rows or np.any(np.diff(time) <= 0):
        raise ValueError(f"Unexpected sample count or duplicate clock in {assignment.flight_id}")
    label = data["attack_enabled"]
    if not label.isin([0, 1]).all() or label.sum() != assignment.positive_rows:
        raise ValueError(f"Invalid labels in {assignment.flight_id}")
    elapsed = time - time[0]
    segment = np.cumsum(np.r_[False, np.diff(time) > GAP_US])
    # Keep the first observed row in each 50 ms bucket without interpolation.
    sampled = np.flatnonzero(np.r_[True, np.diff(elapsed // CADENCE_US) > 0])
    now = time[sampled]
    frame = {
        "flight_id": assignment.flight_id,
        "timestamp_s": elapsed[sampled] / 1e6,
        "attack_label": label.to_numpy(dtype=np.int8)[sampled],
        "split": assignment.split,
        **{name: data[name].to_numpy()[sampled] for name in BASE_FEATURES},
    }
    latitude = np.deg2rad(data["latitude_deg"].to_numpy())
    longitude = np.deg2rad(data["longitude_deg"].to_numpy())
    altitude = data["altitude"].to_numpy()
    pressure_altitude = data["pressure_altitude"].to_numpy()
    velocity = {axis: data[f"velocity_{axis}"].to_numpy() for axis in ("east", "north", "up")}
    acceleration = data[[f"linear_acceleration_{axis}" for axis in ("x", "y", "z")]]
    window = acceleration.set_axis(pd.to_timedelta(elapsed, unit="us"))
    sample_delta = np.diff(time, prepend=time[0]) / 1e6
    integrated_velocity = {
        axis: np.cumsum(sample_delta * (values + np.r_[values[0], values[:-1]]) / 2)
        for axis, values in velocity.items()
    }
    if integrated:
        longitude_step = (np.diff(longitude, prepend=longitude[0]) + np.pi) % (2 * np.pi) - np.pi
        east_step = 6_371_000 * np.cos((latitude + np.r_[latitude[0], latitude[:-1]]) / 2) * longitude_step
        north_step = 6_371_000 * np.diff(latitude, prepend=latitude[0])
        gaps = np.r_[True, np.diff(time) > GAP_US]
        anchors = np.maximum.accumulate(np.where(gaps, np.arange(len(time)), 0))
        for axis, displacement in (("east", east_step), ("north", north_step)):
            velocity_step = np.diff(integrated_velocity[axis], prepend=0)
            observed = np.cumsum(np.where(gaps, 0, displacement - velocity_step))
            frame[f"gps_displacement_error_{axis}_observed"] = observed[sampled]
            frame[f"gps_displacement_error_{axis}_segment"] = (
                observed[sampled] - observed[anchors[sampled]]
            )
        for scope in ("observed", "segment"):
            frame[f"gps_displacement_error_horizontal_{scope}"] = np.hypot(
                frame[f"gps_displacement_error_east_{scope}"],
                frame[f"gps_displacement_error_north_{scope}"],
            )
    for horizon, suffix in HORIZONS:
        previous = np.searchsorted(time, now - horizon, side="right") - 1
        idx = np.maximum(previous, 0)
        delta_us = now - time[idx]
        valid = ((previous >= 0) & (segment[idx] == segment[sampled])
                 & (delta_us <= horizon + CADENCE_US))
        delta_s = np.where(valid, delta_us / 1e6, np.nan)
        dlon = (longitude[sampled] - longitude[idx] + np.pi) % (2 * np.pi) - np.pi
        offsets = {
            "east": 6_371_000 * np.cos((latitude[sampled] + latitude[idx]) / 2) * dlon,
            "north": 6_371_000 * (latitude[sampled] - latitude[idx]),
            "up": altitude[sampled] - altitude[idx],
        }
        for axis, values in velocity.items():
            integral = integrated_velocity[axis]
            mean = (integral[sampled] - integral[idx]) / delta_s
            frame[f"gps_velocity_residual_{axis}_{suffix}"] = offsets[axis] / delta_s - mean
            frame[f"gps_velocity_change_{axis}_{suffix}"] = np.where(
                valid, values[sampled] - values[idx], np.nan,
            )
        frame[f"altitude_change_residual_{suffix}"] = np.where(
            valid, offsets["up"] - (pressure_altitude[sampled] - pressure_altitude[idx]), np.nan,
        )
        std = window.rolling(pd.Timedelta(microseconds=horizon), min_periods=4).std(ddof=0)
        for axis in ("x", "y", "z"):
            frame[f"imu_acceleration_std_{axis}_{suffix}"] = np.where(
                valid, std[f"linear_acceleration_{axis}"].to_numpy()[sampled], np.nan,
            )
    return pd.DataFrame(frame)


def prepare(dataset, inventory_path, output, integrated=False):
    inventory = pd.read_csv(inventory_path)
    pilot = pd.read_csv(dataset / "lighthouse_subset/splits.csv")
    selected = select_flights(inventory, pilot)
    version = "lighthouse-ocean-integrated-v1" if integrated else "lighthouse-ocean-v1"
    export_subset(dataset, selected, output, version, integrated=integrated)


def export_subset(dataset, selected, output, version, *, integrated):
    if output.exists():
        raise FileExistsError(output)
    exported = []
    with tempfile.TemporaryDirectory(prefix="px4-build-", dir=dataset.resolve()) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        with ((stage / "samples.csv").open("x", newline="", encoding="utf-8") as development,
              (stage / "test_samples.csv").open("x", newline="", encoding="utf-8") as test):
            for assignment in selected.itertuples(index=False):
                source = dataset / "PX4_Simulation_Data" / assignment.file
                with source.open("rb") as file:
                    if hashlib.file_digest(file, "sha256").hexdigest() != assignment.sha256:
                        raise ValueError(f"Source changed since audit: {source}")
                frame = extract_features(pd.read_csv(source), assignment, integrated=integrated)
                if frame["attack_label"].any() != (assignment.condition == "Attacked"):
                    raise ValueError(f"Sampling lost the attack interval for {assignment.flight_id}")
                destination = test if assignment.split == "test" else development
                frame.to_csv(destination, index=False, header=destination.tell() == 0, float_format="%.8g")
                entry = {
                    "flight_id": assignment.flight_id,
                    "split": assignment.split,
                    "scenario": assignment.scenario,
                    "condition": assignment.condition,
                    "source_path": source.relative_to(dataset).as_posix(),
                    "sha256": assignment.sha256,
                    "source_samples": assignment.rows,
                    "sample_count": len(frame),
                    "positive_samples": int(frame["attack_label"].sum()),
                    "attack_events": assignment.positive_segments,
                    "attack_onset_s": json.loads(assignment.segment_starts_s_from_first_sample)[0]
                    if assignment.positive_segments else "",
                }
                if hasattr(assignment, "test_cohort"):
                    entry["test_cohort"] = assignment.test_cohort
                exported.append(entry)
        pd.DataFrame(exported).to_csv(stage / "splits.csv", index=False)
        schema_features = FEATURES | INTEGRATED_FEATURES if integrated else FEATURES
        schema = {
            "version": version,
            "data_kind": "approved_telemetry",
            "features": [
                {"name": name, "unit": unit, "provenance": provenance, "causal": True}
                for name, (unit, provenance) in schema_features.items()
            ],
        }
        (stage / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
    print(pd.DataFrame(exported).groupby("split")[["sample_count", "positive_samples"]].agg(["count", "sum"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=Path("reports/px4_merged_inventory.csv"))
    parser.add_argument("--integrated", action="store_true", help="Export the cumulative GPS error variant")
    args = parser.parse_args()
    output = args.dataset / "lighthouse_ocean_subset"
    if args.integrated:
        output = output / "integrated"
    prepare(args.dataset, args.inventory, output, integrated=args.integrated)


if __name__ == "__main__":
    main()
