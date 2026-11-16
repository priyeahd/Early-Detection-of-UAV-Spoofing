"""Export all audited PX4 merged flights using frozen Ocean and fresh test cohorts."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_lighthouse_ocean import export_subset


def assign_flights(inventory, pilot, ocean):
    data = inventory.copy()
    data["logged_at"] = data["flight_id"].map(lambda name: datetime.strptime(name, "log_%d_%m_%Y_%H_%M_%S"))
    previous = ocean.set_index("flight_id")
    if not data["flight_id"].is_unique or not data["sha256"].is_unique:
        raise ValueError("Duplicate source flight or content hash")
    if not data.set_index("flight_id").loc[previous.index, "sha256"].eq(previous["sha256"]).all():
        raise ValueError("Ocean sources differ from the full inventory")
    data["split"] = data["flight_id"].map(previous["split"]).fillna("train")
    data["test_cohort"] = np.where(data["split"].eq("test"), "ocean_replay", "")
    known_ids = set(pilot["flight_id"]) | set(ocean["flight_id"])
    ocean_ids = set(ocean["flight_id"])
    for _, group in data.groupby(["scenario", "condition"]):
        group = group.sort_values("logged_at")
        novel = [(position, index) for position, (index, flight) in enumerate(group["flight_id"].items())
                 if flight not in known_ids]
        fresh_count = 6
        blocks = [novel[start:start + fresh_count] for start in range(len(novel) - fresh_count + 1)]
        # Keep the fresh cohort clustered in the source order when possible.
        fresh = min(blocks, key=lambda block: (block[-1][0] - block[0][0], -block[-1][0]))
        fresh_indices = [index for _, index in fresh]
        data.loc[fresh_indices, ["split", "test_cohort"]] = ["test", "fresh"]
        remaining = group.loc[~group["flight_id"].isin(ocean_ids) & ~group.index.isin(fresh_indices)]
        extra_validation = remaining.iloc[np.linspace(0, len(remaining) - 1, 4, dtype=int)]
        data.loc[extra_validation.index, "split"] = "validation"
    expected = {"train": 252, "validation": 48, "test": 60}
    if data["split"].value_counts().to_dict() != expected:
        raise ValueError("Unexpected split sizes")
    if set(data.loc[data["test_cohort"].eq("fresh"), "flight_id"]) & known_ids:
        raise ValueError("Fresh cohort overlaps a past experiment")
    return data.sort_values("flight_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=Path("reports/px4_merged_inventory.csv"))
    args = parser.parse_args()
    source = args.dataset / "PX4_Simulation_Data"
    inventory = pd.read_csv(args.inventory)
    files = {file.relative_to(source).as_posix() for file in (source / "Merged").glob("*/*/*.csv")}
    if len(files) != 360 or files != set(inventory["file"]):
        raise ValueError("Expected all 360 merged files from the source audit")
    pilot = pd.read_csv(args.dataset / "lighthouse_subset/splits.csv")
    ocean = pd.read_csv(args.dataset / "lighthouse_ocean_subset/splits.csv")
    selected = assign_flights(inventory, pilot, ocean)
    export_subset(args.dataset, selected, args.dataset / "sail_1", "sail-1-v1", integrated=True)


if __name__ == "__main__":
    main()
