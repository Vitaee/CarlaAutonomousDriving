from __future__ import annotations
import argparse
import pathlib
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Threshold (in absolute steering angle) above which we consider a sample a turn
TURN_THRESHOLD: float = 0.05  # radians (adjust depending on your dataset)

# Aesthetic settings
sns.set_theme(style="darkgrid")


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def load_csv(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load the driving log into a :class:`pandas.DataFrame`."""
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    expected_cols = {
        "frame_filename",
        "steering_angle",
        "throttle",
        "brake",
        "speed_kmh",
        "camera_position",
        "frame_number",
        "timestamp",
    }
    missing_cols = expected_cols.difference(df.columns)
    if missing_cols:
        print(
            f"[ERROR] The following required columns are missing in the CSV: {', '.join(missing_cols)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Ensure proper dtypes
    df = df.astype(
        {
            "steering_angle": float,
            "throttle": float,
            "brake": float,
            "speed_kmh": float,
            "frame_number": int,
            "timestamp": float,
        }
    )
    return df


def compute_turn_distribution(df: pd.DataFrame) -> Tuple[int, int, int]:
    """Return counts of (left, right, straight) based on steering angle sign."""
    left = (df["steering_angle"] > TURN_THRESHOLD).sum()
    right = (df["steering_angle"] < -TURN_THRESHOLD).sum()
    straight = len(df) - left - right
    return left, right, straight


# -----------------------------------------------------------------------------
# Visualization functions
# -----------------------------------------------------------------------------

def plot_turn_pie(df: pd.DataFrame, ax: plt.Axes) -> None:
    left, right, straight = compute_turn_distribution(df)
    labels = [
        f"Left ({left / len(df):.1%})",
        f"Right ({right / len(df):.1%})",
        f"Straight ({straight / len(df):.1%})",
    ]
    ax.pie([left, right, straight], labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Turn Direction Distribution")


def plot_speed_time(df: pd.DataFrame, ax: plt.Axes) -> None:
    sns.lineplot(data=df, x="timestamp", y="speed_kmh", ax=ax, linewidth=1.0)
    ax.set_title("Speed over Time")
    ax.set_xlabel("Timestamp [s]")
    ax.set_ylabel("Speed [km/h]")


def plot_speed_hist(df: pd.DataFrame, ax: plt.Axes) -> None:
    sns.histplot(df["speed_kmh"], bins=30, kde=True, ax=ax)
    ax.set_title("Speed Distribution")
    ax.set_xlabel("Speed [km/h]")


def plot_steering_time(df: pd.DataFrame, ax: plt.Axes) -> None:
    sns.lineplot(data=df, x="timestamp", y="steering_angle", ax=ax, linewidth=1.0)
    ax.set_title("Steering Angle over Time")
    ax.set_xlabel("Timestamp [s]")
    ax.set_ylabel("Steering Angle [rad]")


def plot_steering_hist(df: pd.DataFrame, ax: plt.Axes) -> None:
    sns.histplot(df["steering_angle"], bins=30, kde=True, ax=ax)
    ax.set_title("Steering Angle Distribution")
    ax.set_xlabel("Steering Angle [rad]")


def plot_scatter_speed_vs_steering(df: pd.DataFrame, ax: plt.Axes) -> None:
    sns.scatterplot(
        data=df,
        x="steering_angle",
        y="speed_kmh",
        hue="camera_position",
        palette="viridis",
        ax=ax,
        s=20,
    )
    ax.set_title("Speed vs. Steering Angle")
    ax.set_xlabel("Steering Angle [rad]")
    ax.set_ylabel("Speed [km/h]")
    ax.legend(title="Camera")


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CARLA driving log CSV.")
    parser.add_argument("csv", type=pathlib.Path, help="Path to the driving_log.csv file")
    parser.add_argument(
        "--save", action="store_true", help="Save plots to disk (PNG) in addition to showing them"
    )
    args = parser.parse_args()

    df = load_csv(args.csv)

    # Create a figure grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # Flatten axes for easy indexing
    ax_list = axes.flatten()

    plot_turn_pie(df, ax_list[0])
    plot_speed_time(df, ax_list[1])
    plot_speed_hist(df, ax_list[2])
    plot_steering_time(df, ax_list[3])
    plot_steering_hist(df, ax_list[4])
    plot_scatter_speed_vs_steering(df, ax_list[5])

    plt.tight_layout()

    if args.save:
        out_dir = args.csv.parent / "plots"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{args.csv.stem}_visualization.png"
        fig.savefig(out_path, dpi=300)
        print(f"[INFO] Figure saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    main() 


"""
A simple utility to explore CARLA driving logs. It reads the CSV exported by the
collector (frame_filename, steering_angle, throttle, brake, speed_kmh, ...)
and generates:
  • Percentage of left / right / straight samples (pie chart)
  • Speed-over-time line plot and histogram
  • Steering-angle-over-time line plot and histogram
  • Scatter plot of Steering Angle vs. Speed

Usage
-----
python visualize_driving_log.py data/dataset_carla_001_Town01/steering_data.csv
python visualize_driving_log.py data_weathers/dataset_carla_001_Town01/steering_data.csv --save
The script will open interactive figures and also save PNG copies next to the
CSV file under a new directory named "plots".
"""
