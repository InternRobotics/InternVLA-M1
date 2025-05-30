import os
from pathlib import Path
from collections import defaultdict

TARGET_PREFIXES = ["Close", "Open", "Move", "Pick"]

def extract_task_type(path: Path):
    """
    Traverse upward to find a folder with name matching one of the task types.
    """
    for part in path.parts[::-1]:  # reverse to start from leaf upwards
        for prefix in TARGET_PREFIXES:
            if part.startswith(prefix) and "InScene" in part:
                return prefix
    return None

def count_videos_in_dir(dir_path: Path):
    """Count .mp4 and success_*.mp4 files directly in dir_path."""
    all_mp4 = list(dir_path.glob("*.mp4"))
    success_mp4 = [f for f in all_mp4 if f.name.startswith("success_")]
    return len(all_mp4), len(success_mp4)

def task_stats(root: Path):
    stats = defaultdict(lambda: {"total": 0, "success": 0})

    for dirpath, _, _ in os.walk(root):
        path = Path(dirpath)
        total, success = count_videos_in_dir(path)

        if total > 0:
            task_type = extract_task_type(path)
            if task_type:
                stats[task_type]["total"] += total
                stats[task_type]["success"] += success

    # Output
    for task in TARGET_PREFIXES:
        total = stats[task]["total"]
        success = stats[task]["success"]
        ratio = (success / total) if total > 0 else 0
        print(f"[{task}] - total: {total}, success: {success}, success rate: {ratio:.2%}")

    overall_total = sum(v["total"] for v in stats.values())
    overall_success = sum(v["success"] for v in stats.values())
    print("\n=== Overall Summary (filtered) ===")
    print(f"Total .mp4 files      : {overall_total}")
    print(f"Total success videos  : {overall_success}")
    print(f"Overall success ratio : {overall_success / overall_total:.2%}" if overall_total > 0 else "No .mp4 files found.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", help="Root directory to search")
    args = parser.parse_args()

    task_stats(Path(args.root_dir).resolve())
