import argparse
import json
from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def load_cases(dataset_path: Path) -> list[dict]:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    return data["cases"]

def chunk_indices(total: int, chunk_size: int) -> list[list[int]]:
    return [list(range(i, min(i + chunk_size, total))) for i in range(0, total, chunk_size)]

def build_partitions(cases: list[dict], chunk_size: int, prefix: str) -> list[dict]:
    partitions = []
    chunks = chunk_indices(len(cases), chunk_size)
    for idx, indices in enumerate(chunks, start=1):
        partition_id = f"{prefix}{idx:02d}"
        partitions.append(
            {
                "partition_id": partition_id,
                "start_index": indices[0],
                "end_index": indices[-1],
                "size": len(indices),
                "case_indices": indices,
            }
        )
    return partitions

def save_partitions(
    *,
    partitions: list[dict],
    total_cases: int,
    requested_chunk_size: int,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_cases": total_cases,
        "requested_chunk_size": requested_chunk_size,
        "total_partitions": len(partitions),
        "partitions": partitions,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic case partitions")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root() / "data" / "processed" / "test_cases.json",
        help="Path to processed dataset",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="Number of adversarial cases per partition",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "data" / "processed" / "partitions.json",
        help="Output path for partition definition",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="part_",
        help="ID prefix for partitions",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    cases = load_cases(args.dataset)
    partitions = build_partitions(cases, args.chunk_size, args.prefix)
    save_partitions(
        partitions=partitions,
        total_cases=len(cases),
        requested_chunk_size=args.chunk_size,
        output_path=args.output,
    )
    print(
        f"Created {len(partitions)} partitions (chunk size {args.chunk_size}) at {args.output}"
    )

if __name__ == "__main__":
    main()
