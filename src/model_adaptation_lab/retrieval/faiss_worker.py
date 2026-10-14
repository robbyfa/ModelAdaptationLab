"""CPU FAISS isolation: never import torch here (macOS OpenMP wheel conflict)."""

import importlib
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    np = importlib.import_module("numpy")
    faiss = importlib.import_module("faiss")
    faiss.omp_set_num_threads(1)
    mode, path = sys.argv[1:3]
    if mode == "build":
        raw, count, dimensions = sys.argv[3:6]
        index = faiss.IndexFlatIP(int(dimensions))
        values = np.memmap(raw, dtype="float32", mode="r", shape=(int(count), int(dimensions)))
        index.add(values)
        faiss.write_index(index, path)
        return
    index = faiss.read_index(str(Path(path)))
    print(json.dumps({"ready": True, "count": index.ntotal, "dimensions": index.d}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            query = np.ascontiguousarray([request["query"]], dtype="float32")
            if query.shape != (1, index.d) or not np.isfinite(query).all():
                raise ValueError("Invalid query vector")
            count = min(int(request["top_k"]), index.ntotal)
            size = min(count + 1, index.ntotal)
            scores, positions = index.search(query, size)
            while size < index.ntotal and scores[0, -1] >= scores[0, count - 1]:
                size = min(index.ntotal, size * 2)
                scores, positions = index.search(query, size)
            result: dict[str, Any] = {
                "scores": scores[0].tolist(),
                "positions": positions[0].tolist(),
            }
        except (ValueError, KeyError, RuntimeError) as exc:
            result = {"error": str(exc)}
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
