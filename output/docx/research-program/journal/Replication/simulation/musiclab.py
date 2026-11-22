"""Score next downloads in MusicLab Experiment 1. See the package README."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def load_world(path, world):
    songs = []
    skipped = 0
    previous = ""
    with path.open(encoding="ascii") as source:
        for row in csv.reader(line for line in source if line.strip() and not line.startswith("%")):
            assert len(row) == 51 and int(row[1]) == world
            assert row[-1] >= previous
            previous = row[-1]
            flags = [int(x) for x in row[2:-1]]
            assert set(flags) <= {0, 1} and sum(flags) <= 1
            if sum(flags):
                songs.append(flags.index(1))
            else:
                skipped += 1
    return np.asarray(songs), skipped


def design(songs):
    counts = np.zeros(48)
    history = []
    for song in songs:
        history.append(np.log1p(counts))
        counts[song] += 1
    return np.asarray(history)


def score(songs, history, weights, alpha):
    logits = np.log(weights) + alpha * history
    probs = np.exp(logits - np.logaddexp.reduce(logits, axis=1)[:, None])
    log_score = np.log(probs[np.arange(len(songs)), songs]).sum()
    budget = (1 - probs.max(axis=1)).sum()
    return float(log_score), float(budget)


def analyze(directory):
    worlds, sources = {}, {}
    for world in range(1, 10):
        path = directory / f"world{world}.txt"
        worlds[world], skipped = load_world(path, world)
        sources[world] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "downloads": len(worlds[world]), "unmarked_rows": skipped}
    baseline = np.bincount(worlds[9], minlength=48) + 1
    designs = {w: design(worlds[w]) for w in range(1, 9)}
    grid = np.arange(21) / 20
    alpha = float(max(grid, key=lambda a: score(worlds[1], designs[1], baseline, a)[0]))
    scores = []
    for w in range(1, 9):
        fitted, budget = score(worlds[w], designs[w], baseline, alpha)
        fixed, fixed_budget = score(worlds[w], designs[w], baseline, 0)
        scores.append({"world": w, "n": len(worlds[w]), "score_gain_nats": fitted - fixed,
                       "fitted_budget": budget, "fixed_budget": fixed_budget})
    return {"source_files": sources, "alpha_world1": alpha, "world_scores": scores,
            "held_out_downloads": sum(r["n"] for r in scores[1:]),
            "held_out_score_gain": sum(r["score_gain_nats"] for r in scores[1:]),
            "published_experiment_1_listens": {"social": 21971, "independent": 5394},
            "published_experiment_1_downloads": {"social": 6626, "independent": 1578,
                                                 "reported_all": 8203}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing world1.txt to world9.txt")
    args = parser.parse_args()
    result = analyze(args.directory)
    target = Path(__file__).with_name("musiclab_results.json")
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "source_files"}, indent=2))
