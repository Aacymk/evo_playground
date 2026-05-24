"""
tests/test_logging.py

Logging consistency tests — verify that what gets written to CSV and
checkpoints accurately reflects the world state at the time of writing.

1. CSV columns match CSV_COLUMNS exactly
2. alive_population in each row matches actual alive count at that frame
3. total_born is monotonically increasing across rows
4. total_deaths is monotonically increasing across rows
5. fitness values are in [0, 1]
6. Behavioral archetype percentages sum to 1.0 per row
7. Checkpoint .npz loads correctly and contains valid weight shapes
8. No duplicate frames in the CSV
9. LOG_INTERVAL is respected (rows written at correct frame numbers)
10. best_alive_fitness >= average_alive_fitness in every row
"""

import sys, os, csv, glob, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import seed
import numpy as np


def _run_with_logging(frames=1000, log_interval=200, checkpoint_interval=5,
                      run_id="test_run", tmpdir=None):
    """Run the sim for N frames with logging enabled. Returns (world, logger)."""
    import config
    from tests.conftest import fast_config, restore_config

    # Save everything we're about to touch
    orig = fast_config()
    orig['LOG_INTERVAL']        = config.LOG_INTERVAL
    orig['CHECKPOINT_INTERVAL'] = config.CHECKPOINT_INTERVAL

    config.LOG_INTERVAL        = log_interval
    config.CHECKPOINT_INTERVAL = checkpoint_interval

    from evolog.generation_logger import GenerationLogger
    from simulation.world import World

    seed(0)
    logger = GenerationLogger(run_id=run_id, runs_dir=tmpdir)
    world  = World(logger=logger)
    for _ in range(frames):
        world.update()

    restore_config(orig)
    return world, logger


def test_csv_columns_match_spec():
    """Every column in CSV_COLUMNS appears in the written file, no extras."""
    from evolog.generation_logger import CSV_COLUMNS
    with tempfile.TemporaryDirectory() as tmpdir:
        world, logger = _run_with_logging(tmpdir=tmpdir)
        rows = list(csv.DictReader(open(logger.csv_path)))
        assert len(rows) > 0, "No rows written to CSV"
        header = list(rows[0].keys())
        assert header == CSV_COLUMNS, (
            f"Column mismatch.\nExpected: {CSV_COLUMNS}\nGot:      {header}"
        )
        print(f"  All {len(CSV_COLUMNS)} columns present in correct order ✓")


def test_no_duplicate_frames():
    """Each row should have a unique frame number."""
    with tempfile.TemporaryDirectory() as tmpdir:
        world, logger = _run_with_logging(tmpdir=tmpdir)
        rows   = list(csv.DictReader(open(logger.csv_path)))
        frames = [int(r['frame']) for r in rows]
        assert len(frames) == len(set(frames)), (
            f"Duplicate frame numbers in CSV: {[f for f in frames if frames.count(f) > 1]}"
        )


def test_log_interval_respected():
    """Rows should be written at multiples of LOG_INTERVAL (within ±1 frame)."""
    interval = 300
    with tempfile.TemporaryDirectory() as tmpdir:
        world, logger = _run_with_logging(frames=1000, log_interval=interval, tmpdir=tmpdir)
        rows   = list(csv.DictReader(open(logger.csv_path)))
        frames = [int(r['frame']) for r in rows]
        for f in frames:
            assert f % interval == 0, (
                f"Row written at frame {f} which is not a multiple of {interval}"
            )


def test_fitness_values_in_range():
    """All fitness columns should be in [0, 1] when present (mature cols may be None)."""
    fitness_cols = [
        'average_alive_fitness', 'best_alive_fitness',
        'mature_average_fitness',
        'fitness_10th_percentile', 'fitness_50th_percentile',
        'fitness_75th_percentile', 'fitness_90th_percentile',
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        world, logger = _run_with_logging(tmpdir=tmpdir)
        rows = list(csv.DictReader(open(logger.csv_path)))
        for row in rows:
            for col in fitness_cols:
                raw = row[col]
                if raw == '' or raw is None:
                    continue   # mature columns are None when no mature agents exist
                val = float(raw)
                assert 0.0 <= val <= 1.0, (
                    f"Frame {row['frame']}: {col}={val} out of [0, 1]"
                )


def test_best_fitness_gte_average():
    """best_alive_fitness should always be >= average_alive_fitness."""
    with tempfile.TemporaryDirectory() as tmpdir:
        world, logger = _run_with_logging(tmpdir=tmpdir)
        rows = list(csv.DictReader(open(logger.csv_path)))
        for row in rows:
            best = float(row['best_alive_fitness'])
            avg  = float(row['average_alive_fitness'])
            assert best >= avg - 1e-6, (
                f"Frame {row['frame']}: best_fitness {best} < avg_fitness {avg}"
            )
            # mature_average_fitness should also be <= best when present
            if row['mature_average_fitness'] not in ('', None):
                mavg = float(row['mature_average_fitness'])
                assert best >= mavg - 1e-6, (
                    f"Frame {row['frame']}: best_fitness {best} < mature_avg {mavg}"
                )


def test_archetype_percentages_sum_to_one():
    """pct_stationary + pct_straight_runner + ... should sum to 1.0 per row."""
    pct_cols = [
        'pct_stationary', 'pct_straight_runner', 'pct_erratic',
        'pct_left_turner', 'pct_right_turner', 'pct_balanced_explorer',
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        world, logger = _run_with_logging(tmpdir=tmpdir)
        rows = list(csv.DictReader(open(logger.csv_path)))
        for row in rows:
            total = sum(float(row[c]) for c in pct_cols)
            assert abs(total - 1.0) < 1e-4, (
                f"Frame {row['frame']}: archetype percentages sum to {total}, not 1.0 "
                f"({', '.join(f'{c}={row[c]}' for c in pct_cols)})"
            )


def test_total_born_monotonic_in_csv():
    """total_agents_ever_born must never decrease across rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        world, logger = _run_with_logging(tmpdir=tmpdir)
        rows = list(csv.DictReader(open(logger.csv_path)))
        prev = 0
        for row in rows:
            val = int(row['total_agents_ever_born'])
            assert val >= prev, (
                f"total_agents_ever_born went from {prev} to {val} at frame {row['frame']}"
            )
            prev = val


def test_total_deaths_monotonic_in_csv():
    """total_agents_dead must never decrease across rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        world, logger = _run_with_logging(tmpdir=tmpdir)
        rows = list(csv.DictReader(open(logger.csv_path)))
        prev = 0
        for row in rows:
            val = int(row['total_agents_dead'])
            assert val >= prev, (
                f"total_agents_dead went from {prev} to {val} at frame {row['frame']}"
            )
            prev = val


def test_checkpoint_npz_loads_correctly():
    """Checkpoint .npz contains valid weight arrays with correct shapes."""
    import config
    from config import BRAIN_INPUTS, BRAIN_OUTPUTS
    with tempfile.TemporaryDirectory() as tmpdir:
        world, logger = _run_with_logging(
            frames=600, log_interval=200, checkpoint_interval=2, tmpdir=tmpdir
        )
        chk_files = sorted(glob.glob(os.path.join(logger.checkpoint_dir, "*.npz")))
        assert chk_files, "No checkpoint files written"

        data = np.load(chk_files[-1])
        n = int(data['pool_size'][0])
        assert n > 0, "Checkpoint has 0 agents in pool"

        for i in range(min(n, 5)):  # check first 5
            num_layers = int(data[f'agent{i}_num_layers'][0])
            layer_sizes = data[f'agent{i}_layer_sizes'].tolist()
            assert layer_sizes[0]  == BRAIN_INPUTS,  f"Wrong input size: {layer_sizes}"
            assert layer_sizes[-1] == BRAIN_OUTPUTS, f"Wrong output size: {layer_sizes}"
            for j in range(num_layers):
                W = data[f'agent{i}_W{j}']
                b = data[f'agent{i}_b{j}']
                assert W.shape == (layer_sizes[j+1], layer_sizes[j]), (
                    f"agent{i} W{j} shape {W.shape} != expected "
                    f"({layer_sizes[j+1]}, {layer_sizes[j]})"
                )
                assert b.shape == (layer_sizes[j+1],), (
                    f"agent{i} b{j} shape {b.shape} != expected ({layer_sizes[j+1]},)"
                )
        print(f"  Checkpoint {os.path.basename(chk_files[-1])}: {n} agents, shapes OK ✓")


def test_alive_population_matches_world():
    """
    alive_population in each logged row must match the actual world.agents
    count at that frame. We verify by running the world and intercepting
    the log call.
    """
    import config
    config.LOG_INTERVAL = 100
    config.CHECKPOINT_INTERVAL = 0

    from simulation.world import World
    from evolog.generation_logger import GenerationLogger
    import tempfile

    seed(0)
    recorded = []   # (frame, logged_pop, actual_pop)

    class SpyLogger(GenerationLogger):
        def log_frame(self, world):
            actual = len(world.agents)
            super().log_frame(world)
            # Read back what was just written
            rows = list(csv.DictReader(open(self.csv_path)))
            if rows:
                logged = int(rows[-1]['alive_population'])
                recorded.append((world.frame, logged, actual))

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = SpyLogger(run_id="spy_test", runs_dir=tmpdir)
        world  = World(logger=logger)
        for _ in range(600):
            world.update()

    assert recorded, "No log rows intercepted"
    for frame, logged, actual in recorded:
        assert logged == actual, (
            f"Frame {frame}: logged alive_population={logged} != "
            f"actual world population={actual}"
        )
    print(f"  alive_population matches world at all {len(recorded)} checkpoints ✓")


if __name__ == "__main__":
    print("Running logging tests...")
    test_csv_columns_match_spec()
    test_no_duplicate_frames()
    test_log_interval_respected()
    test_fitness_values_in_range()
    test_best_fitness_gte_average()
    test_archetype_percentages_sum_to_one()
    test_total_born_monotonic_in_csv()
    test_total_deaths_monotonic_in_csv()
    test_checkpoint_npz_loads_correctly()
    test_alive_population_matches_world()
    print("All logging tests passed.")
