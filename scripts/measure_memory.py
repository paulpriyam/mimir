"""Measure peak memory of a whole command's process tree.

    python scripts/measure_memory.py .venv/bin/python -m mimir --index-dir /tmp/i index ~/mimir-testbed

The models live in subprocesses, so a single process's ru_maxrss misses them - which is the
easy way to under-report indexing memory. This samples every process in the run's process
group and reports the peak, which is the number that decides whether a machine can cope.

Measured this way on the testbed, with phased indexing:

    embed only (the floor)   2635 MB
    + speech phase           2765 MB
    + captions (full run)    3037 MB
    naive sum, if the models overlapped   ~5.3 GB

Note: launch the real interpreter rather than a wrapper like `uv run`, which re-execs and can
make the child look like it exited immediately.
"""

import subprocess
import sys
import time

cmd = sys.argv[1:]
if not cmd:
    print(__doc__.splitlines()[2].strip(), file=sys.stderr)
    raise SystemExit(2)

proc = subprocess.Popen(cmd, start_new_session=True)
group = proc.pid
peak, samples = 0.0, []
started = time.time()

while proc.poll() is None:
    listing = subprocess.run(["ps", "-Ao", "rss=,pgid="], capture_output=True, text=True).stdout
    total = 0
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == str(group):
            total += int(parts[0])
    if total:
        samples.append((round(time.time() - started, 1), total / 1024))
        peak = max(peak, total / 1024)
    time.sleep(0.4)

print(f"\npeak tree RSS: {peak:.0f} MB over {len(samples)} samples")
for seconds, mb in samples[::max(1, len(samples) // 14)]:
    print(f"  {seconds:6.1f}s  {mb:6.0f} MB")
sys.exit(proc.returncode or 0)
