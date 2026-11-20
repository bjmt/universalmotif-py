"""Run a command and report its wall time and the peak memory of its whole
process tree.

    python treemem.py COMMAND ...

/usr/bin/time reports the peak of one process. TF-MoDISco forks worker
processes that run beside the main one, so this samples, every 0.2 s, the
resident memory of the command and all its descendants, and reports the
largest sum and the largest single process. The sum counts pages the forked
workers share with their parent once per process, so it is an upper bound
on the memory used, and the largest single process a lower one.
"""

import subprocess
import sys
import time


def tree_rss(root: int) -> list[int]:
    """The resident memory, in bytes, of root and every process below it."""
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in out.splitlines():
        pid, ppid, kb = (int(x) for x in line.split())
        children.setdefault(ppid, []).append(pid)
        rss[pid] = kb * 1024
    todo, found = [root], []
    while todo:
        p = todo.pop()
        if p in rss:
            found.append(rss[p])
        todo.extend(children.get(p, ()))
    return found


if __name__ == "__main__":
    t0 = time.time()
    proc = subprocess.Popen(sys.argv[1:])
    peak_sum = peak_one = 0
    while proc.poll() is None:
        sizes = tree_rss(proc.pid)
        if sizes:
            peak_sum, peak_one = max(peak_sum, sum(sizes)), max(peak_one, max(sizes))
        time.sleep(0.2)
    print(f"treemem: {time.time() - t0:.1f} s, tree peak {peak_sum / 1e9:.2f} GB, "
          f"largest process {peak_one / 1e9:.2f} GB", file=sys.stderr, flush=True)
    sys.exit(proc.returncode)
