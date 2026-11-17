"""Record benchmark runs and Nextflow traces as rows in tracked CSV histories.

bench [pytest args]    run tests/benchmarks, append one row per benchmark to benchmarks/history.csv and
                       print the median ratio against benchmarks/baseline/latest.json (same CPU only)
trace <trace.txt> <label>
                       append one row per Nextflow process to benchmarks/pipeline_history.csv
"""
from __future__ import annotations

import csv
import json
import logging
import re
import resource
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_CSV = ROOT / 'benchmarks' / 'history.csv'
PIPE_CSV = ROOT / 'benchmarks' / 'pipeline_history.csv'
BASELINE = ROOT / 'benchmarks' / 'baseline' / 'latest.json'
REGRESSION = 1.10
BENCH_COLS = ['date', 'commit', 'dirty', 'branch', 'cpu', 'system', 'python', 'benchmark', 'median_s',
              'iqr_s', 'mean_s', 'stddev_s', 'min_s', 'rounds', 'session_peak_rss_mb']
PIPE_COLS = ['date', 'label', 'commit', 'process', 'n_tasks', 'n_failed', 'total_realtime_h',
             'max_realtime_min', 'cpu_hours', 'max_peak_rss_gb']
_SIZE = {'B': 1, 'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3, 'TB': 1024 ** 4}
_TIME = {'ms': 1e-3, 's': 1.0, 'm': 60.0, 'h': 3600.0, 'd': 86400.0}

log = logging.getLogger('benchmarks.record')


def _append(path: Path, cols: list[str], rows: list[dict[str, object]]) -> None:
    """Append rows to a CSV history, writing the header when the file is new."""
    new = not path.exists()
    with path.open('a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator='\n')
        if new:
            w.writeheader()
        w.writerows(rows)


def _git(*args: str) -> str:
    """Return stripped git output, or '' when git is unavailable."""
    r = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ''


def _cpu(machine_info: dict) -> str:
    """CPU model from pytest-benchmark, else sysctl (macOS) or /proc/cpuinfo (Linux)."""
    brand = machine_info.get('cpu', {}).get('brand_raw', '')
    if brand:
        return brand
    if sys.platform == 'darwin':
        r = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True)
        return r.stdout.strip()
    info = Path('/proc/cpuinfo')
    if info.exists():
        m = re.search(r'model name\s*:\s*(.+)', info.read_text())
        return m.group(1).strip() if m else ''
    return ''


def _peak_rss_mb() -> float:
    """Peak RSS of finished child processes (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return peak / 1024 ** 2 if sys.platform == 'darwin' else peak / 1024


def bench(extra: list[str]) -> int:
    """Run the benchmark suite once and record every benchmark it completes."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / 'bench.json'
        cmd = [sys.executable, '-m', 'pytest', 'tests/benchmarks', '-q', f'--benchmark-json={out}', *extra]
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if not out.exists():
            log.error('pytest wrote no benchmark JSON (exit %d); is pytest-benchmark installed?', rc)
            return rc or 1
        run = json.loads(out.read_text())
    rss = round(_peak_rss_mb(), 1)
    mi = run.get('machine_info', {})
    cpu = _cpu(mi)
    commit, branch = _git('rev-parse', '--short=8', 'HEAD'), _git('rev-parse', '--abbrev-ref', 'HEAD')
    dirty = bool(_git('status', '--porcelain', '--untracked-files=no'))
    rows = []
    for b in run['benchmarks']:
        s = b['stats']
        rows.append({'date': run['datetime'][:19], 'commit': commit, 'dirty': dirty, 'branch': branch, 'cpu': cpu,
                     'system': f"{mi.get('system', '')} {mi.get('release', '')}".strip(),
                     'python': mi.get('python_version', ''), 'benchmark': b['name'],
                     'median_s': f"{s['median']:.6g}", 'iqr_s': f"{s['iqr']:.6g}", 'mean_s': f"{s['mean']:.6g}",
                     'stddev_s': f"{s['stddev']:.6g}", 'min_s': f"{s['min']:.6g}", 'rounds': s['rounds'],
                     'session_peak_rss_mb': rss})
    _append(BENCH_CSV, BENCH_COLS, rows)
    log.info('appended %d rows to %s (session peak RSS %.0f MB)', len(rows), BENCH_CSV.name, rss)
    _compare(run, cpu)
    return rc


def _compare(run: dict, cpu: str) -> None:
    """Log new/baseline median ratios; skip when the baseline was measured on another CPU."""
    if not BASELINE.exists():
        log.warning('no baseline at %s', BASELINE)
        return
    base = json.loads(BASELINE.read_text())
    base_cpu = base.get('machine_info', {}).get('cpu', {}).get('brand_raw', '')
    if base_cpu != cpu:
        log.warning('baseline CPU %r differs from %r: ratios not comparable, skipped', base_cpu, cpu)
        return
    ref = {b['name']: b['stats']['median'] for b in base['benchmarks']}
    for b in run['benchmarks']:
        if b['name'] in ref:
            ratio = b['stats']['median'] / ref[b['name']]
            flag = '  REGRESSION' if ratio > REGRESSION else ''
            log.info('%-40s %8.3fx baseline%s', b['name'], ratio, flag)


def _size_bytes(v: str) -> float:
    """Parse a Nextflow size ('1.2 GB' or raw bytes)."""
    v = v.strip()
    if v in ('', '-'):
        return float('nan')
    m = re.fullmatch(r'([\d.]+)\s*([KMGT]?B)?', v)
    return float(m.group(1)) * _SIZE[m.group(2) or 'B'] if m else float('nan')


def _seconds(v: str) -> float:
    """Parse a Nextflow duration ('1h 2m 3s', '500ms' or raw milliseconds)."""
    v = v.strip()
    if v in ('', '-'):
        return float('nan')
    if re.fullmatch(r'\d+', v):
        return int(v) / 1000
    parts = re.findall(r'([\d.]+)\s*(ms|s|m|h|d)', v)
    return sum(float(n) * _TIME[u] for n, u in parts) if parts else float('nan')


def trace(path: Path, label: str) -> int:
    """Summarise a Nextflow trace per process and append it to the pipeline history."""
    with path.open() as fh:
        tasks = list(csv.DictReader(fh, delimiter='\t'))
    by_proc: dict[str, list[dict[str, str]]] = {}
    for t in tasks:
        by_proc.setdefault(t['name'].split(' (')[0], []).append(t)
    commit, date = _git('rev-parse', '--short=8', 'HEAD'), datetime.now(timezone.utc).isoformat(timespec='seconds')
    rows = []
    for proc, ts in sorted(by_proc.items()):
        real = [_seconds(t.get('realtime', '')) for t in ts]
        cpu = [r * float(str(t.get('%cpu', '0')).rstrip('%') or 0) / 100 for r, t in zip(real, ts)]
        rss = [r for r in (_size_bytes(t.get('peak_rss', '')) for t in ts) if r == r]
        ok = [x for x in real if x == x]
        rows.append({'date': date, 'label': label, 'commit': commit, 'process': proc, 'n_tasks': len(ts),
                     'n_failed': sum(t.get('status') not in ('COMPLETED', 'CACHED') for t in ts),
                     'total_realtime_h': round(sum(ok) / 3600, 3),
                     'max_realtime_min': round(max(ok) / 60, 2) if ok else '',
                     'cpu_hours': round(sum(c for c in cpu if c == c) / 3600, 3),
                     'max_peak_rss_gb': round(max(rss) / 1024 ** 3, 3) if rss else ''})
    _append(PIPE_CSV, PIPE_COLS, rows)
    log.info('appended %d process rows from %d tasks to %s', len(rows), len(tasks), PIPE_CSV.name)
    return 0


def main() -> int:
    """Dispatch on the first positional argument."""
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if len(sys.argv) >= 2 and sys.argv[1] == 'bench':
        return bench(sys.argv[2:])
    if len(sys.argv) == 4 and sys.argv[1] == 'trace':
        return trace(Path(sys.argv[2]), sys.argv[3])
    log.error(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main())
