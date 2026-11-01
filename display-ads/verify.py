#!/usr/bin/env python3
"""Verify matching/display-ads potential inequalities on the full request range.

Both parts use one freshly constructed potential table in a combined run.
Saved JSON files are outputs only and are never used as input certificates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import uuid

import flint
from flint import arb, ctx

from potential import PotentialTable
import verify_large
import verify_small


ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ('verify.py', 'verify_small.py', 'verify_large.py', 'potential.py')


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def write_json(path, report):
    """Replace one complete JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=('all', 'small', 'large'), default='all',
                        help='Request range to certify (default: all).')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'results',
                        help='Directory for summary.json and per-part results.')
    args = parser.parse_args(argv)
    if sys.flags.optimize:
        parser.error('Do not use python -O: certificate assertions must be enabled.')
    ctx.prec = 160
    lam = arb(9)/2
    mu = (-lam).exp()
    run_id = str(uuid.uuid4())
    sources = source_hashes()
    common = {
        'run_id': run_id, 'lambda': '9/2', 'precision_bits': ctx.prec,
        'potential_integration_cells': 8192, 'potential_table_cells': 8192,
        'python': sys.version.split()[0], 'python_flint': flint.__version__,
        'platform': platform.platform(), 'source_sha256': sources,
    }
    summary = {
        **common, 'complete': False, 'status': 'running', 'mode': args.only,
        'full_request_range_certified': False,
        'potential_table_builds': 0, 'parts': {},
    }
    # An interrupted run cannot leave an old successful summary looking current.
    # Per-part files must have this summary's run_id to belong to the same run.
    write_json(args.output_dir/'summary.json', summary)
    started = time.perf_counter()
    try:
        table_started = time.perf_counter()
        table = PotentialTable(8192, 8192, lam, mu)
        summary['potential_table_seconds'] = time.perf_counter()-table_started
        summary['potential_table_builds'] = 1
        summary['gamma_zero_lower'] = str(table.g0_lower)
        summary['gamma_zero_upper'] = str(table.g0_upper)
        print(f"Shared potential table complete in "
              f"{summary['potential_table_seconds']:.3f}s", flush=True)
        write_json(args.output_dir/'summary.json', summary)
        parts = ('small', 'large') if args.only == 'all' else (args.only,)
        for part in parts:
            result = (verify_small if part == 'small' else verify_large).verify(table, lam, mu)
            result = {**common, **result, 'potential_table_record': 'summary.json'}
            if source_hashes() != sources:
                raise RuntimeError('Source files changed while verification was running.')
            write_json(args.output_dir/f'{part}.json', result)
            summary['parts'][part] = {
                'file': f'{part}.json', 'complete': result['complete'],
                'minimum_lower_bound': result['minimum_lower_bound'],
                'certificate_seconds': result['certificate_seconds'],
                'request_intervals': len(result['intervals']),
            }
            write_json(args.output_dir/'summary.json', summary)
        summary['complete'] = all(p['complete'] for p in summary['parts'].values())
        summary['full_request_range_certified'] = summary['complete'] and args.only == 'all'
        summary['status'] = 'passed'
        summary['total_seconds'] = time.perf_counter()-started
        write_json(args.output_dir/'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    except BaseException as error:
        summary['complete'] = False
        summary['full_request_range_certified'] = False
        summary['status'] = 'failed' if not isinstance(error, KeyboardInterrupt) else 'interrupted'
        summary['error'] = f'{type(error).__name__}: {error}'
        summary['total_seconds'] = time.perf_counter()-started
        write_json(args.output_dir/'summary.json', summary)
        raise


if __name__ == '__main__':
    main()
