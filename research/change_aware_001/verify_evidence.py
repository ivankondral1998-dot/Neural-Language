#!/usr/bin/env python3
"""Check stored experiment outputs; no recompilation or mutation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

def verify(p:Path,kind:str):
    rows=json.loads((p/'results.json').read_text())
    metadata=json.loads((p/'metadata.json').read_text())
    assert len(rows)==16,(p,len(rows))
    assert metadata['reps']==5 and metadata['steps']==250 and metadata['warmup']==32
    assert metadata['quick'] is False
    if kind=='runtime':
        gates=[r['auto_passes_pre_registered_gate'] for r in rows]
        assert all(r['max_relative_error']<=1e-9 for r in rows)
        assert metadata['source_blobs']['language/runtime/neural_runtime.hpp']=='ee71cdab32b248814aa23f26318130bd155c824f'
        for r in rows:
            assert (r['auto_paired_wins_20pct']>=4 and r['runtime_auto_us']<=0.8*r['runtime_full_us'])==r['auto_passes_pre_registered_gate']
        print(p, 'actual runtime: passed gates',sum(gates),'/',len(gates), 'numerical comparisons',sum(r['comparisons'] for r in rows))
    else:
        gates=[r['passes_20pct_gate'] for r in rows]
        assert all(r['max_relative_error']<=1e-9 for r in rows)
        for r in rows:
            assert (r['paired_improvements_20pct']>=4 and r['delta_us']<=.8*r['blas_us'])==r['passes_20pct_gate']
        print(p,'independent candidate: passed gates',sum(gates),'/',len(gates), 'numerical comparisons',sum(r['comparisons'] for r in rows))
    return sum(gates)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--evidence',type=Path,default=Path(__file__).parent/'evidence')
    o=ap.parse_args();p=o.evidence
    observed=[verify(p/'runtime_gcc','runtime'),verify(p/'runtime_clang','runtime'),
              verify(p/'independent_gcc','independent'),verify(p/'independent_clang','independent')]
    assert observed==[16,15,10,9],observed
    print('EVIDENCE_COHERENCE_PASS:',observed)
if __name__=='__main__':main()
