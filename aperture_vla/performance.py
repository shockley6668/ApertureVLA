"""Shared benchmark statistics and an exclusive mainline GPU lease."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def latency_statistics(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() or (x<0).any():
        raise ValueError('Expected at least two finite nonnegative latency measurements')
    return {'count':len(x),'p50_ms':float(np.percentile(x,50)),
            'p95_ms':float(np.percentile(x,95)),'mean_ms':float(x.mean()),
            'min_ms':float(x.min()),'max_ms':float(x.max())}



def parse_exclusive_compute_processes(xml, current_pid=None):
    """Return only C-type processes; nvidia-smi also lists C+G/G desktop contexts."""
    rows=[]
    for process in ET.fromstring(xml).findall('.//processes/process_info'):
        pid_text=process.findtext('pid')
        process_type=process.findtext('type','').strip()
        if not pid_text or process_type!='C':
            continue
        pid=int(pid_text)
        if current_pid is None or pid!=current_pid:
            rows.append({'pid':pid,'process_name':process.findtext('process_name','').strip()})
    return rows


def assert_no_other_exclusive_compute_process(current_pid):
    xml=subprocess.check_output(['nvidia-smi','-q','-x'],text=True)
    other=parse_exclusive_compute_processes(xml,current_pid)
    if other:
        detail='; '.join(f"{row['pid']} {row['process_name']}" for row in other)
        raise RuntimeError('Another GPU compute process is active: '+detail)


def validate_input_artifact(payload):
    if payload.get('evidence')!='balanced_libero_performance_inputs_v1':
        raise ValueError('Unsupported performance input artifact')
    manifest=payload['manifest']
    counts=Counter((r['suite'],r['task_id']) for r in manifest)
    suites={'libero_spatial','libero_object','libero_goal','libero_10'}
    if set(counts)!={(suite,i) for suite in suites for i in range(10)} or len(set(counts.values()))!=1:
        raise ValueError('Input task quotas must cover all 40 tasks equally')
    n=len(manifest)
    if len(payload['instructions'])!=n or any(not isinstance(x,str) or not x.strip() for x in payload['instructions']):
        raise ValueError('Every performance input needs a nonempty instruction')
    identities={(r['suite'],r['task_id'],r['episode_index'],r['step_index']) for r in manifest}
    if len(identities)!=n:raise ValueError('Duplicate performance inputs')
    expected={'normalized_pixels':(n,2,3,256,256),'raw_rgb':(n,2,256,256,3),
              'normalized_states':(n,8),'raw_states':(n,8)}
    for name,shape in expected.items():
        if tuple(payload[name].shape)!=shape:raise ValueError(f'Invalid shape for {name}')
        if payload[name].is_floating_point() and not torch.isfinite(payload[name]).all():
            raise ValueError(f'Non-finite input in {name}')
    if payload['raw_rgb'].dtype!=torch.uint8:raise ValueError('RGB must be uint8')
    return n


def compact_forward_inputs(policy, payload):
    """Use the evaluated policy's RGB/state transforms for forward timing.

    The fixed artifact's normalized tensors belong to its reference encoder.
    Only raw images/states can be reused when benchmarking another encoder.
    This CPU preparation is outside the forward-only timed region.
    """
    batches, states = [], []
    for images, state in zip(payload['raw_rgb'], payload['raw_states'], strict=True):
        pixels, normalized_state = policy._build_batch(
            images[0].numpy(), images[1].numpy(), state.numpy()
        )
        batches.append(pixels)
        states.append(normalized_state)
    pixels, states = torch.cat(batches), torch.cat(states)
    if not torch.isfinite(pixels).all() or not torch.isfinite(states).all():
        raise ValueError('Non-finite policy-preprocessed benchmark input')
    return pixels, states


@contextmanager
def mainline_gpu_lease(root):
    import fcntl
    path=Path(root)/'outputs/data/corrected_mainline.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Mainline GPU job is still active') from error
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)
