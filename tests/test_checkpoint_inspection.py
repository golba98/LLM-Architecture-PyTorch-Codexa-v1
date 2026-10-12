"""Native storage inspection is read-only and accounts tied archive storages once."""

import hashlib

import pytest
import torch

from llm_architecture.checkpoint_inspection import inspect_checkpoint_storage


def test_inspection_tied_storages_optimizer_precision_and_checksum(tmp_path):
    weight=torch.arange(8,dtype=torch.float32)
    value=dict(model_state_dict={'input':weight,'output':weight},
        optimizer_state_dict={'state':{0:{'state1':torch.arange(8,dtype=torch.uint8)}}},
        config={'model':{},'training':{}},training_state={'optimizer_step':1},run_id='fixture')
    path=tmp_path/'checkpoint.pt'
    torch.save(value,path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar=path.with_suffix('.pt.sha256')
    sidecar.write_text(digest+'  '+path.name+'\n')
    before=(path.stat(),sidecar.stat())
    result=inspect_checkpoint_storage(path)
    assert result['components']['model_state_dict']['bytes']==32
    assert result['components']['optimizer_state_dict']['bytes']==8
    assert result['components']['optimizer_state_dict']['by_dtype']=={'torch.uint8':8}
    assert result['sha256']==digest and not result['precision_conversion']
    assert path.stat().st_mtime_ns==before[0].st_mtime_ns
    assert sidecar.stat().st_mtime_ns==before[1].st_mtime_ns
    sidecar.write_text('0'*64+'  '+path.name+'\n')
    with pytest.raises(ValueError,match='checksum mismatch'):
        inspect_checkpoint_storage(path)
