"""
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
Final Robust Version for Windows/Jupyter Stability.
"""

import inspect
import importlib
import functools
from collections import defaultdict
from typing import Any, Dict, Optional, List
import torch

GLOBAL_CONFIG = defaultdict(dict)

def register(dct: Any=GLOBAL_CONFIG, name=None, force=False):
    def decorator(foo):
        register_name = foo.__name__ if name is None else name
        if not force:
            if inspect.isclass(dct):
                assert not hasattr(dct, foo.__name__), f'module {dct.__name__} has {foo.__name__}'
            else:
                assert foo.__name__ not in dct, f'{foo.__name__} has been already registered'

        if inspect.isfunction(foo):
            @functools.wraps(foo)
            def wrap_func(*args, **kwargs):
                return foo(*args, **kwargs)
            if isinstance(dct, dict):
                dct[foo.__name__] = wrap_func
            elif inspect.isclass(dct):
                setattr(dct, foo.__name__, wrap_func)
            return wrap_func

        elif inspect.isclass(foo):
            dct[register_name] = extract_schema(foo) 
        return foo
    return decorator

def extract_schema(module: type):
    argspec = inspect.getfullargspec(module.__init__)
    arg_names = [arg for arg in argspec.args if arg != 'self']
    num_defaults = len(argspec.defaults) if argspec.defaults is not None else 0
    num_requires = len(arg_names) - num_defaults

    schema = dict()
    schema['_name'] = module.__name__
    schema['_pymodule'] = importlib.import_module(module.__module__)
    schema['_inject'] = getattr(module, '__inject__', [])
    schema['_share'] = getattr(module, '__share__', [])
    schema['_kwargs'] = {}
    
    for i, name in enumerate(arg_names):
        schema[name] = argspec.defaults[i - num_requires] if i >= num_requires else None
        schema['_kwargs'][name] = schema[name]
        
    return schema

def create(type_or_name, global_cfg=GLOBAL_CONFIG, **kwargs):
    assert type(type_or_name) in (type, str), 'create should be modules or name.'
    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__

# --- 1. SPECIAL HANDLING FOR DATALOADERS ---
    if name in ['train_dataloader', 'val_dataloader', 'DataLoader']:
        import torch.utils.data
        base_cfg = global_cfg.get(name, {}).copy()
        base_cfg.update(kwargs)
        
        # A. BUILD THE DATASET
        if 'dataset' in base_cfg:
            ds_info = base_cfg.pop('dataset')
            if isinstance(ds_info, dict):
                ds_type = ds_info.pop('type', ds_info.get('_name', 'CocoDetection'))
                # Build Transforms if they are a dict
                if 'transforms' in ds_info and isinstance(ds_info['transforms'], dict):
                    trans_cfg = ds_info.pop('transforms')
                    t_type = trans_cfg.pop('type', 'Compose')
                    ds_info['transforms'] = create(t_type, global_cfg, **trans_cfg)
                
                base_cfg['dataset'] = create(ds_type, global_cfg, **ds_info)
            else:
                base_cfg['dataset'] = ds_info

        # B. BUILD THE COLLATE_FN (The current fix)
        if 'collate_fn' in base_cfg and isinstance(base_cfg['collate_fn'], dict):
            collate_info = base_cfg.pop('collate_fn')
            # RT-DETR usually uses 'BatchCompose' or 'DefaultCollate'
            collate_type = collate_info.pop('type', 'BatchCompose')
            base_cfg['collate_fn'] = create(collate_type, global_cfg, **collate_info)

        # C. CLEANUP FOR PYTORCH
        for k in ['type', 'total_batch_size', '_pymodule', '_name', 'shuffle']:
            base_cfg.pop(k, None)
            
        return torch.utils.data.DataLoader(**base_cfg)
    # --- 2. REGISTRY LOOKUP ---
    if name not in global_cfg:
        import torch.nn as nn
        import torch.optim as optim
        if hasattr(nn, name): return getattr(nn, name)(**kwargs)
        if hasattr(optim, name): return getattr(optim, name)(**kwargs)
        raise ValueError(f'The module {name} is not registered')

    # Get a copy of the config for this component
    cfg = global_cfg[name].copy()

    # Handle Nested Types (like 'type: AdamW')
    if isinstance(cfg, dict) and 'type' in cfg:
        _type = cfg.pop('type')
        # Ensure we don't have duplicate 'params' for optimizers
        for k, v in kwargs.items():
            cfg[k] = v
        return create(_type, global_cfg, **cfg)

    # --- 3. MODULE INJECTION FALLBACK ---
    if '_pymodule' not in cfg:
        if name in ['CocoDetection', 'CocoDataset']:
            import src.dataset.coco
            cfg['_pymodule'] = src.dataset.coco
        elif name == 'AdamW':
            import torch.optim
            cfg['_pymodule'] = torch.optim
        else:
            import torch.nn as nn
            if hasattr(nn, name): cfg['_pymodule'] = nn

    # --- 4. PREPARE KWARGS ---
    module = getattr(cfg['_pymodule'], name)    
    module_kwargs = {k: v for k, v in cfg.items() if not k.startswith('_')}
    module_kwargs.update({k: v for k, v in kwargs.items() if not k.startswith('_')})

    # Shared variables (like num_classes)
    for k in cfg.get('_share', []):
        if k in global_cfg:
            module_kwargs[k] = global_cfg[k]

    # Injections (Recursive Calls for Backbones, Encoders, etc.)
    for k in cfg.get('_inject', []):
        _k = cfg[k]
        if _k is None: continue
        if isinstance(_k, str) and _k in global_cfg:
            module_kwargs[k] = create(_k, global_cfg)
        elif isinstance(_k, dict) and 'type' in _k:
            _inject_cfg = _k.copy()
            _inject_type = _inject_cfg.pop('type')
            module_kwargs[k] = create(_inject_type, global_cfg, **_inject_cfg)

    # Final Instantiation: remove 'type' to avoid TypeError
    final_params = {k: v for k, v in module_kwargs.items() if not k.startswith('_') and k != 'type'}
    return module(**final_params)