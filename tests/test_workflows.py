#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.core import utils
import indexed_gzip
import nibabel as nib
import pytest
import shutil
import os
import sys
import json
import ast
import yaml
import itertools
from types import SimpleNamespace
from pathlib import Path
from pynets.core.utils import flatten
from pynets.cli.pynets_run import build_workflow


@pytest.mark.parametrize("node_size", [6, None])
@pytest.mark.parametrize("hpass", [100, None])
@pytest.mark.parametrize("smooth", [6, None])
@pytest.mark.parametrize("parc", [True, False])
def test_func_workflows(node_size, hpass, smooth, parc):
    """
    Test functional connectometry
    """
    base_dir = str(Path(__file__).parent/"examples")
    space = 'MNI152NLin2009cAsym'
    func_desc = 'smoothAROMAnonaggr'

    intermodal_dict = {k: [] for k in ['funcs', 'confs', 'dwis', 'bvals', 'bvecs', 'anats', 'masks',
                                       'subjs', 'seshs']}
    if analysis_level == 'group':
        if len(modality) > 1:
            i = 0
            for mod in modality:
                outs = sweep_directory(bids_dir, modality=mod, space=space, func_desc=func_desc, sesh=session_label)
                if mod == 'func':
                    if i == 0:
                        funcs, confs, _, _, _, anats, masks, subjs, seshs = outs
                    else:
                        funcs, confs, _, _, _, _, _, _, _ = outs
                    intermodal_dict['funcs'].append(funcs)
                    intermodal_dict['confs'].append(confs)
                elif mod == 'dwi':
                    if i == 0:
                        _, _, dwis, bvals, bvecs, anats, masks, subjs, seshs = outs
                    else:
                        _, _, dwis, bvals, bvecs, _, _, _, _ = outs
                    intermodal_dict['dwis'].append(dwis)
                    intermodal_dict['bvals'].append(bvals)
                    intermodal_dict['bvecs'].append(bvecs)
                intermodal_dict['anats'].append(anats)
                intermodal_dict['masks'].append(masks)
                intermodal_dict['subjs'].append(subjs)
                intermodal_dict['seshs'].append(seshs)
                i += 1
        else:
            intermodal_dict = None
            outs = sweep_directory(bids_dir, modality=modality[0], space=space, func_desc=func_desc,
                                   sesh=session_label)
            funcs, confs, dwis, bvals, bvecs, anats, masks, subjs, seshs = outs
    elif analysis_level == 'participant':
        if len(modality) > 1:
            i = 0
            for mod in modality:
                outs = sweep_directory(bids_dir, modality=mod, space=space, func_desc=func_desc,
                                       subj=participant_label, sesh=session_label)
                if mod == 'func':
                    if i == 0:
                        funcs, confs, _, _, _, anats, masks, subjs, seshs = outs
                    else:
                        funcs, confs, _, _, _, _, _, _, _ = outs
                    intermodal_dict['funcs'].append(funcs)
                    intermodal_dict['confs'].append(confs)
                elif mod == 'dwi':
                    if i == 0:
                        _, _, dwis, bvals, bvecs, anats, masks, subjs, seshs = outs
                    else:
                        _, _, dwis, bvals, bvecs, _, _, _, _ = outs
                    intermodal_dict['dwis'].append(dwis)
                    intermodal_dict['bvals'].append(bvals)
                    intermodal_dict['bvecs'].append(bvecs)
                intermodal_dict['anats'].append(anats)
                intermodal_dict['masks'].append(masks)
                intermodal_dict['subjs'].append(subjs)
                intermodal_dict['seshs'].append(seshs)
                i += 1
        else:
            intermodal_dict = None
            outs = sweep_directory(bids_dir, modality=modality[0], space=space, func_desc=func_desc,
                                   subj=participant_label, sesh=session_label)
            funcs, confs, dwis, bvals, bvecs, anats, masks, subjs, seshs = outs
    else:
        raise ValueError('Analysis level invalid. Must be `participant` or `group`. See --help.')

    if intermodal_dict:
        funcs, confs, dwis, bvals, bvecs, anats, masks, subjs, seshs = [list(set(list(flatten(i)))) for i in
                                                                        intermodal_dict.values()]

    arg_list = []
    for mod in modalities:
        arg_list.append(arg_dict[mod])

    arg_list.append(arg_dict['gen'])

    args_dict_all = {}
    models = []
    for d in arg_list:
        if 'mod' in d.keys():
            if len(modality) == 1:
                if any(x in d['mod'] for x in func_models):
                    if 'dwi' in modality:
                        del d['mod']
                elif any(x in d['mod'] for x in struct_models):
                    if 'func' in modality:
                        del d['mod']
            else:
                if any(x in d['mod'] for x in func_models) or any(x in d['mod'] for x in struct_models):
                    models.append(ast.literal_eval(d['mod']))
        args_dict_all.update(d)

    if len(modality) > 1:
        args_dict_all['mod'] = str(list(set(flatten(models))))

    print('Arguments parsed from bids_config.json:\n')
    print(args_dict_all)

    for key, val in args_dict_all.items():
        if isinstance(val, str):
            args_dict_all[key] = ast.literal_eval(val)

    id_list = []
    for i in sorted(list(set(subjs))):
        for ses in sorted(list(set(seshs))):
            id_list.append(i + '_' + ses)

    args_dict_all['work'] = bids_args.work
    args_dict_all['output_dir'] = output_dir
    args_dict_all['plug'] = bids_args.plug
    args_dict_all['pm'] = bids_args.pm
    args_dict_all['v'] = bids_args.v
    if funcs is not None:
        args_dict_all['func'] = sorted(funcs)
    else:
        args_dict_all['func'] = None
    if confs is not None:
        args_dict_all['conf'] = sorted(confs)
    else:
        args_dict_all['conf'] = None
    if dwis is not None:
        args_dict_all['dwi'] = sorted(dwis)
        args_dict_all['bval'] = sorted(bvals)
        args_dict_all['bvec'] = sorted(bvecs)
    else:
        args_dict_all['dwi'] = None
        args_dict_all['bval'] = None
        args_dict_all['bvec'] = None
    if anats is not None:
        args_dict_all['anat'] = sorted(anats)
    else:
        args_dict_all['anat'] = None
    if masks is not None:
        args_dict_all['m'] = sorted(masks)
    else:
        args_dict_all['m'] = None
    args_dict_all['g'] = None
    if ('dwi' in modality) and (bids_args.way is not None):
        args_dict_all['way'] = bids_args.way
    else:
        args_dict_all['way'] = None
    args_dict_all['id'] = id_list
    args_dict_all['ua'] = bids_args.ua
    args_dict_all['ref'] = bids_args.ref
    args_dict_all['roi'] = bids_args.roi
    args_dict_all['templ'] = bids_args.templ
    args_dict_all['templm'] = bids_args.templm
    if ('func' in modality) and (bids_args.cm is not None):
        args_dict_all['cm'] = bids_args.cm
    else:
        args_dict_all['cm'] = None

    # Mimic argparse with SimpleNamespace object
    args = SimpleNamespace(**args_dict_all)
    print(args)

