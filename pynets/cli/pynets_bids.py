"""
PyNets BIDS cli
"""
import bids
from pynets.core.utils import as_list, merge_dicts


def sweep_directory(derivatives_path, modality, space='MNI152NLin6Asym', func_desc='smoothAROMAnonaggr', subj=None,
                    sesh=None):
    """
    Given a BIDS derivatives directory containing preprocessed functional MRI or diffusion MRI data
    (e.g. fMRIprep or dMRIprep), crawls the outputs and prepares necessary inputs for the PyNets pipeline.

    *Note: Since this function searches for derivative file inputs, it does not impose strict BIDS compliance, which
    can therefore create errors in the case that files are missing or redundant. Please ensure that there redundant
    files are removed and that BIDS naming conventions are followed closely.
    """

    if modality == 'dwi':
        dwis = []
        bvals = []
        bvecs = []
    elif modality == 'func':
        funcs = []
        confs = []
    masks = []
    anats = []

    # initialize BIDs tree on derivatives_path
    layout = bids.layout.BIDSLayout(derivatives_path, validate=False, derivatives=True, absolute_paths=True)

    # get all files matching the specific modality we are using
    if not subj:
        # list of all the subjects
        subjs = layout.get_subjects()
    else:
        # make it a list so we can iterate
        subjs = as_list(subj)

    # Accommodate for different spaces
    if space is None:
        if modality == 'dwi':
            spaces = layout.get_spaces(
                suffix='dwi',
                extension=['.nii', '.nii.gz'])
        elif modality == 'func':
            spaces = layout.get_spaces(
                suffix='bold',
                extension=['.nii', '.nii.gz'])
        if spaces:
            spaces = sorted(spaces)
            space = spaces[0]
            if len(spaces) > 1:
                print(
                    'No space was provided, but multiple spaces were detected: %s. '
                    'Selecting the first (ordered lexicographically): %s'
                    % (', '.join(spaces), space))

    for sub in subjs:
        if not sesh:
            seshs = layout.get_sessions(subject=sub)
            # in case there are non-session level inputs
            seshs += []
        else:
            # make a list so we can iterate
            seshs = as_list(sesh)

        print("\n%s%s" % ('Subject(s): ', sub))
        print("%s%s\n" % ('Session(s): ', seshs))

        for ses in seshs:
            # the attributes for our modality img
            mod_attributes = [sub, ses]
            # the keys for our modality img
            mod_keys = ['subject', 'session']
            # our query we will use for each modality img
            mod_query = {'datatype': modality}

            for attr, key in zip(mod_attributes, mod_keys):
                if attr:
                    mod_query[key] = attr

            # grab anat
            anat_attributes = [sub, ses]  # the attributes for our anat img
            anat_keys = ['subject', 'session']  # the keys for our modality img
            # our query for the anatomical image
            anat_query = {'datatype': 'anat', 'suffix': 'T1w',
                          'extensions': ['.nii', '.nii.gz']}
            for attr, key in zip(anat_attributes, anat_keys):
                if attr:
                    anat_query[key] = attr
            # make a query to fine the desired files from the BIDSLayout
            anat = layout.get(**anat_query)
            anat = [i for i in anat if 'MNI' not in i.filename and 'space' not in i.filename]

            if anat:
                for an in anat:
                    anats.append(an.path)

            if modality == 'dwi':
                dwi = layout.get(**merge_dicts(mod_query, {'extensions': ['.nii', '.nii.gz'], 'suffix': 'dwi'}))
                bval = layout.get(**merge_dicts(mod_query, {'extensions': 'bval'}))
                bvec = layout.get(**merge_dicts(mod_query, {'extensions': 'bvec'}))
                mask = layout.get(**merge_dicts(mod_query, {'extensions': ['.nii', '.nii.gz'],
                                                            'suffix': 'mask',
                                                            'desc': 'brain', 'space': space}))
                if dwi and bval and bvec:
                    if not mask:
                        for (dw, bva, bve) in zip(dwi, bval, bvec):
                            if dw.path not in dwis:
                                dwis.append(dw.path)
                                bvals.append(bva.path)
                                bvecs.append(bve.path)
                    else:
                        for (dw, bva, bve, mas) in zip(dwi, bval, bvec, mask):
                            if dw.path not in dwis:
                                dwis.append(dw.path)
                                bvals.append(bva.path)
                                bvecs.append(bve.path)
                                masks.append(mas.path)

            elif modality == 'func':
                func = layout.get(**merge_dicts(mod_query, {'extensions': ['.nii', '.nii.gz'], 'suffix': 'bold',
                                                            'space': space}))
                func = [i for i in func if func_desc in i.filename]
                conf = layout.get(**merge_dicts(mod_query, {'extensions': ['.tsv', '.tsv.gz']}))
                conf = [i for i in conf if 'confounds_regressors' in i.filename]
                mask = layout.get(**merge_dicts(mod_query, {'extensions': ['.nii', '.nii.gz'],
                                                            'suffix': 'mask',
                                                            'desc': 'brain', 'space': space}))
                if func:
                    if not conf and not mask:
                        for fun in func:
                            if fun.path not in funcs:
                                funcs.append(fun.path)
                    elif not conf and mask:
                        for fun, mas in zip(func, mask):
                            if fun.path not in funcs:
                                funcs.append(fun.path)
                                masks.append(mas.path)
                    elif conf and not mask:
                        for fun, con in zip(func, conf):
                            if fun.path not in funcs:
                                funcs.append(fun.path)
                                confs.append(con.path)
                    else:
                        for fun, con, mas in zip(func, conf, mask):
                            if fun.path not in funcs:
                                funcs.append(fun.path)
                                masks.append(mas.path)
                                confs.append(con.path)

    if len(anats) == 0:
        anats = None

    if len(masks) == 0:
        masks = None

    if modality == 'dwi':
        if not len(dwis) or not len(bvals) or not len(bvecs):
            print("No dMRI files found in BIDs spec. Skipping...")
            return None, None, None, None, None, None, subjs, seshs
        else:
            return None, None, dwis, bvals, bvecs, anats, masks, subjs, seshs

    elif modality == 'func':
        if not len(funcs):
            print("No fMRI files found in BIDs spec. Skipping...")
            return None, None, None, None, None, None, subjs, seshs
        else:
            return funcs, confs, None, None, None, anats, masks, subjs, seshs
    else:
        raise ValueError('Incorrect modality passed. Choices are \'func\' and \'dwi\'.')


def get_bids_parser():
    """Parse command-line inputs"""
    import argparse

    # Parse args
    parser = argparse.ArgumentParser(description='PyNets BIDS CLI: A Fully-Automated Workflow for Reproducible '
                                                 'Ensemble Sampling of Functional and Structural Connectomes')
    parser.add_argument(
        "input_dir",
        help="""The directory with the input dataset
        formatted according to the BIDS standard.
        To use data from s3, just pass `s3://<bucket>/<dataset>` as the input directory.""",
    )
    parser.add_argument(
        "modality",
        metavar='modality',
        default=None,
        nargs='+',
        choices=['dwi', 'func'],
        help='Specify data modality to process from bids directory. Options are `dwi` and `func`.'
    ),
    parser.add_argument(
        "--participant_label",
        help="""The label(s) of the
        participant(s) that should be analyzed. The label
        corresponds to sub-<participant_label> from the BIDS
        spec (so it does not include "sub-"). If this
        parameter is not provided all subjects should be
        analyzed. Multiple participants can be specified
        with a space separated list.""",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--session_label",
        help="""The label(s) of the
        session that should be analyzed. The label
        corresponds to ses-<participant_label> from the BIDS
        spec (so it does not include "ses-"). If this
        parameter is not provided all sessions should be
        analyzed. Multiple sessions can be specified
        with a space separated list.""",
        nargs="+",
        default=None,
    )
    parser.add_argument('-ua',
                        metavar='Path to parcellation file in MNI-space',
                        default=None,
                        nargs='+',
                        help='Optionally specify a path to a parcellation/atlas Nifti1Image file in MNI152 space. '
                             'Labels should be spatially distinct across hemispheres and ordered with consecutive '
                             'integers with a value of 0 as the background label. If specifying a list of paths to '
                             'multiple user atlases, separate them by space.\n')
    parser.add_argument('-cm',
                        metavar='Cluster mask',
                        default=None,
                        nargs='+',
                        help='Optionally specify the path to a Nifti1Image mask file to constrained functional '
                             'clustering. If specifying a list of paths to multiple cluster masks, separate '
                             'them by space.\n')
    parser.add_argument('-roi',
                        metavar='Path to binarized Region-of-Interest (ROI) Nifti1Image',
                        default=None,
                        nargs='+',
                        help='Optionally specify a binarized ROI mask and retain only those nodes '
                             'of a parcellation contained within that mask for connectome estimation.\n')
    parser.add_argument('-templ',
                        metavar='Path to template file',
                        default=None,
                        help='Optionally specify a path to a template Nifti1Image file. If none is specified, then '
                             'will use the MNI152 template by default.\n')
    parser.add_argument('-templm',
                        metavar='Path to template mask file',
                        default=None,
                        help='Optionally specify a path to a template mask Nifti1Image file. If none is specified, '
                             'then will use the MNI152 template mask by default.\n')
    parser.add_argument('-ref',
                        metavar='Atlas reference file path',
                        default=None,
                        help='Specify the path to the atlas reference .txt file that maps labels to '
                             'intensities corresponding to the atlas parcellation file specified with the -ua flag.\n')

    return parser


def main():
    """Initializes main script from command-line call to generate single-subject or multi-subject workflow(s)"""
    import sys
    import json
    import ast
    from types import SimpleNamespace
    from pathlib import Path
    try:
        from pynets.core.utils import do_dir_path
    except ImportError:
        print('PyNets not installed! Ensure that you are referencing the correct site-packages and using Python3.5+')

    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h flag.\n")
        sys.exit()

    print('Obtaining Derivatives Layout...')

    modalities = ['func', 'dwi']

    bids_args = get_bids_parser().parse_args()
    modality = bids_args.modality

    with open("%s%s" % (str(Path(__file__).parent.parent), '/bids_config.json'), 'r') as stream:
    # with open('/Users/derekpisner/Applications/PyNets/pynets/bids_config.json') as stream:
        arg_dict = json.load(stream)

    if len(modality) > 1:
        arg_list = []
        for mod in modality:
            outs = sweep_directory(bids_args.input_dir, modality=mod, subj=bids_args.participant_label,
                                   sesh=bids_args.session_label)
            arg_list.append(arg_dict[mod])
    else:
        outs = sweep_directory(bids_args.input_dir, modality=modality[0], subj=bids_args.participant_label,
                               sesh=bids_args.session_label)
        arg_list = [arg_dict[modality[0]]]
        arg_list.append(dict.fromkeys([arg_dict[mod] for mod in modalities if mod != modality[0]][0], None))
    arg_list.append(arg_dict['gen'])

    args_dict_all = {}
    for d in arg_list:
        if 'mod' in d.keys():
            if d['mod'] == None or d['mod'] == [None] or d['mod'] == "None" or d['mod'] == "['None']":
                del d['mod']
        args_dict_all.update(d)

    for key, val in args_dict_all.items():
        if isinstance(val, str):
            args_dict_all[key] = ast.literal_eval(val)

    funcs, confs, dwis, bvals, bvecs, anats, masks, subjs, seshs = outs

    id_list = []
    for i in subjs:
        for ses in seshs:
            id_list.append(i + '_' + ses)
    if len(modality) > 1:
        id_list = id_list*2

    args_dict_all['func'] = funcs
    args_dict_all['conf'] = confs
    args_dict_all['dwi'] = dwis
    args_dict_all['bval'] = bvals
    args_dict_all['bvec'] = bvecs
    args_dict_all['anat'] = anats
    args_dict_all['m'] = masks
    args_dict_all['id'] = id_list
    args_dict_all['ua'] = bids_args.ua
    args_dict_all['ref'] = bids_args.ref
    args_dict_all['roi'] = bids_args.roi
    args_dict_all['templ'] = bids_args.templ
    args_dict_all['templm'] = bids_args.templm
    if modality == 'func':
        args_dict_all['cm'] = bids_args.cm
    else:
        args_dict_all['cm'] = None

    # Mimic argparse with SimpleNamespace object
    args = SimpleNamespace(**args_dict_all)

    try:
        import gc
        from pynets.cli.pynets_run import build_workflow
        from multiprocessing import set_start_method, Process, Manager
        set_start_method('forkserver')
        with Manager() as mgr:
            retval = mgr.dict()
            p = Process(target=build_workflow, args=(args, retval))
            p.start()
            p.join()

            if p.exitcode != 0:
                sys.exit(p.exitcode)

            # Clean up master process before running workflow, which may create forks
            gc.collect()
    except:
        print('\nWARNING: Forkserver failed to initialize. Are you using Python3 ?')
        retval = dict()
        build_workflow(args, retval)

    return


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
