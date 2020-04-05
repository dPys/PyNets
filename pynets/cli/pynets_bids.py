"""
PyNets BIDS cli
"""
import bids
from pynets.core.utils import as_list, merge_dicts


def sweep_directory(derivatives_path, modality, space='MNI152NLin2009cAsym', func_desc='smoothAROMAnonaggr', subj=None,
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
            anat_attributes = [sub, ses]
            anat_keys = ['subject', 'session']
            # our query for the anatomical image
            anat_query = {'datatype': 'anat', 'suffix': 'T1w',
                          'extensions': ['.nii', '.nii.gz']}
            for attr, key in zip(anat_attributes, anat_keys):
                if attr:
                    anat_query[key] = attr
            # make a query to find the desired files from the BIDSLayout
            anat = layout.get(**anat_query)
            anat = [i for i in anat if 'MNI' not in i.filename and 'space' not in i.filename]

            if anat:
                for an in anat:
                    anats.append(an.path)

            if modality == 'dwi':
                dwi = layout.get(**merge_dicts(mod_query, {'extensions': ['.nii', '.nii.gz'], 'suffix': ['dwi']}))
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
                func = layout.get(**merge_dicts(mod_query, {'extensions': ['.nii', '.nii.gz'],
                                                            'suffix': ['bold', 'masked', func_desc],
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
            return None, None, None, None, None, None, None, subjs, seshs
        else:
            return None, None, dwis, bvals, bvecs, anats, masks, subjs, seshs

    elif modality == 'func':
        if not len(funcs):
            print("No fMRI files found in BIDs spec. Skipping...")
            return None, None, None, None, None, None, None, subjs, seshs
        else:
            return funcs, confs, None, None, None, anats, masks, subjs, seshs
    else:
        raise ValueError('Incorrect modality passed. Choices are \'func\' and \'dwi\'.')


def get_bids_parser():
    """Parse command-line inputs"""
    import argparse

    # Parse args
    # Primary inputs
    parser = argparse.ArgumentParser(description='PyNets BIDS CLI: A Fully-Automated Workflow for Reproducible '
                                                 'Ensemble Sampling of Functional and Structural Connectomes')
    parser.add_argument("input_dir",
                        help="""The directory with the input dataset formatted according to the BIDS standard. To use 
                        data from s3, just pass `s3://<bucket>/<dataset>` as the input directory.""")
    parser.add_argument("output_dir",
                        help="""The directory to store pynets derivatives locally.""")
    parser.add_argument("modality",
                        metavar='modality',
                        nargs='+',
                        choices=['dwi', 'func'],
                        help='Specify data modality to process from bids directory. Options are `dwi` and `func`.')
    parser.add_argument("--participant_label",
                        help="""The label(s) of the participant(s) that should be analyzed. The label corresponds to 
                            sub-<participant_label> from the BIDS spec (so it does not include "sub-"). If this 
                            parameter is not provided all subjects should be analyzed. Multiple participants can be 
                            specified with a space separated list.""",
                        nargs="+",
                        default=None)
    parser.add_argument("--session_label",
                        help="""The label(s) of the session that should be analyzed. The label  corresponds to
                         ses-<participant_label> from the BIDS spec (so it does not include "ses-"). If this parameter 
                         is not provided all sessions should be analyzed. Multiple sessions can be specified with a 
                         space separated list.""",
                        nargs="+",
                        default=None)
    parser.add_argument("--push_location",
                        action="store",
                        help="Name of folder on s3 to push output data to, if the folder does not exist, it will be "
                             "created. Format the location as `s3://<bucket>/<path>`",
                        default=None)

    # Secondary file inputs
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
    parser.add_argument('-way',
                        metavar='Path to binarized Nifti1Image to constrain tractography',
                        default=None,
                        nargs='+',
                        help='Optionally specify a binarized ROI mask in MNI-space to constrain tractography in the '
                             'case of dmri connectome estimation.\n')

    # Debug/Runtime settings
    parser.add_argument('-config',
                        metavar='Optional path to a config.json file with runtime settings.',
                        default=None,
                        help='Including this flag will override the bids_config.json template in the base directory '
                             'of pynets. See the template ad `pynets -h` for available settings.\n')
    parser.add_argument('-pm',
                        metavar='Cores,memory',
                        default='4,8',
                        help='Number of cores to use, number of GB of memory to use for single subject run, entered as '
                             'two integers seperated by comma.\n')
    parser.add_argument('-plug',
                        metavar='Scheduler type',
                        default='MultiProc',
                        nargs=1,
                        choices=['Linear', 'MultiProc', 'SGE', 'PBS', 'SLURM', 'SGEgraph', 'SLURMgraph',
                                 'LegacyMultiProc'],
                        help='Include this flag to specify a workflow plugin other than the default MultiProc.\n')
    parser.add_argument('-v',
                        default=False,
                        action='store_true',
                        help='Verbose print for debugging.\n')
    parser.add_argument('-work',
                        metavar='Working directory',
                        default='/tmp/work',
                        help='Specify the path to a working directory for pynets to run. Default is /tmp/work.\n')
    return parser


def main():
    """Initializes main script from command-line call to generate single-subject or multi-subject workflow(s)"""
    import os
    import sys
    import json
    import ast
    import yaml
    from types import SimpleNamespace
    from pathlib import Path
    try:
        import pynets
    except ImportError:
        print('PyNets not installed! Ensure that you are referencing the correct site-packages and using Python3.6+')

    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h flag.\n")
        sys.exit()

    print('Obtaining Derivatives Layout...')

    modalities = ['func', 'dwi']

    bids_args = get_bids_parser().parse_args()
    participant_label = bids_args.participant_label
    session_label = bids_args.session_label
    modality = bids_args.modality
    bids_config = bids_args.config

    if bids_config:
        with open(bids_config, 'r') as stream:
            arg_dict = json.load(stream)
    else:
        with open("%s%s" % (str(Path(__file__).parent.parent), '/bids_config.json'), 'r') as stream:
        # with open('/Users/derekpisner/Applications/PyNets/pynets/bids_config.json') as stream:
            arg_dict = json.load(stream)

    # Available functional and structural connectivity models
    with open("%s%s" % (str(Path(__file__).parent.parent), '/runconfig.yaml'), 'r') as stream:
    # with open('/Users/derekpisner/Applications/PyNets/pynets/runconfig.yaml') as stream:
        hardcoded_params = yaml.load(stream)
        try:
            func_models = hardcoded_params['available_models']['func_models']
        except KeyError:
            print('ERROR: available functional models not successfully extracted from runconfig.yaml')
            sys.exit()
        try:
            struct_models = hardcoded_params['available_models']['struct_models']
        except KeyError:
            print('ERROR: available structural models not successfully extracted from runconfig.yaml')
            sys.exit()

    # S3
    # Primary inputs
    s3 = bids_args.input_dir.startswith("s3://")

    # secondary inputs
    sec_s3_objs = []
    if isinstance(bids_args.ua, list):
        for i in bids_args.ua:
            if i.startswith("s3://"):
                print('Downloading user atlas: ', i, ' from S3...')
                sec_s3_objs.append(i)
    if isinstance(bids_args.cm, list):
        for i in bids_args.cm:
            if i.startswith("s3://"):
                print('Downloading clustering mask: ', i, ' from S3...')
                sec_s3_objs.append(i)
    if isinstance(bids_args.roi, list):
        for i in bids_args.roi:
            if i.startswith("s3://"):
                print('Downloading ROI mask: ', i, ' from S3...')
                sec_s3_objs.append(i)
    if isinstance(bids_args.way, list):
        for i in bids_args.way:
            if i.startswith("s3://"):
                print('Downloading tractography waymask: ', i, ' from S3...')
                sec_s3_objs.append(i)

    if bids_args.templ:
        if bids_args.templ.startswith("s3://"):
            print('Downloading brain template: ', bids_args.templ, ' from S3...')
            sec_s3_objs.append(bids_args.templ)
    if bids_args.templm:
        if bids_args.templm.startswith("s3://"):
            print('Downloading brain template mask: ', bids_args.templm, ' from S3...')
            sec_s3_objs.append(bids_args.templm)
    if bids_args.ref:
        if bids_args.ref.startswith("s3://"):
            print('Downloading atlas labeling reference file: ', bids_args.ref, ' from S3...')
            sec_s3_objs.append(bids_args.ref)

    if s3 or len(sec_s3_objs) > 0:
        from boto3.session import Session
        from pynets.core import cloud_utils
        from pynets.core.utils import as_directory

        home = os.path.expanduser("~")
        creds = bool(cloud_utils.get_credentials())

        if s3:
            buck, remo = cloud_utils.parse_path(bids_args.input_dir)
            input_dir = as_directory(home + "/.pynets/input", remove=False)
            if (not creds) and bids_args.push_location:
                raise AttributeError("""No AWS credentials found, but "--push_location" flag called. Pushing will most 
                likely fail.""")
            else:
                output_dir = as_directory(home + "/.pynets/output", remove=False)

            # Get S3 input data if needed
            if participant_label and session_label:
                info = "sub-" + participant_label[0] + '/ses-' + session_label[0] + '/' + modality[0]
            cloud_utils.s3_get_data(buck, remo, input_dir, info=info)

        if len(sec_s3_objs) > 0:
            [access_key, secret_key] = cloud_utils.get_credentials()

            session = Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )

            s3_r = session.resource('s3')
            s3_c = cloud_utils.s3_client(service="s3")
            sec_dir = as_directory(home + "/.pynets/secondary_files", remove=False)
            for s3_obj in [i for i in sec_s3_objs if i is not None]:
                buck, remo = cloud_utils.parse_path(s3_obj)
                bucket = s3_r.Bucket(buck)
                s3_c.download_file(buck, remo, sec_dir + '/' + os.path.basename(s3_obj))

            if isinstance(bids_args.ua, list):
                local_ua = bids_args.ua.copy()
                for i in local_ua:
                    if i.startswith("s3://"):
                        local_ua[local_ua.index(i)] = sec_dir + '/' + os.path.basename(i)
                bids_args.ua = local_ua
            if isinstance(bids_args.cm, list):
                local_cm = bids_args.cm.copy()
                for i in bids_args.cm:
                    if i.startswith("s3://"):
                        local_cm[local_cm.index(i)] = sec_dir + '/' + os.path.basename(i)
                bids_args.cm = local_cm
            if isinstance(bids_args.roi, list):
                local_roi = bids_args.roi.copy()
                for i in bids_args.roi:
                    if i.startswith("s3://"):
                        local_roi[local_roi.index(i)] = sec_dir + '/' + os.path.basename(i)
                bids_args.roi = local_roi
            if isinstance(bids_args.way, list):
                local_way = bids_args.way.copy()
                for i in bids_args.way:
                    if i.startswith("s3://"):
                        local_way[local_way.index(i)] = sec_dir + '/' + os.path.basename(i)
                bids_args.way = local_way

            if bids_args.templ:
                if bids_args.templ.startswith("s3://"):
                    bids_args.templ = sec_dir + '/' + os.path.basename(bids_args.templ)
            if bids_args.templm:
                if bids_args.templm.startswith("s3://"):
                    bids_args.templm = sec_dir + '/' + os.path.basename(bids_args.templm)
            if bids_args.ref:
                if bids_args.ref.startswith("s3://"):
                    bids_args.ref = sec_dir + '/' + os.path.basename(bids_args.ref)
    else:
        output_dir = bids_args.output_dir
        if output_dir is None:
            raise ValueError('Must specify an output directory')

    arg_list = []
    if len(modality) > 1:
        for mod in modality:
            outs = sweep_directory(input_dir, modality=mod, subj=bids_args.participant_label[0],
                                   sesh=bids_args.session_label[0])
    else:
        outs = sweep_directory(input_dir, modality=modality[0], subj=bids_args.participant_label[0],
                               sesh=bids_args.session_label[0])
    for mod in modalities:
        arg_list.append(arg_dict[mod])

    arg_list.append(arg_dict['gen'])

    args_dict_all = {}
    for d in arg_list:
        if 'mod' in d.keys():
            if len(modality) == 1:
                if any(x in d['mod'] for x in func_models):
                    if modality[0] == 'dwi':
                        del d['mod']
                if any(x in d['mod'] for x in struct_models):
                    if modality[0] == 'func':
                        del d['mod']
        args_dict_all.update(d)

    print('Arguments parsed from bids_config.json:\n')
    print(args_dict_all)

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

    args_dict_all['work'] = bids_args.work
    args_dict_all['output_dir'] = output_dir
    args_dict_all['plug'] = bids_args.plug
    args_dict_all['pm'] = bids_args.pm
    args_dict_all['v'] = bids_args.v
    args_dict_all['func'] = funcs
    args_dict_all['conf'] = confs
    args_dict_all['dwi'] = dwis
    args_dict_all['bval'] = bvals
    args_dict_all['bvec'] = bvecs
    args_dict_all['anat'] = anats
    args_dict_all['m'] = masks
    args_dict_all['g'] = None
    args_dict_all['way'] = bids_args.way
    args_dict_all['id'] = id_list
    args_dict_all['ua'] = bids_args.ua
    args_dict_all['ref'] = bids_args.ref
    args_dict_all['roi'] = bids_args.roi
    args_dict_all['templ'] = bids_args.templ
    args_dict_all['templm'] = bids_args.templm
    if modality[0] == 'func':
        args_dict_all['cm'] = bids_args.cm
    else:
        args_dict_all['cm'] = None

    # Mimic argparse with SimpleNamespace object
    args = SimpleNamespace(**args_dict_all)

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

    if bids_args.push_location:
        print(f"Pushing to s3 at {bids_args.push_location}.")
        push_buck, push_remo = cloud_utils.parse_path(bids_args.push_location)
        for id in id_list:
            cloud_utils.s3_push_data(
                push_buck,
                push_remo,
                output_dir,
                subject=id.split('_')[0],
                session=id.split('_')[1],
                creds=creds,
            )
    return


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    main()
