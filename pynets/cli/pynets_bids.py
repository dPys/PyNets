"""
PyNets BIDS cli
"""
import bids

from pynets.core.utils import as_list, merge_dicts


def sweep_directory(
        derivatives_path,
        modality,
        space,
        subj=None,
        sesh=None,
        run=None):
    """
    Given a BIDS derivatives directory containing preprocessed functional MRI
    or diffusion MRI data (e.g. fMRIprep or dMRIprep), crawls the outputs and
    prepares necessary inputs for the PyNets pipeline.

    *Note: Since this function searches for derivative file inputs, it does
     not impose strict BIDS compliance, which can therefore create errors in
     the case that files are missing or redundant. Please ensure that there
     redundant files are removed and that BIDS naming conventions are
     followed closely.
    """

    if modality == "dwi":
        dwis = []
        bvals = []
        bvecs = []
    elif modality == "func":
        funcs = []
        confs = []
    masks = []
    anats = []

    # initialize BIDs tree on derivatives_path
    layout = bids.layout.BIDSLayout(
        derivatives_path, validate=False, derivatives=True, absolute_paths=True
    )

    # get all files matching the specific modality we are using
    all_subs = layout.get_subjects()
    if not subj:
        # list of all the subjects
        subjs = all_subs
    elif isinstance(subj, list):
        subjs = [sub for sub in subj if sub in all_subs]
    else:
        # make it a list so we can iterate
        subjs = as_list(subj)

    # Accommodate for different spaces
    if space is None:
        if modality == "dwi":
            spaces = layout.get_spaces(
                suffix="dwi", extension=[
                    ".nii", ".nii.gz"])
        elif modality == "func":
            spaces = layout.get_spaces(
                suffix="bold", extension=[
                    ".nii", ".nii.gz"])
        if spaces:
            spaces = sorted(spaces)
            space = spaces[0]
            if len(spaces) > 1:
                space_list = ", ".join(spaces)
                print(
                    f"No space was provided, but multiple spaces were "
                    f"detected: {space_list}. Selecting the first (ordered"
                    f" lexicographically): {space}")

    for sub in subjs:
        all_seshs = layout.get_sessions(subject=sub)
        if not sesh:
            seshs = all_seshs
            # in case there are non-session level inputs
            seshs += []
        elif isinstance(sesh, list):
            seshs = [ses for ses in sesh if ses in all_seshs]
        else:
            # make a list so we can iterate
            seshs = as_list(sesh)

        print(f"Subject: {sub}\nSession(s): {seshs}\nModality: {modality}")

        for ses in seshs:
            # the attributes for our modality img
            mod_attributes = [sub, ses]
            # the keys for our modality img
            mod_keys = ["subject", "session"]
            # our query we will use for each modality img
            mod_query = {"datatype": modality}

            for attr, key in zip(mod_attributes, mod_keys):
                if attr:
                    mod_query[key] = attr

            # grab anat
            anat_attributes = [sub, ses]
            anat_keys = ["subject", "session"]
            # our query for the anatomical image
            anat_query = {
                "datatype": "anat",
                "suffix": ["T1w", "anat"],
                "extension": [".nii", ".nii.gz"],
            }
            for attr, key in zip(anat_attributes, anat_keys):
                if attr:
                    anat_query[key] = attr
            # make a query to find the desired files from the BIDSLayout
            anat = layout.get(**anat_query)
            anat = [
                i for i in anat if "MNI" not in i.filename and "space" not in
                                   i.filename]
            if len(anat) > 1 and run is not None:
                anat = [i for i in anat if f"run-{run}" in i.filename]

            if anat:
                for an in anat:
                    anats.append(an.path)

            # grab mask
            mask_query = {
                "datatype": "anat",
                "suffix": "mask",
                "extension": [".nii", ".nii.gz"],
            }
            for attr, key in zip(anat_attributes, anat_keys):
                if attr:
                    mask_query[key] = attr

            mask = layout.get(**mask_query)
            if len(mask) > 1 and run is not None:
                mask = [i for i in mask if f"run-{run}" in i.filename]
            mask = [
                i for i in mask if "MNI" not in i.filename and "space" not in
                                   i.filename]

            if modality == "dwi":
                dwi = layout.get(
                    **merge_dicts(
                        mod_query,
                        {"extension": [".nii", ".nii.gz"],
                         },
                    )
                )
                if len(dwi) > 1 and run is not None:
                    dwi = [i for i in dwi if f"run-{run}" in i.filename]
                bval = layout.get(
                    **merge_dicts(mod_query, {"extension": ["bval", "bvals"]}))
                if len(bval) > 1 and run is not None:
                    bval = [i for i in bval if f"run-{run}" in i.filename]
                bvec = layout.get(
                    **merge_dicts(mod_query, {"extension": ["bvec", "bvecs"]}))
                if len(bvec) > 1 and run is not None:
                    bvec = [i for i in bvec if f"run-{run}" in i.filename]

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

            elif modality == "func":
                func = layout.get(
                    **merge_dicts(
                        mod_query,
                        {
                            "extension": [".nii", ".nii.gz"],
                        },
                    )
                )
                if len(func) > 1 and run is not None:
                    func = [i for i in func if f"run-{run}" in i.filename]
                if len(func) > 1 and space is not None:
                    if "MNI" in [i.filename for i in func]:
                        raise ValueError('MNI-space BOLD images are not '
                                         'currently supported, but are all '
                                         'that are currently detected. '
                                         'Is a T1w/anat-coregistered '
                                         'preprocessed BOLD image available? '
                                         'See documentation for more details.')
                    else:
                        func = [i for i in func if space in i.filename]

                conf = layout.get(
                    **merge_dicts(mod_query, {"extension":
                                              [".tsv", ".tsv.gz"]})
                )
                conf = [i for i in conf if "confounds_regressors" in
                        i.filename]
                if len(conf) > 1 and run is not None:
                    conf = [i for i in conf if f"run-{run}" in i.filename]

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

    if modality == "dwi":
        if not len(dwis) or not len(bvals) or not len(bvecs):
            print("No dMRI files found in BIDS spec. Skipping...\n")
            return None, None, None, None, None, None, None, subjs, seshs
        else:
            return None, None, dwis, bvals, bvecs, anats, masks, subjs, seshs

    elif modality == "func":
        if not len(funcs):
            print("No fMRI files found in BIDS spec. Skipping...\n")
            return None, None, None, None, None, None, None, subjs, seshs
        else:
            return funcs, confs, None, None, None, anats, masks, subjs, seshs
    else:
        raise ValueError(
            "Incorrect modality passed. Choices are 'func' and 'dwi'.")


def get_bids_parser():
    """Parse command-line inputs"""
    import argparse

    # Parse args
    # Primary inputs
    parser = argparse.ArgumentParser(
        description="PyNets BIDS CLI: A Fully-Automated Workflow for "
                    "Reproducible Ensemble Sampling of Functional and "
                    "Structural Connectomes")
    parser.add_argument(
        "bids_dir",
        help="""The directory with the input dataset formatted according to
         the BIDS standard. To use data from s3, just pass
          `s3://<bucket>/<dataset>` as the input directory.""",
    )
    parser.add_argument(
        "output_dir",
        help="""The directory to store pynets derivatives locally.""")
    parser.add_argument(
        "analysis_level",
        choices=["participant", "group"],
        help="Whether to instantiate an individual or group workflow",
    )
    parser.add_argument(
        "modality",
        nargs="+",
        choices=[
            "dwi",
            "func"],
        help="Specify data modality to process from bids directory. "
             "Options are `dwi` and `func`.",
    )
    parser.add_argument(
        "--participant_label",
        help="""The label(s) of the participant(s) that should be analyzed.
        The label corresponds to sub-<participant_label> from the BIDS spec
        (so it does not include "sub-"). If this parameter is not provided all
        subjects found in `bids_dir` will be analyzed. Multiple participants
        can be specified with a space separated list.""",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--session_label",
        help="""The label(s) of the session that should be analyzed.
        The label  corresponds to ses-<participant_label> from the BIDS spec
        (so it does not include "ses-"). If this parameter is not provided
        all sessions should be analyzed. Multiple sessions can be specified
         with a space separated list.""",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--run_label",
        help="""The label(s) of the run, if any, within a given session that
        should be analyzed. The label corresponds to run-<run_label> from the
        BIDS spec (so it does not include "run-"). If this parameter
        is not provided all runs should be analyzed. Specifying multiple runs
        is not yet supported.""",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--push_location",
        action="store",
        help="Name of folder on s3 to push output data to, "
             "if the folder does not exist, it will be created. "
             "Format the location as `s3://<bucket>/<path>`",
        default=None,
    )

    # Secondary file inputs
    parser.add_argument(
        "-ua",
        metavar="Path to parcellation file in MNI-space",
        default=None,
        nargs="+",
        help="Optionally specify a path to a parcellation/atlas Nifti1Image "
             "file in MNI152 space. Labels should be spatially distinct across"
             " hemispheres and ordered with consecutive integers with a value "
             "of 0 as the background label. If specifying a list of paths to "
             "multiple user atlases, separate them by space.\n",
    )
    parser.add_argument(
        "-cm",
        metavar="Cluster mask",
        default=None,
        nargs="+",
        help="Optionally specify the path to a Nifti1Image mask file to "
             "constrained functional clustering. If specifying a list of "
             "paths to multiple cluster masks, separate them by space.\n",
    )
    parser.add_argument(
        "-roi",
        metavar="Path to binarized Region-of-Interest (ROI) Nifti1Image in "
                "template MNI space",
        default=None,
        nargs="+",
        help="Optionally specify a binarized ROI mask in template MNI space "
             "and retain only those nodes of a parcellation contained within "
             "that mask for connectome estimation.\n",
    )
    parser.add_argument(
        "-ref",
        metavar="Atlas reference file path",
        default=None,
        help="Specify the path to the atlas reference .txt file that maps "
             "labels to intensities corresponding to the atlas parcellation "
             "file specified with the -ua flag.\n",
    )
    parser.add_argument(
        "-way",
        metavar="Path to binarized Nifti1Image to constrain tractography",
        default=None,
        nargs="+",
        help="Optionally specify a binarized ROI mask in template MNI-space "
             "to constrain tractography in the case of dmri connectome "
             "estimation.\n",
    )

    # Debug/Runtime settings
    parser.add_argument(
        "-config",
        metavar="Optional path to a config.json file with runtime settings.",
        default=None,
        help="Including this flag will override the bids_config.json template "
             "in the base directory of pynets. See the template ad "
             "`pynets -h` for available settings.\n",
    )
    parser.add_argument(
        "-pm",
        metavar="Cores,memory",
        default="auto",
        help="Number of cores to use, number of GB of memory to use for "
             "single subject run, entered as two integers seperated by comma. "
             "Otherwise, default is `auto`, which uses all resources "
             "detected on the current compute node.\n",
    )
    parser.add_argument(
        "-plug",
        metavar="Scheduler type",
        default="MultiProc",
        nargs=1,
        choices=[
            "Linear",
            "MultiProc",
            "SGE",
            "PBS",
            "SLURM",
            "SGEgraph",
            "SLURMgraph",
            "LegacyMultiProc",
        ],
        help="Include this flag to specify a workflow plugin other than the "
             "default MultiProc.\n",
    )
    parser.add_argument(
        "-v",
        default=False,
        action="store_true",
        help="Verbose print for debugging.\n")
    parser.add_argument(
        "-clean",
        default=False,
        action="store_true",
        help="Clean up temporary runtime directory after workflow "
             "termination.\n",
    )
    parser.add_argument(
        "-work",
        metavar="Working directory",
        default="/tmp/work",
        help="Specify the path to a working directory for pynets to run. "
             "Default is /tmp/work.\n",
    )
    return parser


def main():
    """Initializes main script from command-line call to generate
    single-subject or multi-subject workflow(s)"""
    import os
    import gc
    import sys
    import json
    from pynets.core.utils import build_args_from_config
    import itertools
    from types import SimpleNamespace
    import pkg_resources
    from pynets.core.utils import flatten
    from pynets.cli.pynets_run import build_workflow
    from multiprocessing import set_start_method, Process, Manager
    from colorama import Fore, Style

    try:
        import pynets
    except ImportError:
        print(
            "PyNets not installed! Ensure that you are referencing the correct"
            " site-packages and using Python3.6+"
        )

    if len(sys.argv) < 1:
        print("\nMissing command-line inputs! See help options with the -h"
              " flag.\n")
        sys.exit(1)

    print(f"{Fore.LIGHTBLUE_EX}\nBIDS API\n")

    print(Style.RESET_ALL)

    print(f"{Fore.LIGHTGREEN_EX}Obtaining Derivatives Layout...")

    print(Style.RESET_ALL)

    modalities = ["func", "dwi"]
    space = 'T1w'

    bids_args = get_bids_parser().parse_args()
    participant_label = bids_args.participant_label
    session_label = bids_args.session_label
    run = bids_args.run_label
    if isinstance(run, list):
        run = str(run[0]).zfill(2)
    modality = bids_args.modality
    bids_config = bids_args.config
    analysis_level = bids_args.analysis_level
    clean = bids_args.clean

    if analysis_level == "group" and participant_label is not None:
        raise ValueError(
            "Error: You have indicated a group analysis level run, but"
            " specified a participant label!"
        )

    if analysis_level == "participant" and participant_label is None:
        raise ValueError(
            "Error: You have indicated a participant analysis level run, but"
            " not specified a participant "
            "label!")

    if bids_config:
        with open(bids_config, "r") as stream:
            arg_dict = json.load(stream)
    else:
        with open(
            pkg_resources.resource_filename("pynets",
                                            "config/bids_config.json"),
            "r",
        ) as stream:
            arg_dict = json.load(stream)
        stream.close()

    # S3
    # Primary inputs
    s3 = bids_args.bids_dir.startswith("s3://")

    if not s3:
        bids_dir = bids_args.bids_dir

    # secondary inputs
    sec_s3_objs = []
    if isinstance(bids_args.ua, list):
        for i in bids_args.ua:
            if i.startswith("s3://"):
                print("Downloading user atlas: ", i, " from S3...")
                sec_s3_objs.append(i)
    if isinstance(bids_args.cm, list):
        for i in bids_args.cm:
            if i.startswith("s3://"):
                print("Downloading clustering mask: ", i, " from S3...")
                sec_s3_objs.append(i)
    if isinstance(bids_args.roi, list):
        for i in bids_args.roi:
            if i.startswith("s3://"):
                print("Downloading ROI mask: ", i, " from S3...")
                sec_s3_objs.append(i)
    if isinstance(bids_args.way, list):
        for i in bids_args.way:
            if i.startswith("s3://"):
                print("Downloading tractography waymask: ", i, " from S3...")
                sec_s3_objs.append(i)

    if bids_args.ref:
        if bids_args.ref.startswith("s3://"):
            print(
                "Downloading atlas labeling reference file: ",
                bids_args.ref,
                " from S3...",
            )
            sec_s3_objs.append(bids_args.ref)

    if s3 or len(sec_s3_objs) > 0:
        from boto3.session import Session
        from pynets.core import cloud_utils
        from pynets.core.utils import as_directory

        home = os.path.expanduser("~")
        creds = bool(cloud_utils.get_credentials())

        if s3:
            buck, remo = cloud_utils.parse_path(bids_args.bids_dir)
            os.makedirs(f"{home}/.pynets", exist_ok=True)
            os.makedirs(f"{home}/.pynets/input", exist_ok=True)
            os.makedirs(f"{home}/.pynets/output", exist_ok=True)
            bids_dir = as_directory(f"{home}/.pynets/input", remove=False)
            if (not creds) and bids_args.push_location:
                raise AttributeError(
                    """No AWS credentials found, but `--push_location` flag
                     called. Pushing will most likely fail.""")
            else:
                output_dir = as_directory(
                    f"{home}/.pynets/output", remove=False)

            # Get S3 input data if needed
            if analysis_level == "participant":
                for partic, ses in list(
                    itertools.product(participant_label, session_label)
                ):
                    if ses is not None:
                        info = "sub-" + partic + "/ses-" + ses
                    elif ses is None:
                        info = "sub-" + partic
                    cloud_utils.s3_get_data(
                        buck, remo, bids_dir, modality, info=info)
            elif analysis_level == "group":
                if len(session_label) > 1 and session_label[0] != "None":
                    for ses in session_label:
                        info = "ses-" + ses
                        cloud_utils.s3_get_data(
                            buck, remo, bids_dir, modality, info=info
                        )
                else:
                    cloud_utils.s3_get_data(buck, remo, bids_dir, modality)

        if len(sec_s3_objs) > 0:
            [access_key, secret_key] = cloud_utils.get_credentials()

            session = Session(
                aws_access_key_id=access_key, aws_secret_access_key=secret_key
            )

            s3_r = session.resource("s3")
            s3_c = cloud_utils.s3_client(service="s3")
            sec_dir = as_directory(
                home + "/.pynets/secondary_files", remove=False)
            for s3_obj in [i for i in sec_s3_objs if i is not None]:
                buck, remo = cloud_utils.parse_path(s3_obj)
                s3_c.download_file(
                    buck, remo, f"{sec_dir}/{os.path.basename(s3_obj)}")

            if isinstance(bids_args.ua, list):
                local_ua = bids_args.ua.copy()
                for i in local_ua:
                    if i.startswith("s3://"):
                        local_ua[local_ua.index(
                            i)] = f"{sec_dir}/{os.path.basename(i)}"
                bids_args.ua = local_ua
            if isinstance(bids_args.cm, list):
                local_cm = bids_args.cm.copy()
                for i in bids_args.cm:
                    if i.startswith("s3://"):
                        local_cm[local_cm.index(
                            i)] = f"{sec_dir}/{os.path.basename(i)}"
                bids_args.cm = local_cm
            if isinstance(bids_args.roi, list):
                local_roi = bids_args.roi.copy()
                for i in bids_args.roi:
                    if i.startswith("s3://"):
                        local_roi[
                            local_roi.index(i)
                        ] = f"{sec_dir}/{os.path.basename(i)}"
                bids_args.roi = local_roi
            if isinstance(bids_args.way, list):
                local_way = bids_args.way.copy()
                for i in bids_args.way:
                    if i.startswith("s3://"):
                        local_way[
                            local_way.index(i)
                        ] = f"{sec_dir}/{os.path.basename(i)}"
                bids_args.way = local_way

            if bids_args.ref:
                if bids_args.ref.startswith("s3://"):
                    bids_args.ref = f"{sec_dir}/" \
                                    f"{os.path.basename(bids_args.ref)}"
    else:
        output_dir = bids_args.output_dir
        if output_dir is None:
            raise ValueError("Must specify an output directory")

    intermodal_dict = {
        k: []
        for k in [
            "funcs",
            "confs",
            "dwis",
            "bvals",
            "bvecs",
            "anats",
            "masks",
            "subjs",
            "seshs",
        ]
    }
    if analysis_level == "group":
        if len(modality) > 1:
            i = 0
            for mod_ in modality:
                outs = sweep_directory(
                    bids_dir,
                    modality=mod_,
                    space=space,
                    sesh=session_label,
                    run=run
                )
                if mod_ == "func":
                    if i == 0:
                        funcs, confs, _, _, _, anats, masks, subjs, seshs =\
                            outs
                    else:
                        funcs, confs, _, _, _, _, _, _, _ = outs
                    intermodal_dict["funcs"].append(funcs)
                    intermodal_dict["confs"].append(confs)
                elif mod_ == "dwi":
                    if i == 0:
                        _, _, dwis, bvals, bvecs, anats, masks, subjs, seshs =\
                            outs
                    else:
                        _, _, dwis, bvals, bvecs, _, _, _, _ = outs
                    intermodal_dict["dwis"].append(dwis)
                    intermodal_dict["bvals"].append(bvals)
                    intermodal_dict["bvecs"].append(bvecs)
                intermodal_dict["anats"].append(anats)
                intermodal_dict["masks"].append(masks)
                intermodal_dict["subjs"].append(subjs)
                intermodal_dict["seshs"].append(seshs)
                i += 1
        else:
            intermodal_dict = None
            outs = sweep_directory(
                bids_dir,
                modality=modality[0],
                space=space,
                sesh=session_label,
                run=run
            )
            funcs, confs, dwis, bvals, bvecs, anats, masks, subjs, seshs = outs
    elif analysis_level == "participant":
        if len(modality) > 1:
            i = 0
            for mod_ in modality:
                outs = sweep_directory(
                    bids_dir,
                    modality=mod_,
                    space=space,
                    subj=participant_label,
                    sesh=session_label,
                    run=run
                )
                if mod_ == "func":
                    if i == 0:
                        funcs, confs, _, _, _, anats, masks, subjs, seshs =\
                            outs
                    else:
                        funcs, confs, _, _, _, _, _, _, _ = outs
                    intermodal_dict["funcs"].append(funcs)
                    intermodal_dict["confs"].append(confs)
                elif mod_ == "dwi":
                    if i == 0:
                        _, _, dwis, bvals, bvecs, anats, masks, subjs, seshs =\
                            outs
                    else:
                        _, _, dwis, bvals, bvecs, _, _, _, _ = outs
                    intermodal_dict["dwis"].append(dwis)
                    intermodal_dict["bvals"].append(bvals)
                    intermodal_dict["bvecs"].append(bvecs)
                intermodal_dict["anats"].append(anats)
                intermodal_dict["masks"].append(masks)
                intermodal_dict["subjs"].append(subjs)
                intermodal_dict["seshs"].append(seshs)
                i += 1
        else:
            intermodal_dict = None
            outs = sweep_directory(
                bids_dir,
                modality=modality[0],
                space=space,
                subj=participant_label,
                sesh=session_label,
                run=run
            )
            funcs, confs, dwis, bvals, bvecs, anats, masks, subjs, seshs = outs
    else:
        raise ValueError(
            "Analysis level invalid. Must be `participant` or `group`. See"
            " --help."
        )

    if intermodal_dict:
        funcs, confs, dwis, bvals, bvecs, anats, masks, subjs, seshs = [
            list(set(list(flatten(i)))) for i in intermodal_dict.values()
        ]

    args_dict_all = build_args_from_config(modality, arg_dict)

    id_list = []
    for i in sorted(list(set(subjs))):
        for ses in sorted(list(set(seshs))):
            id_list.append(i + "_" + ses)

    args_dict_all["work"] = bids_args.work
    args_dict_all["output_dir"] = output_dir
    args_dict_all["plug"] = bids_args.plug
    args_dict_all["pm"] = bids_args.pm
    args_dict_all["v"] = bids_args.v
    args_dict_all["clean"] = bids_args.clean
    if funcs is not None:
        args_dict_all["func"] = sorted(funcs)
    else:
        args_dict_all["func"] = None
    if confs is not None:
        args_dict_all["conf"] = sorted(confs)
    else:
        args_dict_all["conf"] = None
    if dwis is not None:
        args_dict_all["dwi"] = sorted(dwis)
        args_dict_all["bval"] = sorted(bvals)
        args_dict_all["bvec"] = sorted(bvecs)
    else:
        args_dict_all["dwi"] = None
        args_dict_all["bval"] = None
        args_dict_all["bvec"] = None
    if anats is not None:
        args_dict_all["anat"] = sorted(anats)
    else:
        args_dict_all["anat"] = None
    if masks is not None:
        args_dict_all["m"] = sorted(masks)
    else:
        args_dict_all["m"] = None
    args_dict_all["g"] = None
    if ("dwi" in modality) and (bids_args.way is not None):
        args_dict_all["way"] = bids_args.way
    else:
        args_dict_all["way"] = None
    args_dict_all["id"] = id_list
    args_dict_all["ua"] = bids_args.ua
    args_dict_all["ref"] = bids_args.ref
    args_dict_all["roi"] = bids_args.roi
    if ("func" in modality) and (bids_args.cm is not None):
        args_dict_all["cm"] = bids_args.cm
    else:
        args_dict_all["cm"] = None

    # Mimic argparse with SimpleNamespace object
    args = SimpleNamespace(**args_dict_all)
    print(args)

    set_start_method("forkserver")
    with Manager() as mgr:
        retval = mgr.dict()
        p = Process(target=build_workflow, args=(args, retval))
        p.start()
        p.join()
        if p.is_alive():
            p.terminate()

        retcode = p.exitcode or retval.get("return_code", 0)

        pynets_wf = retval.get("workflow", None)
        work_dir = retval.get("work_dir")
        plugin_settings = retval.get("plugin_settings", None)
        plugin_settings = retval.get("plugin_settings", None)
        execution_dict = retval.get("execution_dict", None)
        run_uuid = retval.get("run_uuid", None)

        retcode = retcode or int(pynets_wf is None)
        if retcode != 0:
            sys.exit(retcode)

        # Clean up master process before running workflow, which may create
        # forks
        gc.collect()

    mgr.shutdown()

    if bids_args.push_location:
        print(f"Pushing to s3 at {bids_args.push_location}.")
        push_buck, push_remo = cloud_utils.parse_path(bids_args.push_location)
        for id in id_list:
            cloud_utils.s3_push_data(
                push_buck,
                push_remo,
                output_dir,
                modality,
                subject=id.split("_")[0],
                session=id.split("_")[1],
                creds=creds,
            )

    sys.exit(0)

    return


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_" \
               "importlib.BuiltinImporter'>)"
    main()
