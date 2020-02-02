"""
BIDS-functions to return inputs for the run.py functions.
"""
import bids
from itertools import product
from pynets.pynets_run import build_workflow
from pynets.core.utils import as_list, merge_dicts


def sweep_directory(bdir, subj=None, sesh=None, task=None, run=None, modality=None):
    """
    Given a BIDs formatted directory, crawls the BIDs dir and prepares the
    necessary inputs for the PyNets pipeline. Uses regexes to check matches for
    BIDs compliance.
    """
    if modality == 'dwi':
        dwis = []
        bvals = []
        bvecs = []
    elif modality == 'func':
        funcs = []
    anats = []
    # initialize BIDs tree on bdir
    layout = bids.layout.BIDSLayout(bdir, validate=False)
    # get all files matching the specific modality we are using
    if not subj:
        # list of all the subjects
        subjs = layout.get_subjects()
    else:
        # make it a list so we can iterate
        subjs = as_list(subj)
        assert subj in subjs, "subject {} is not in the bids folder".format(subj)
    for sub in subjs:
        if not sesh:
            seshs = layout.get_sessions(subject=sub, derivatives=False)
            # in case there are non-session level inputs
            seshs += [None]
        else:
            # make a list so we can iterate
            seshs = as_list(sesh)

        if not task:
            tasks = layout.get_tasks(subject=sub, derivatives=False)
            tasks += [None]
        else:
            tasks = as_list(task)

        if not run:
            runs = layout.get_runs(subject=sub, derivatives=False)
            runs += [None]
        else:
            runs = as_list(run)

        print('\n')
        print(sub)
        print("%s%s" % ('Subject:', sub))
        print("%s%s" % ('Sessions:', seshs))
        print("%s%s" % ('Tasks:', tasks))
        print("%s%s" % ('Runs:', runs))
        print('\n')
        # all the combinations of sessions and tasks that are possible
        for (ses, tas, ru) in product(seshs, tasks, runs):
            # the attributes for our modality img
            mod_attributes = [sub, ses, tas, ru]
            # the keys for our modality img
            mod_keys = ['subject', 'session', 'task', 'run']
            # our query we will use for each modality img
            mod_query = {'datatype': modality}
            if modality == 'dwi':
                type_img = 'dwi'  # use the dwi image
            elif modality == 'func':
                type_img = 'bold'  # use the bold image
            else:
                raise ValueError('ERROR: No valid bids modality specified.')
            mod_query['suffix'] = type_img

            for attr, key in zip(mod_attributes, mod_keys):
                if attr:
                    mod_query[key] = attr

            anat_attributes = [sub, ses]  # the attributes for our anat img
            anat_keys = ['subject', 'session']  # the keys for our modality img
            # our query for the anatomical image
            anat_query = {'datatype': 'anat', 'suffix': 'T1w',
                          'extensions': 'nii.gz|nii'}
            for attr, key in zip(anat_attributes, anat_keys):
                if attr:
                    anat_query[key] = attr
            # make a query to fine the desired files from the BIDSLayout
            anat = layout.get(**anat_query)
            if modality == 'dwi':
                dwi = layout.get(**merge_dicts(mod_query, {'extensions': 'nii.gz|nii'}))
                bval = layout.get(**merge_dicts(mod_query, {'extensions': 'bval'}))
                bvec = layout.get(**merge_dicts(mod_query, {'extensions': 'bvec'}))
                if dwi and bval and bvec:
                    for (dw, bva, bve) in zip(dwi, bval, bvec):
                        if dw.filename not in dwis:
                            # if all the required files exist, append by the first
                            # match (0 index)
                            if anat:
                                anats.append(anat[0].filename)
                            dwis.append(dw.filename)
                            bvals.append(bva.filename)
                            bvecs.append(bve.filename)

            elif modality == 'func':
                func = layout.get(**merge_dicts(mod_query, {'extensions': 'nii.gz|nii'}))
                if func and anat:
                    for fun in func:
                        if fun.filename not in funcs:
                            funcs.append(fun.filename)
                            anats.append(anat[0].filename)
    if modality == 'dwi':
        if not len(dwis) or not len(bvals) or not len(bvecs):
            print("No dMRI files found in BIDs spec. Skipping...")
        return dwis, bvals, bvecs, anats

    elif modality == 'func':
        if not len(funcs):
            print("No fMRI files found in BIDs spec. Skipping...")
        if anats:
            return funcs, None, None, anats
        else:
            return funcs, None, None, None
    else:
        raise ValueError('Incorrect modality passed. Choices are \'func\' and \'dwi\'.')


def main(bids_dir, subject, modality, output_dir):
    outs = sweep_directory(bids_dir, subj=subject, sesh=None, task=None, run=None, modality=modality)
    return outs, output_dir


# class build_args(object):
#     def __init__(self, ):
#         self.func_file = args.func
#         self.mask = args.m
#         self.dwi = args.dwi
#         self.bval = args.bval
#         self.bvec = args.bvec
#         self.ID = args.id
#         self.conf = conf
#         self.anat_loc = args.anat
#         self.ref_txt = args.ref
#
#         resources = args.pm
#         if resources:
#             self.procmem = list(eval(str(resources)))
#         else:
#             from multiprocessing import cpu_count
#             nthreads = cpu_count()
#             self.procmem = [int(nthreads), int(float(nthreads) * 2)]
#
#         self.node_size = args.ns
#         self.node_size_list = args.node_size_list
#         self.smooth = args.sm
#         self.smooth_list = args.smooth_list
#         self.conn_model = args.conn_model
#         self.conn_model_list = args.conn_model_list
#         self.dens_thresh = args.dt
#         self.disp_filt = args.df
#         self.clust_type = args.ct
#         self.clust_type_list = clust_type_list
#         self.clust_mask = args.cm
#         self.clust_mask_list = args.clust_mask_list
#         self.network = args.n
#         self.multi_nets = multi_nets
#         self.uatlas_select = args.ua
#         self.user_atlas_list = user_atlas_list
#         self.atlas_select = args.a
#         self.multi_atlas = multi_atlas
#
#         self.c_boot = args.b
#         self.block_size = args.bs
#         self.thr = args.thr
#         self.roi = args.roi
#         self.adapt_thresh = args.at
#         self.plot_switch = args.plt
#         self.min_thr = args.min_thr
#         self.max_thr = args.max_thr
#         self.step_thr = args.step_thr
#         self.parc = args.parc
#         self.k = args.k
#         self.k_min = args.k_min
#         self.k_max = args.k_max
#         self.k_step = args.k_step
#         self.prune = args.prune
#         self.norm = args.norm
#         self.binary = args.bin
#         self.plugin_type = args.plug
#         self.use_AAL_naming = args.names
#         self.verbose = args.v


if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    bids_dir = '/Users/derekpisner/Downloads/test_subs/HNU1'
    subject = '0025427'
    output_dir = '/Users/derekpisner/Downloads/test_subs/outputs'
    modality = 'dwi'
    #[outs, output_dir] = main(bids_dir, subject, modality, output_dir)

    #kargs = build_args()
    #build_workflow(kargs)
