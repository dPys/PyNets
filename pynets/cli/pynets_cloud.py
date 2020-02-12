#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
@author: Derek Pisner (dPys)
"""
import subprocess
import re
import os
import sys
import json
from copy import deepcopy
from collections import OrderedDict
from argparse import ArgumentParser
from pathlib import Path
import pynets
from pynets.core.cloud_utils import get_credentials
from pynets.core.cloud_utils import get_matching_s3_objects
from pynets.core.cloud_utils import s3_client


def batch_submit(
    bucket,
    path,
    jobdir,
    credentials=None,
    dataset=None,
    modif=None
):
    """Searches through an S3 bucket, gets all subject-ids, creates json files for each,
    submits batch jobs, and returns list of job ids to query status upon later

    Parameters
    ----------
    bucket : str
        The S3 bucket with the input dataset formatted according to the BIDS standard.
    path : str
        The directory where the dataset is stored on the S3 bucket
    jobdir : str
        Directory of batch jobs to generate/check up on
    credentials : [type], optional
        AWS formatted csv of credentials, by default None
    dataset : str, optional
        Name given to the output directory containing analyzed data set "pynets-<version>-<dataset>", by default None
    modif : str, optional
        Name of folder on s3 to push to. If empty, push to a folder with pynets's version number, by default ""
    """

    print(f"Getting list from s3://{bucket}/{path}/...")
    threads = crawl_bucket(bucket, path, jobdir)

    print("Generating job for each subject...")
    jobs = create_json(
        bucket,
        path,
        threads,
        jobdir,
        credentials=credentials,
        dataset=dataset,
        modif=modif
    )

    print("Submitting jobs to the queue...")
    submit_jobs(jobs, jobdir)


def crawl_bucket(bucket, path, jobdir):
    """Gets subject list for a given s3 bucket and path

    Parameters
    ----------
    bucket : str
        s3 bucket
    path : str
        The directory where the dataset is stored on the S3 bucket
    jobdir : str
        Directory of batch jobs to generate/check up on

    Returns
    -------
    OrderedDict
        dictionary containing all subjects and sessions from the path location
    """
    from pynets.core.utils import flatten

    # if jobdir has seshs info file in it, use that instead
    sesh_path = f"{jobdir}/seshs.json"
    if os.path.isfile(sesh_path):
        print("seshs.json found -- loading bucket info from there")
        with open(sesh_path, "r") as f:
            seshs = json.load(f)
        print("Information obtained from s3.")
        return seshs

    # set up bucket crawl
    subj_pattern = r"(?<=sub-)(\w*)(?=/ses)"
    sesh_pattern = r"(?<=ses-)(\d*)"
    all_subfiles = get_matching_s3_objects(bucket, path + "/sub-")
    all_subfiles = [obj for obj in all_subfiles]
    all_subs = [re.findall(subj_pattern, obj) for obj in all_subfiles]
    subjs = list(set([i for i in flatten(all_subs)]))
    seshs = OrderedDict()

    # populate seshs
    for subj in subjs:
        prefix = f"{path}/sub-{subj}/"
        all_seshfiles = get_matching_s3_objects(bucket, prefix)
        all_seshfiles = [obj for obj in all_seshfiles]
        all_seshs = [re.findall(sesh_pattern, obj) for obj in all_seshfiles]
        sesh = list(set([i for i in flatten(all_seshs)]))

        if sesh != []:
            seshs[subj] = sesh
            print(f"{subj} added to seshs.")
        else:
            seshs[subj] = None
            print(f"{subj} not added (no sessions).")

    # print session IDs and create json cache
    print(
        (
            "Session IDs: "
            + ", ".join(
                [
                    subj + "-" + sesh if sesh is not None else subj
                    for subj in subjs
                    for sesh in seshs[subj]
                ]
            )
        )
    )
    with open(sesh_path, "w") as f:
        json.dump(seshs, f)
    print(f"{sesh_path} created.")
    print("Information obtained from s3.")
    return seshs


def create_json(
    bucket,
    path,
    threads,
    jobdir,
    credentials=None,
    dataset=None,
    modif=""
):
    """Creates the json files for each of the jobs

    Parameters
    ----------
    bucket : str
        The S3 bucket with the input dataset formatted according to the BIDS standard.
    path : str
        The directory where the dataset is stored on the S3 bucket
    threads : OrderedDict
        dictionary containing all subjects and sessions from the path location
    jobdir : str
        Directory of batch jobs to generate/check up on
    credentials : [type], optional
        AWS formatted csv of credentials, by default None
    dataset : [type], optional
        Name added to the generated json job files "pynets_<version>_<dataset>_sub-<sub>_ses-<ses>", by default None
    modif : str, optional
        Name of folder on s3 to push to. If empty, push to a folder with pynets's version number, by default ""

    Returns
    -------
    list
        list of job jsons
    """
    from pathlib import Path
    jobsjson = f"{jobdir}/jobs.json"
    if os.path.isfile(jobsjson):
        with open(jobsjson, "r") as f:
            jobs = json.load(f)
        return jobs

    # set up infrastructure
    out = subprocess.check_output(f"mkdir -p {jobdir}", shell=True)
    out = subprocess.check_output(f"mkdir -p {jobdir}/jobs/", shell=True)
    out = subprocess.check_output(f"mkdir -p {jobdir}/ids/", shell=True)
    seshs = threads

    templ = os.path.dirname(__file__)

    with open("%s%s" % (str(Path(__file__).parent.parent), '/cloud_config.json'), 'r') as inf:
        template = json.load(inf)

    cmd = template["containerOverrides"]["command"]
    env = template["containerOverrides"]["environment"]

    # modify template
    if credentials is not None:
        env[0]["value"], env[1]["value"] = get_credentials()
    else:
        env = []
    template["containerOverrides"]["environment"] = env

    # edit non-defaults
    jobs = []
    cmd[cmd.index("<INPUT>")] = f's3://{bucket}/{path}'
    cmd[cmd.index("<PUSH>")] = f's3://{bucket}/{path}/{modif}'

    # edit participant-specific values ()
    # loop over every session of every participant
    for subj in seshs.keys():
        print(f"... Generating job for sub-{subj}")
        # and for each subject number in each participant number,
        for sesh in seshs[subj]:
            # add format-specific commands,
            job_cmd = deepcopy(cmd)
            job_cmd[job_cmd.index("<SUBJ>")] = subj
            if sesh is not None:
                job_cmd[job_cmd.index("<SESH>")] = sesh

            # then, grab the template,
            # add additional parameters,
            # make the json file for this iteration,
            # and add the path to its json file to `jobs`.
            job_json = deepcopy(template)
            ver = pynets.__version__.replace(".", "-")
            if dataset:
                name = f"pynets_{ver}_{dataset}_sub-{subj}"
            else:
                name = f"pynets_{ver}_sub-{subj}"
            if sesh is not None:
                name = f"{name}_ses-{sesh}"
            print(job_cmd)
            job_json["jobName"] = name
            job_json["containerOverrides"]["command"] = job_cmd
            job = os.path.join(jobdir, "jobs", name + ".json")
            with open(job, "w") as outfile:
                json.dump(job_json, outfile)
            jobs += [job]

    # return list of job jsons
    with open(jobsjson, "w") as f:
        json.dump(jobs, f)
    return jobs


def submit_jobs(jobs, jobdir):
    """Give list of jobs to submit, submits them to AWS Batch

    Parameters
    ----------
    jobs : list
        Name of the json files for all jobs to submit
    jobdir : str
        Directory of batch jobs to generate/check up on

    Returns
    -------
    int
        0
    """

    batch = s3_client(service="batch")

    for job in jobs:
        with open(job, "r") as f:
            kwargs = json.load(f)
        print(f"... Submitting job {job}...")
        submission = batch.submit_job(**kwargs)
        print((f'Job Name: {submission["jobName"]}, Job ID: {submission["jobId"]}'))
        sub_file = os.path.join(jobdir, "ids", submission["jobName"] + ".json")
        with open(sub_file, "w") as outfile:
            json.dump(submission, outfile)
        print("Submitted.")
    return 0


def kill_jobs(jobdir, reason='"Killing job"'):
    """Given a list of jobs, kills them all

    Parameters
    ----------
    jobdir : str
        Directory of batch jobs to generate/check up on
    reason : str, optional
        Task you want to perform on the jobs, by default '"Killing job"'
    """

    print(f"Cancelling/Terminating jobs in {jobdir}/ids/...")
    jobs = os.listdir(jobdir + "/ids/")
    batch = s3_client(service="batch")
    jids = []
    names = []

    # grab info about all the jobs
    for job in jobs:
        with open(f"{jobdir}/ids/{job}", "r") as inf:
            submission = json.load(inf)
        jid = submission["jobId"]
        name = submission["jobName"]
        jids.append(jid)
        names.append(name)

    for jid in jids:
        print(f"Terminating job {jid}")
        batch.terminate_job(jobId=jid, reason=reason)


def main():
    parser = ArgumentParser(description='PyNets AWS Cloud CLI: A Fully-Automated Workflow for Reproducible '
                                        'Ensemble Sampling of Functional and Structural Connectomes')

    parser.add_argument(
        "--bucket",
        help="""The S3 bucket with the input dataset
         formatted according to the BIDS standard.""",
    )
    parser.add_argument(
        "--bidsdir",
        help="""The path of the directory where the dataset
        lives on the S3 bucket.""",
    )
    parser.add_argument(
        "--jobdir",
        action="store",
        help="""Local directory where the generated batch jobs will be
        saved/run through in case of batch termination or check-up."""
    )
    parser.add_argument(
        "--credentials",
        action="store",
        help="csv formatted AWS credentials."
    )
    parser.add_argument("--dataset", action="store", help="Dataset name")
    parser.add_argument(
        "--modif",
        action="store",
        help="""Name of folder on s3 to push to. Data will be saved at '<bucket>/<bidsdir>/<modif>' on the s3 bucket
        If empty, push to a folder with pynets's version number.""",
        default="",
    )

    result = parser.parse_args()

    bucket = result.bucket
    path = result.bidsdir
    path = path.strip("/") if path is not None else path
    dset = path.split("/")[-1] if path is not None else None
    creds = result.credentials
    jobdir = result.jobdir
    modif = result.modif

    if jobdir is None:
        jobdir = "./"

    print("Beginning batch submission process...")
    if not os.path.exists(jobdir):
        print("job directory not found. Creating...")
        Path(jobdir).mkdir(parents=True)
    batch_submit(
        bucket,
        path,
        jobdir,
        credentials=creds,
        dataset=dset,
        modif=modif
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
