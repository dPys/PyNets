#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2017
@author: Derek Pisner (dPys)
"""
from configparser import ConfigParser
import os
import sys
import boto3


def get_credentials():
    """Searches for and returns AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY

    Returns
    -------
    tuple
        Two strings inside of a tuple, (Access_key, Secret_access_key)

    Raises
    ------
    AttributeError
        No AWS credentials are found
    """
    # add option to pass profile name
    try:
        config = ConfigParser()
        config.read(os.getenv("HOME") + "/.aws/credentials")
        return (
            config.get("default", "aws_access_key_id"),
            config.get("default", "aws_secret_access_key"),
        )
    except:
        ACCESS = os.getenv("AWS_ACCESS_KEY_ID")
        SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not ACCESS and SECRET:
        raise AttributeError("No AWS credentials found.")
    return ACCESS, SECRET


def s3_client(service="s3"):
    """
    create an s3 client.

    Parameters
    ----------
    service : str
        Type of service.

    Returns
    -------
    boto3.client
        client with proper credentials.
    """

    try:
        ACCESS, SECRET = get_credentials()
    except AttributeError:
        return boto3.client(service)
    return boto3.client(service, aws_access_key_id=ACCESS, aws_secret_access_key=SECRET)


def parse_path(s3_datapath):
    """
    Return bucket and prefix from full s3 path.

    Parameters
    ----------
    s3_datapath : str
        path to a bucket.
        Should be of the form s3://bucket/prefix/.

    Returns
    -------
    tuple
        bucket and prefix.
    """
    bucket_path = str(s3_datapath).split("//")[1]
    parts = bucket_path.split("/")
    bucket = parts[0].strip("/")
    prefix = "/".join(parts[1:])
    return bucket, prefix


def get_matching_s3_objects(bucket, prefix="", suffix=""):
    """
    Generate objects in an S3 bucket.

    Parameters
    ----------
    bucket : str
        Name of the s3 bucket.
    prefix : str, optional
        Only fetch objects whose key starts with this prefix, by default ''
    suffix : str, optional
        Only fetch objects whose keys end with this suffix, by default ''
    """
    s3 = s3_client(service="s3")
    kwargs = {"Bucket": bucket}

    # If the prefix is a single string (not a tuple of strings), we can
    # do the filtering directly in the S3 API.
    if isinstance(prefix, str):
        kwargs["Prefix"] = prefix

    while True:

        # The S3 API response is a large blob of metadata.
        # 'Contents' contains information about the listed objects.
        resp = s3.list_objects_v2(**kwargs)

        try:
            contents = resp["Contents"]
        except KeyError:
            print("No contents found.")
            return

        for obj in contents:
            key = obj["Key"]
            if key.startswith(prefix) and key.endswith(suffix):
                yield key

        # The S3 API is paginated, returning up to 1000 keys at a time.
        # Pass the continuation token into the next response, until we
        # reach the final page (when this field is missing).
        try:
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        except KeyError:
            break


def s3_get_data(bucket, remote, local, info="", force=False):
    """Given and s3 directory, copies files/subdirectories in that directory to local

    Parameters
    ----------
    bucket : str
        s3 bucket you are accessing data from
    remote : str
        The path to the data on your S3 bucket. The data will be
        downloaded to the provided bids_dir on your machine.
    local : list
        Local input directory where you want the files copied to and subject/session info [input, sub-#/ses-#]
    info : str, optional
        Relevant subject and session information in the form of sub-#/ses-#
    force : bool, optional
        Whether to overwrite the local directory containing the s3 files if it already exists, by default False
    """

    if info == "sub-":
        print("Subject not specified, comparing input folder to remote directory...")
    else:
        if os.path.exists(os.path.join(local, info)) and not force:
            if os.listdir(os.path.join(local, info)):
                print(
                    f"Local directory: {os.path.join(local, info)} already exists. Not pulling s3 data. Delete "
                    f"contents to re-download data."
                )
                return

    # get client with credentials if they exist
    client = s3_client(service="s3")

    # check that bucket exists
    bkts = [bk["Name"] for bk in client.list_buckets()["Buckets"]]
    if bucket not in bkts:
        raise ValueError(
            "Error: could not locate bucket. Available buckets: " + ", ".join(bkts)
        )

    bpath = get_matching_s3_objects(bucket, f"{remote}/{info}")

    # go through all folders inside of remote directory and download relevant files
    for obj in bpath:
        bdir, data = os.path.split(obj)
        localpath = os.path.join(local, bdir.replace(f"{remote}/", ""))
        # Make directory for data if it doesn't exist
        if not os.path.exists(localpath):
            os.makedirs(localpath)
        if not os.path.exists(f"{localpath}/{data}"):
            print(f"Downloading {bdir}/{data} from {bucket} s3 bucket...")
            # Download file
            client.download_file(bucket, f"{bdir}/{data}", f"{localpath}/{data}")
            if os.path.exists(f"{localpath}/{data}"):
                print("Success!")
            else:
                print("Error: File not downloaded")
        else:
            print(f"File {data} already exists at {localpath}/{data}")


def s3_push_data(bucket, remote, outDir, subject=None, session=None, creds=True):
    """Pushes data to a specified S3 bucket

    Parameters
    ----------
    bucket : str
        s3 bucket you are pushing files to
    remote : str
        The path to the directory on your S3 bucket containing the data used in the pipeline, the string in 'modifier'
        will be put after the first directory specified in the path as its own directory
        (/remote[0]/modifier/remote[1]/...)
    outDir : str
        Path of local directory being pushed to the s3 bucket
    subject : str
        subject we're pushing with
    session : str
        session we're pushing with
    creds : bool, optional
        Whether s3 credentials are being provided, may fail to push big files if False, by default True
    """

    # get client with credentials if they exist
    client = s3_client(service="s3")

    # check that bucket exists
    bkts = [bk["Name"] for bk in client.list_buckets()["Buckets"]]
    if bucket not in bkts:
        sys.exit(
            "Error: could not locate bucket. Available buckets: " + ", ".join(bkts)
        )

    # List all files and upload
    for root, _, files in os.walk(outDir):
        for file_ in files:
            if not "tmp/" in root:  # exclude things in the tmp/ folder
                if f"sub-{subject}/ses-{session}" in root:
                    print(f"Uploading: {os.path.join(root, file_)}")
                    spath = root[root.find("sub-"):]  # remove everything before /sub-*
                    client.upload_file(
                        os.path.join(root, file_),
                        bucket,
                        f"{remote}/{os.path.join(spath, file_)}",
                        ExtraArgs={"ACL": "public-read"},
                    )
