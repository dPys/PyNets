# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:40:07 2017
Copyright (C) 2018
@author: Derek Pisner
"""
import os
import shutil
import numpy as np
import nibabel as nib
import warnings
warnings.filterwarnings("ignore")


try:
    FSLDIR = os.environ['FSLDIR']
except KeyError:
    print('FSLDIR environment variable not set!')


def segment_t1w(t1w, basename, opts=''):
    """
    A function to use FSL's FAST to segment an anatomical
    image into GM, WM, and CSF prob maps.

    Parameters
    ----------
    t1w : str
        File path to an anatomical T1-weighted image.
    basename : str
        A basename to use for output files.
    opts : str
        Additional options that can optionally be passed to fast.
        Desirable options might be -P, which will use prior probability maps if the input T1w MRI is in standard space.

    Returns
    -------
    out : str
        File path to the probability map Nifti1Image consisting of GM, WM, and CSF in the 4th dimension.
    """
    print("Segmenting Anatomical Image into WM, GM, and CSF...")
    # run FAST, with options -t for the image type and -n to
    # segment into CSF (pve_0), GM (pve_1), WM (pve_2)
    cmd = "fast -t 1 {} -n 3 -o {} {}".format(opts, basename, t1w)
    os.system(cmd)
    out = {}  # the outputs
    out['wm_prob'] = "{}_{}".format(basename, "pve_2.nii.gz")
    out['gm_prob'] = "{}_{}".format(basename, "pve_1.nii.gz")
    out['csf_prob'] = "{}_{}".format(basename, "pve_0.nii.gz")
    return out


def align(inp, ref, xfm=None, out=None, dof=12, searchrad=True, bins=256, interp=None, cost="mutualinfo", sch=None,
          wmseg=None, init=None):
    """
    Aligns two images using linear registration (FSL's FLIRT).

    Parameters
    ----------
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        ref : str
            File path to reference Nifti1Image to use as the target for alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        out : str
            File path to input Nifti1Image output following registration alignment.
        dof : int
            Number of degrees of freedom to use in the alignment.
        searchrad : bool
            Indicating whether to use the predefined searchradius parameter (180 degree sweep in x, y, and z).
            Default is True.
        bins : int
            Number of histogram bins. Default is 256.
        interp : str
            Interpolation method to use. Default is mutualinfo.
        sch : str
            Optional file path to a FLIRT schedule file.
        wmseg : str
            Optional file path to white-matter segmentation Nifti1Image for boundary-based registration (BBR).
        init : str
            File path to a transformation matrix in .xfm format to use as an initial guess for the alignment.
    """
    cmd = "flirt -in {} -ref {}".format(inp, ref)
    if xfm is not None:
        cmd += " -omat {}".format(xfm)
    if out is not None:
        cmd += " -out {}".format(out)
    if dof is not None:
        cmd += " -dof {}".format(dof)
    if bins is not None:
        cmd += " -bins {}".format(bins)
    if interp is not None:
        cmd += " -interp {}".format(interp)
    if cost is not None:
        cmd += " -cost {}".format(cost)
    if searchrad is not None:
        cmd += " -searchrx -180 180 -searchry -180 180 -searchrz -180 180"
    if sch is not None:
        cmd += " -schedule {}".format(sch)
    if wmseg is not None:
        cmd += " -wmseg {}".format(wmseg)
    if init is not None:
        cmd += " -init {}".format(init)
    print(cmd)
    os.system(cmd)
    return


def align_nonlinear(inp, ref, xfm, out, warp, ref_mask=None, in_mask=None, config=None):
    """
    Aligns two images using nonlinear registration and stores the transform between them.

    Parameters
    ----------
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        ref : str
            File path to reference Nifti1Image to use as the target for alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        out : str
            File path to input Nifti1Image output following registration alignment.
        warp : str
            File path to input Nifti1Image output for the nonlinear warp following alignment.
        ref_mask : str
            Optional file path to a mask in reference image space.
        in_mask : str
            Optional file path to a mask in input image space.
        config : str
            Optional file path to config file specifying command line arguments.
    """
    cmd = "fnirt --in={} --ref={} --aff={} --iout={} --cout={} --warpres=8,8,8"
    cmd = cmd.format(inp, ref, xfm, out, warp, config)
    if ref_mask is not None:
        cmd += " --refmask={}".format(ref_mask)
    if in_mask is not None:
        cmd += " --inmask={}".format(in_mask)
    if config is not None:
        cmd += " --config={}".format(config)
    print(cmd)
    os.system(cmd)
    return


def applyxfm(ref, inp, xfm, aligned, interp='trilinear', dof=6):
    """
    Aligns two images with a given transform.

    Parameters
    ----------
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        ref : str
            File path to reference Nifti1Image to use as the target for alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        aligned : str
            File path to input Nifti1Image output following registration alignment.
        interp : str
            Interpolation method to use. Default is trilinear.
        dof : int
            Number of degrees of freedom to use in the alignment.
    """
    cmd = "flirt -in {} -ref {} -out {} -init {} -interp {} -dof {} -applyxfm".format(inp, ref, aligned, xfm, interp,
                                                                                      dof)
    print(cmd)
    os.system(cmd)
    return


def apply_warp(ref, inp, out, warp, xfm=None, mask=None, interp=None, sup=False):
    """
    Applies a warp to a Nifti1Image which transforms the image to the reference space used in generating the warp.

    Parameters
    ----------
        ref : str
            File path to reference Nifti1Image to use as the target for alignment.
        inp : str
            File path to input Nifti1Image to be aligned for registration.
        out : str
            File path to input Nifti1Image output following registration alignment.
        warp : str
            File path to input Nifti1Image output for the nonlinear warp following alignment.
        xfm : str
            File path for the transformation matrix output in .xfm.
        mask : str
            Optional file path to a mask in reference image space.
        interp : str
            Interpolation method to use.
        sup : bool
            Intermediary supersampling of output. Default is False.
    """
    cmd = "applywarp --ref={} --in={} --out={} --warp={}".format(ref, inp, out, warp)
    if xfm is not None:
        cmd += " --premat={}".format(xfm)
    if mask is not None:
        cmd += " --mask={}".format(mask)
    if interp is not None:
        cmd += " --interp={}".format(interp)
    if sup is True:
        cmd += " --super --superlevel=a"
    print(cmd)
    os.system(cmd)
    return


def inverse_warp(ref, out, warp):
    """
    Generates the inverse of a warp from a reference image space to the input image space.
    space used in generating the warp.

    Parameters
    ----------
        ref : str
            File path to reference Nifti1Image to use as the target for alignment.
        out : str
            File path to input Nifti1Image output following registration alignment.
        warp : str
            File path to input Nifti1Image output for the nonlinear warp following alignment.
    """
    cmd = "invwarp --warp=" + warp + " --out=" + out + " --ref=" + ref
    print(cmd)
    os.system(cmd)
    return


def combine_xfms(xfm1, xfm2, xfmout):
    """
    A function to combine two transformations, and output the resulting transformation.

    Parameters
    ----------
        xfm1 : str
            File path to the first transformation.
        xfm2 : str
            File path to the second transformation.
        xfmout : str
            File path to the output transformation.
    """
    cmd = "convert_xfm -omat {} -concat {} {}".format(xfmout, xfm1, xfm2)
    print(cmd)
    os.system(cmd)
    return


def antssyn(template_path, fa_path, working_dir):
    """
    A function to perform ANTS SyN registration

    Parameters
    ----------
        template_path  : str
            File path to the template reference image.
        fa_path : str
            File path to the FA moving image.
        working_dir : str
            Path to the working directory to perform SyN and save outputs.
    """
    try:
        ants_path = os.environ['ANTSPATH']
    except KeyError:
        print('ANTSDIR environment variable not set!')

    cmd = "{}/antsRegistrationSyN.sh -d 3 -f {} -m {} -o {}/".format(ants_path, template_path, fa_path, working_dir)
    print(cmd)
    os.system(cmd)
    return


def antsapplytranstopoints(warped_csv_out, aattp_out, transforms):
    """
    A function to perform ANTS SyN registration

    Parameters
    ----------
        warped_csv_out : str
            File path to a csv file with columns including x,y,z,t (all 4) column headers.
        aattp_out : str
            File path to a csv file containing the output warped points.
        transforms : str
            Path to the transforms generated from ANTS SyN.
    """
    try:
        ants_path = os.environ['ANTSPATH']
    except KeyError:
        print('ANTSDIR environment variable not set!')

    cmd = "{}/antsApplyTransformsToPoints -d 3 -i {} -o {} {}".format(ants_path, warped_csv_out, aattp_out, transforms)
    print(cmd)
    os.system(cmd)
    return


def transform_to_affine(streams, header, affine):
    """
    A function to transform .trk file of tractography streamlines to a given affine.

    Parameters
    ----------
    streams : str
        File path to input streamline array sequence in .trk format.
    header : Dict
        Nibabel trackvis header object to use for transformed streamlines file.
    affine : array
        4 x 4 2D numpy array representing the target affine for streamline transformation.

    Returns
    -------
    streams : str
        File path to transformed streamline array sequence in .trk format.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from dipy.tracking.utils import move_streamlines
    rotation, scale = np.linalg.qr(affine)
    streams = move_streamlines(streams, rotation)
    scale[0:3, 0:3] = np.dot(scale[0:3, 0:3], np.diag(1. / header['voxel_sizes']))
    scale[0:3, 3] = abs(scale[0:3, 3])
    streams = move_streamlines(streams, scale)
    return streams


def check_orient_and_dims(infile, vox_size, bvecs=None):
    """
    An API to reorient any image to RAS+ and resample any image to a given voxel resolution.

    Parameters
    ----------
    infile : str
        File path to a dwi Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    bvecs : str
        File path to corresponding bvecs file if infile is a dwi.

    Returns
    -------
    outfile : str
        File path to the reoriented and/or resample Nifti1Image.
    bvecs : str
        File path to corresponding reoriented bvecs file if outfile is a dwi.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import os
    import os.path as op
    from pynets.registration.reg_utils import reorient_dwi, reorient_img, match_target_vox_res

    outdir = op.dirname(infile)
    img = nib.load(infile)
    vols = img.shape[-1]

    reoriented = "%s%s%s%s" % (outdir, '/', infile.split('/')[-1].split('.nii.gz')[0], '_pre_reor.nii.gz')
    resampled = "%s%s%s%s" % (outdir, '/', os.path.basename(infile).split('.nii.gz')[0], '_pre_res.nii.gz')

    # Check orientation
    if (vols > 1) and (bvecs is not None):
        # dwi case
        outdir = "%s%s" % (outdir, '/std_dmri')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Check orientation
        if not os.path.isfile(reoriented):
            [infile, bvecs] = reorient_dwi(infile, bvecs, outdir)
        # Check dimensions
        if not os.path.isfile(resampled):
            outfile = match_target_vox_res(infile, vox_size, outdir, sens='dwi')
    elif (vols > 1) and (bvecs is None):
        # func case
        outdir = "%s%s" % (outdir, '/std_fmri')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Check orientation
        if not os.path.isfile(reoriented):
            infile = reorient_img(infile, outdir)
        # Check dimensions
        if not os.path.isfile(resampled):
            outfile = match_target_vox_res(infile, vox_size, outdir, sens='func')
    else:
        # t1w case
        outdir = "%s%s" % (outdir, '/std_anat_')
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        # Check orientation
        if not os.path.isfile(reoriented):
            infile = reorient_img(infile, outdir)
        if not os.path.isfile(resampled):
            # Check dimensions
            outfile = match_target_vox_res(infile, vox_size, outdir, sens='t1w')

    print(outfile)

    if bvecs is None:
        return outfile
    else:
        return outfile, bvecs


def reorient_dwi(dwi_prep, bvecs, out_dir):
    """
    A function to reorient any dwi image and associated bvecs to RAS+.

    Parameters
    ----------
    dwi_prep : str
        File path to a dwi Nifti1Image.
    bvecs : str
        File path to corresponding bvecs file.
    out_dir : str
        Path to output directory.

    Returns
    -------
    dwi_prep : str
        File path to the reoriented dwi Nifti1Image.
    bvecs : str
        File path to corresponding reoriented bvecs file.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import shutil
    # Check orientation (dwi_prep)
    cmd_run = os.popen('fslorient -getorient {}'.format(dwi_prep))
    orient = cmd_run.read().strip('\n')
    cmd_run.close()
    dwi_orig = dwi_prep
    dwi_prep = "%s%s%s%s" % (out_dir, '/', dwi_prep.split('/')[-1].split('.nii.gz')[0], '_pre_reor.nii.gz')
    shutil.copyfile(dwi_orig, dwi_prep)
    bvecs_orig = bvecs
    bvecs = "%s%s" % (out_dir, '/bvecs.bvec')
    shutil.copyfile(bvecs_orig, bvecs)
    bvecs_mat = np.genfromtxt(bvecs)
    cmd_run = os.popen('fslorient -getqform {}'.format(dwi_prep))
    qform = cmd_run.read().strip('\n')
    cmd_run.close()
    reoriented = False
    if orient == 'NEUROLOGICAL':
        reoriented = True
        print('Neurological (dwi), reorienting to radiological...')
        # Orient dwi to RADIOLOGICAL
        # Posterior-Anterior Reorientation
        if float(qform.split(' ')[:-1][5]) <= 0:
            dwi_prep_PA = "%s%s" % (out_dir, '/dwi_reor_PA.nii.gz')
            print('Reorienting P-A flip (dwi)...')
            os.system('fslswapdim {} -x -y z {}'.format(dwi_prep, dwi_prep_PA))
            bvecs_mat[:, 1] = -bvecs_mat[:, 1]
            cmd_run = os.popen('fslorient -getqform {}'.format(dwi_prep_PA))
            qform = cmd_run.read().strip('\n')
            cmd_run.close()
            dwi_prep = dwi_prep_PA
        # Inferior-Superior Reorientation
        if float(qform.split(' ')[:-1][10]) <= 0:
            dwi_prep_IS = "%s%s" % (out_dir, '/dwi_reor_IS.nii.gz')
            print('Reorienting I-S flip (dwi)...')
            os.system('fslswapdim {} -x y -z {}'.format(dwi_prep, dwi_prep_IS))
            bvecs_mat[:, 2] = -bvecs_mat[:, 2]
            dwi_prep = dwi_prep_IS
        bvecs_mat[:, 0] = -bvecs_mat[:, 0]
        os.system('fslorient -forceradiological {}'.format(dwi_prep))
        np.savetxt(bvecs, bvecs_mat)
    else:
        print('Radiological (dwi)...')
        # Posterior-Anterior Reorientation
        if float(qform.split(' ')[:-1][5]) <= 0:
            dwi_prep_PA = "%s%s" % (out_dir, '/dwi_reor_PA.nii.gz')
            print('Reorienting P-A flip (dwi)...')
            os.system('fslswapdim {} -x -y z {}'.format(dwi_prep, dwi_prep_PA))
            bvecs_mat[:, 1] = -bvecs_mat[:, 1]
            cmd_run = os.popen('fslorient -getqform {}'.format(dwi_prep_PA))
            qform = cmd_run.read().strip('\n')
            cmd_run.close()
            dwi_prep = dwi_prep_PA
            reoriented = True
        # Inferior-Superior Reorientation
        if float(qform.split(' ')[:-1][10]) <= 0:
            dwi_prep_IS = "%s%s" % (out_dir, '/dwi_reor_IS.nii.gz')
            print('Reorienting I-S flip (dwi)...')
            os.system('fslswapdim {} -x y -z {}'.format(dwi_prep, dwi_prep_IS))
            bvecs_mat[:, 2] = -bvecs_mat[:, 2]
            dwi_prep = dwi_prep_IS
            reoriented = True
        np.savetxt(bvecs, bvecs_mat)

    if reoriented is True:
        imgg = nib.load(dwi_prep)
        data = imgg.get_fdata()
        affine = imgg.affine
        hdr = imgg.header
        imgg = nib.Nifti1Image(data, affine=affine, header=hdr)
        imgg.set_sform(affine)
        imgg.set_qform(affine)
        imgg.update_header()
        nib.save(imgg, dwi_prep)

        print('Reoriented affine: ')
        print(affine)
    else:
        dwi_prep = dwi_orig
        print('Image already in RAS+')

    return dwi_prep, bvecs


def reorient_img(img, out_dir):
    """
    A function to reorient any non-dwi image to RAS+.

    Parameters
    ----------
    img : str
        File path to a Nifti1Image.
    out_dir : str
        Path to output directory.

    Returns
    -------
    img : str
        File path to reoriented Nifti1Image.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import shutil
    import random
    cmd_run = os.popen('fslorient -getorient {}'.format(img))
    orient = cmd_run.read().strip('\n')
    cmd_run.close()
    img_orig = img
    hash = str(random.randint(1, 10000))
    img = "%s%s%s%s%s%s" % (out_dir, '/', img.split('/')[-1].split('.nii.gz')[0], '_pre_reor_', hash, '.nii.gz')
    shutil.copyfile(img_orig, img)
    cmd_run = os.popen('fslorient -getqform {}'.format(img))
    qform = cmd_run.read().strip('\n')
    cmd_run.close()
    reoriented = False
    if orient == 'NEUROLOGICAL':
        reoriented = True
        print('Neurological (img), reorienting to radiological...')
        # Orient img to std
        # Posterior-Anterior Reorientation
        if float(qform.split(' ')[:-1][5]) <= 0:
            img_PA = "%s%s%s" % (out_dir, '/', 'img_reor_PA.nii.gz')
            print('Reorienting P-A flip (img)...')
            os.system('fslswapdim {} -x -y z {}'.format(img, img_PA))
            cmd_run = os.popen('fslorient -getqform {}'.format(img_PA))
            qform = cmd_run.read().strip('\n')
            cmd_run.close()
            img = img_PA
        # Inferior-Superior Reorientation
        if float(qform.split(' ')[:-1][10]) <= 0:
            img_IS = "%s%s%s" % (out_dir, '/', 'img_reor_IS.nii.gz')
            print('Reorienting I-S flip (img)...')
            os.system('fslswapdim {} -x y -z {}'.format(img, img_IS))
            img = img_IS
        os.system('fslorient -forceradiological {}'.format(img))
    else:
        print('Radiological (img)...')
        # Posterior-Anterior Reorientation
        if float(qform.split(' ')[:-1][5]) <= 0:
            img_PA = "%s%s%s" % (out_dir, '/', 'img_reor_PA.nii.gz')
            print('Reorienting P-A flip (img)...')
            os.system('fslswapdim {} -x -y z {}'.format(img, img_PA))
            cmd_run = os.popen('fslorient -getqform {}'.format(img_PA))
            qform = cmd_run.read().strip('\n')
            cmd_run.close()
            img = img_PA
            reoriented = True
        # Inferior-Superior Reorientation
        if float(qform.split(' ')[:-1][10]) <= 0:
            img_IS = "%s%s%s" % (out_dir, '/', 'img_reor_IS.nii.gz')
            print('Reorienting I-S flip (img)...')
            os.system('fslswapdim {} -x y -z {}'.format(img, img_IS))
            img = img_IS
            reoriented = True

    if reoriented is True:
        imgg = nib.load(img)
        data = imgg.get_fdata()
        affine = imgg.affine
        hdr = imgg.header
        imgg = nib.Nifti1Image(data, affine=affine, header=hdr)
        imgg.set_sform(affine)
        imgg.set_qform(affine)
        imgg.update_header()
        nib.save(imgg, img)

        print('Reoriented affine: ')
        print(affine)
    else:
        img = img_orig
        print('Image already in RAS+')

    return img


def match_target_vox_res(img_file, vox_size, out_dir, sens):
    """
    A function to resample an image to a given isotropic voxel resolution.

    Parameters
    ----------
    img_file : str
        File path to a Nifti1Image.
    vox_size : str
        Voxel size in mm. (e.g. 2mm).
    out_dir : str
        Path to output directory.
    sens : str
        Modality of Nifti1Image input (e.g. 'dwi').

    Returns
    -------
    img_file : str
        File path to resampled Nifti1Image.
    """
    import warnings
    warnings.filterwarnings("ignore")
    from dipy.align.reslice import reslice
    # Check dimensions
    img = nib.load(img_file)
    data = img.get_fdata()
    affine = img.affine
    hdr = img.header
    zooms = hdr.get_zooms()[:3]
    if vox_size == '1mm':
        new_zooms = (1., 1., 1.)
    elif vox_size == '2mm':
        new_zooms = (2., 2., 2.)

    if (abs(zooms[0]), abs(zooms[1]), abs(zooms[2])) != new_zooms:
        print('Reslicing image ' + img_file + ' to ' + vox_size + '...')
        img_file_pre = "%s%s%s%s" % (out_dir, '/', os.path.basename(img_file).split('.nii.gz')[0], '_pre_res.nii.gz')
        shutil.copyfile(img_file, img_file_pre)
        data2, affine2 = reslice(data, affine, zooms, new_zooms)
        if sens == 'dwi':
            affine2[0:3, 3] = np.zeros(3)
            affine2[0:3, 0:3] = np.eye(3) * np.array(new_zooms) * np.sign(affine2[0:3, 0:3])
        img2 = nib.Nifti1Image(data2, affine=affine2, header=hdr)
        img2.set_qform(affine2)
        img2.set_sform(affine2)
        img2.update_header()
        nib.save(img2, img_file)
        print('Resliced affine: ')
        print(nib.load(img_file).affine)
    else:
        if sens == 'dwi':
            affine[0:3, 3] = np.zeros(3)
        img = nib.Nifti1Image(data, affine=affine, header=hdr)
        img.set_sform(affine)
        img.set_qform(affine)
        img.update_header()
        nib.save(img, img_file)

    return img_file
