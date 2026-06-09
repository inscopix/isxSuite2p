"""
Copyright © 2024 Inscopix, Inc., a Bruker company. Authored by Ludovic Bellier.
"""
# import os
import numpy as np
from . import utils

try:
    import isx
    HAS_ISX = True
except (ModuleNotFoundError, ImportError):
    HAS_ISX = False

import logging 
logger = logging.getLogger(__name__)


def isxd_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """  finds Inscopix isxd files and writes them to binaries

    Parameters
    ----------
    ops : dictionary
        "nplanes", "data_path", "save_path", "save_folder", "fast_disk",
        "nchannels", "keep_movie_raw", "look_one_level_down"

    Returns
    -------
        ops : dictionary of first plane
            "Ly", "Lx", ops["reg_file"] or ops["raw_file"] is created binary

    """
    if not HAS_ISX:
        raise ImportError("Inscopix isx is required for this file type, please 'pip install isx'")

    # dbs = utils.init_dbs(dbs)
    # the following should be taken from the metadata and not needed but the files are initialized before...
    nplanes = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]
    ncp = nplanes * nchannels
    nfunc = dbs[0]["functional_chan"] - 1 if nchannels > 1 else 0

    if nplanes > 1 and nchannels > 1:
        raise RuntimeError("ISXD files only support multi-plane or multi-channel data, but not both. " 
                           "Please review input ops and ensure either nplanes >= 1 OR nchannels >= 1.")

    # open all binary files for writing
    # ops1, file_list, reg_file, reg_file_chan2 = utils.find_files_open_binaries(ops1)
    file_list = dbs[0]["file_list"]
    iall = 0
    for j in range(dbs[0]["nplanes"]):
        dbs[j]["nframes_per_folder"] = np.zeros(len(file_list), np.int32)
    ik = 0

    for ifile, fname in enumerate(file_list):
        f = isx.Movie.read(fname)
        nframes = f.timing.num_samples
        iblocks = np.arange(0, nframes, dbs[0]["batch_size"])
        if iblocks[-1] < nframes:
            iblocks = np.append(iblocks, nframes)

        # data = nframes x width x height x (nplanes OR nchannels)
        # loop over all frames
        for ichunk, onset in enumerate(iblocks[:-1]):
            offset = iblocks[ichunk + 1]
            im = np.array([f.get_frame_data(x) for x in np.arange(onset, offset)])
            im2mean = im.mean(axis=0).astype(np.float32) / len(iblocks)

            if ik == 0:
                for j in range(nplanes):
                    dbs[j]["meanImg"] = np.zeros((im.shape[1], im.shape[2]),
                                                    np.float32)
                    if nchannels > 1:
                        dbs[j]["meanImg_chan2"] = np.zeros(
                            (im.shape[1], im.shape[2]), np.float32)
                    dbs[j]["nframes"] = 0

            if nchannels > 1:
                for ichan in range(nchannels):
                    nframes = im.shape[0]
                    i0 = (ichan) % nplanes
                    im2write =  im[np.arange(int(i0), nframes, nchannels), :, :]

                    if ichan == nfunc:
                        dbs[0]["meanImg"] += np.squeeze(im2mean)
                        reg_file[j].write(
                            bytearray(im2write[:].astype("int16")))
                    else:
                        dbs[0]["meanImg_chan2"] += np.squeeze(im2mean)
                        reg_file_chan2[j].write(
                            bytearray(im2write[:].astype("int16")))

                    dbs[0]["nframes"] += im2write.shape[0]
                    dbs[0]["nframes_per_folder"][ifile] += im2write.shape[0]
            else:
                nframes = im.shape[0]
                for j in range(0, nplanes):
                    i0 = (j) % nplanes
                    im2write =  im[np.arange(int(i0), nframes, nplanes), :, :]

                    dbs[j]["meanImg"] += np.squeeze(im2mean)
                    reg_file[j].write(
                        bytearray(im2write[:].astype("int16")))

                    dbs[j]["nframes"] += im2write.shape[0]
                    dbs[j]["nframes_per_folder"][ifile] += im2write.shape[0]

            ik += nframes

    # write ops files
    do_registration = settings["run"]["do_registration"]
    # do_nonrigid = ops1[0]["nonrigid"]
    for db in dbs:
        db["Ly"] = im.shape[1]
        db["Lx"] = im.shape[2]
        if not do_registration:
            db["yrange"] = np.array([0, db["Ly"]])
            db["xrange"] = np.array([0, db["Lx"]])
        #ops["meanImg"] /= ops["nframes"]
        #if nchannels>1:
        #    ops["meanImg_chan2"] /= ops["nframes"]
        # Save db and settings to each plane folder
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)
    # close all binary files and write ops files
    for j in range(0, nplanes):
        reg_file[j].close()
        if nchannels > 1:
            reg_file_chan2[j].close()
    return dbs
