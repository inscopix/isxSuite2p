"""
Copyright © 2024 Inscopix, Inc., a Bruker company. Authored by Ludovic Bellier.
"""
import numpy as np

try:
    import isx
    HAS_ISX = True
except (ModuleNotFoundError, ImportError):
    HAS_ISX = False


def _init_plane_accumulators(dbs, nplanes, nchannels, file_count, ly, lx):
    for j in range(nplanes):
        dbs[j]["nframes_per_folder"] = np.zeros(file_count, np.int32)
        dbs[j]["meanImg"] = np.zeros((ly, lx), np.float32)
        if nchannels > 1:
            dbs[j]["meanImg_chan2"] = np.zeros((ly, lx), np.float32)
        dbs[j]["nframes"] = 0


def _write_and_accumulate(db, frame_block, out_file, mean_key, ifile, count_frames):
    if frame_block.shape[0] == 0:
        return
    out_file.write(bytearray(frame_block.astype("int16")))
    db[mean_key] += frame_block.astype(np.float32).sum(axis=0)
    if count_frames:
        db["nframes"] += frame_block.shape[0]
        db["nframes_per_folder"][ifile] += frame_block.shape[0]


def _validate_isxd_to_binary_inputs(dbs, reg_file_chan2):
    if not HAS_ISX:
        raise ImportError("Inscopix isx is required for this file type, please 'pip install isx'")

    nplanes = dbs[0]["nplanes"]
    nchannels = dbs[0]["nchannels"]
    if nchannels not in (1, 2):
        raise ValueError(f"ISXD loader supports nchannels of 1 or 2, got {nchannels}.")
    if nplanes > 1 and nchannels == 2:
        raise ValueError(
            "ISXD files support either multi-plane or multi-channel data, but not both. "
            "Please set either nplanes == 1 or nchannels == 1."
        )
    if nchannels == 2 and reg_file_chan2 is None:
        raise ValueError("Second channel output file handle is required when nchannels == 2.")
    if nchannels == 2 and dbs[0]["functional_chan"] not in (1, 2):
        raise ValueError("functional_chan must be 1 or 2 when nchannels == 2.")

    return nplanes, nchannels


def isxd_to_binary(dbs, settings, reg_file, reg_file_chan2):
    """  finds Inscopix isxd files and writes them to binaries

    Parameters
    ----------
    dbs : list of dict
        Database dictionaries for each plane. Must contain keys "file_list",
        "nplanes", "nchannels", "batch_size", and "functional_chan".
        Updated in-place with "Ly", "Lx", "nframes", "nframes_per_folder",
        "meanImg", and "meanImg_chan2".
    settings : dict
        Suite2p settings dictionary, saved alongside each plane's database.
    reg_file : list of file objects
        Opened binary files for writing each plane's functional channel data.
    reg_file_chan2 : list of file objects
        Opened binary files for writing each plane's second channel data
        (used only when nchannels == 2).

    Returns
    -------
    dbs : list of dict
        Updated database dictionaries with image dimensions, frame counts, and
        mean images populated.

    """
    nplanes, nchannels = _validate_isxd_to_binary_inputs(dbs, reg_file_chan2)

    nfunc = dbs[0]["functional_chan"] - 1 if nchannels == 2 else 0
    interleave = 2 if nchannels == 2 else nplanes
    batch_size = max(1, interleave * int(np.ceil(dbs[0]["batch_size"] / interleave)))

    # open all binary files for writing
    file_list = dbs[0]["file_list"]
    has_initialized = False
    ly, lx = None, None
    for ifile, fname in enumerate(file_list):
        f = isx.Movie.read(fname)
        nframes = f.timing.num_samples
        if nframes == 0:
            continue
        iblocks = np.arange(0, nframes, batch_size)
        if iblocks[-1] < nframes:
            iblocks = np.append(iblocks, nframes)

        # data = nframes x width x height x (nplanes OR nchannels)
        # loop over all frames
        for ichunk, onset in enumerate(iblocks[:-1]):
            offset = iblocks[ichunk + 1]
            im = np.array([f.get_frame_data(x) for x in np.arange(onset, offset)])
            ly, lx = im.shape[1], im.shape[2]

            if not has_initialized:
                _init_plane_accumulators(
                    dbs=dbs,
                    nplanes=nplanes,
                    nchannels=nchannels,
                    file_count=len(file_list),
                    ly=ly,
                    lx=lx,
                )
                has_initialized = True

            if nchannels == 2:
                # multi-channel data (nplanes == 1)
                for ichan in range(2):
                    channel_frames = im[ichan::nchannels]
                    if ichan == nfunc:
                        _write_and_accumulate(
                            db=dbs[0],
                            frame_block=channel_frames,
                            out_file=reg_file[0],
                            mean_key="meanImg",
                            ifile=ifile,
                            count_frames=True,
                        )
                    else:
                        _write_and_accumulate(
                            db=dbs[0],
                            frame_block=channel_frames,
                            out_file=reg_file_chan2[0],
                            mean_key="meanImg_chan2",
                            ifile=ifile,
                            count_frames=False,
                        )
            elif nchannels == 1:
                # single-channel or multi-plane data
                for j in range(nplanes):
                    plane_frames = im[j::nplanes]
                    _write_and_accumulate(
                        db=dbs[j],
                        frame_block=plane_frames,
                        out_file=reg_file[j],
                        mean_key="meanImg",
                        ifile=ifile,
                        count_frames=True,
                    )
            else:
                raise ValueError(f"Unsupported nchannels value: {nchannels}")

    if not has_initialized:
        raise ValueError("No frames were found in ISXD input files.")

    # write ops files
    do_registration = settings["run"]["do_registration"]
    for db in dbs:
        db["Ly"] = ly
        db["Lx"] = lx
        if not do_registration:
            db["yrange"] = np.array([0, db["Ly"]])
            db["xrange"] = np.array([0, db["Lx"]])
        db["meanImg"] /= db["nframes"]
        if nchannels == 2:
            db["meanImg_chan2"] /= db["nframes"]
        # Save db and settings to each plane folder
        np.save(db["db_path"], db)
        np.save(db["settings_path"], settings)
    # close all binary files and write ops files
    for j in range(0, nplanes):
        reg_file[j].close()
        if nchannels == 2:
            reg_file_chan2[j].close()
    return dbs
