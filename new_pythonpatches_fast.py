# from ROOT import TFile
# import numpy as np
# from sys import argv
# from os import listdir
# from os.path import isfile, join
# import os, json
# import argparse

# from utils import read_config

# def main(argv):

#     parser = argparse.ArgumentParser(description='Makes training data set for EM vs track separation')
#     parser.add_argument('-c', '--config', help="JSON with script configuration", default='config.json')
#     parser.add_argument('-t', '--type', help="Input file format")
#     parser.add_argument('-i', '--input', help="Input directory")
#     parser.add_argument('-o', '--output', help="Output directory")
#     args = parser.parse_args()

#     config = read_config(args.config)
#     print('#'*50, '\nPrepare data for CNN')

#     # Configuration parameters
#     INPUT_TYPE = config['prepare_data_em_track']['input_type'] if args.type is None else args.type
#     INPUT_DIR  = config['prepare_data_em_track']['input_dir']  if args.input is None else args.input
#     OUTPUT_DIR = config['prepare_data_em_track']['output_dir'] if args.output is None else args.output

#     PATCH_SIZE_W = config['prepare_data_em_track']['patch_size_w']
#     PATCH_SIZE_D = config['prepare_data_em_track']['patch_size_d']

#     doing_nue = config['prepare_data_em_track']['doing_nue']
#     selected_view_idx = config['prepare_data_em_track']['selected_view_idx']
#     patch_fraction = config['prepare_data_em_track']['patch_fraction']
#     empty_fraction = config['prepare_data_em_track']['empty_fraction']
#     clean_track_fraction = config['prepare_data_em_track']['clean_track_fraction']
#     muon_track_fraction = config['prepare_data_em_track']['muon_track_fraction']
#     crop_event = config['prepare_data_em_track']['crop_event']
#     blur_kernel = np.asarray(config['prepare_data_em_track']['blur'])
#     white_noise = config['prepare_data_em_track']['noise']
#     coherent_noise = config['prepare_data_em_track']['coherent']

#     print(f'Using {patch_fraction}% of data from view {selected_view_idx}')
#     print(f'Noise: blur={blur_kernel}, white={white_noise}, coherent={coherent_noise}')

#     db = []
#     db_y = []
#     cnt_ind = 0

#     fcount = 0
#     event_list = []
#     # if INPUT_TYPE == "root":
#     #     # Collect ROOT files in directory
#     #     fnames = [f for f in os.listdir(INPUT_DIR) if f.endswith('.root')]
#     #     for n in fnames:
#     #         rootFile = TFile(os.path.join(INPUT_DIR, n))
#     #         # Build list of event keys from the 'datadump' directory inside the ROOT file
#     #         rootModule = 'datadump'
#     #         keys = []
#     #         if rootFile.Get(rootModule):
#     #             for k in rootFile.Get(rootModule).GetListOfKeys():
#     #                 name = k.GetName()
#     #                 if name.endswith('_raw'):
#     #                     keys.append(rootModule+'/'+name[:-4])
#     #         event_list.append((rootFile, keys))
#     # else:
#     #     keys = [f[:-4] for f in os.listdir(INPUT_DIR) if f.endswith('.raw')]
#     #     event_list.append((INPUT_DIR, keys))

#     rootModule = 'datadump'

#     # if INPUT_TYPE == "root":
#     #     INPUT_DIR = "/exp/dune/app/users/kadhikar/CNNimage"
#     #     fnames = [f for f in os.listdir(INPUT_DIR) if f.endswith('.root')]

#     #     for n in fnames:
#     #         rootFile = TFile(os.path.join(INPUT_DIR, n))

#     #         # print("FILE =", n)

#     #         # rootFile.ls()
#     #         # epTree = rootFile.Get("pandoraEndpoints")
#     #         # print("epTree =", epTree)
#     #         # if epTree:
#     #         #     print("entries =", epTree.GetEntries())


#     #         if not rootFile.Get(rootModule):
#     #             print("WARNING: no datadump directory in", n)
#     #             continue

#     #         keys = [
#     #             rootModule + '/' + k.GetName()[:-4]
#     #             for k in rootFile.Get(rootModule).GetListOfKeys()
#     #             if '_raw' in k.GetName()
#     #         ]

#     #         event_list.append((rootFile, keys))

#     if INPUT_TYPE == "root":

#         INPUT_DIR = "/exp/dune/app/users/kadhikar/CNNimage"

#         fnames = [
#             f for f in os.listdir(INPUT_DIR)
#             if f.endswith(".root")
#         ]

#         for n in fnames:

#             fullpath = os.path.join(INPUT_DIR, n)

#             rootFile = TFile.Open(fullpath)

#             if not rootFile or rootFile.IsZombie():
#                 print("Cannot open", fullpath)
#                 continue

#             epTree = rootFile.Get("pandoraEndpoints")

#             print("FILE =", n)
#             print("epTree =", epTree)

#             if not epTree:
#                 print("NO pandoraEndpoints tree found")
#                 continue

#             print("endpoint entries =", epTree.GetEntries())

#             datadump = rootFile.Get(rootModule)

#             if not datadump:
#                 print("NO datadump directory found")
#                 continue

#             keys = [
#                 rootModule + "/" + k.GetName()[:-4]
#                 for k in datadump.GetListOfKeys()
#                 if "_raw" in k.GetName()
#             ]

#             event_list.append(
#                 {
#                     "file": rootFile,
#                     "tree": epTree,
#                     "keys": keys
#                 }
#             )

#     else:
#         keys = [f[:-4] for f in os.listdir(INPUT_DIR) if '.raw' in f]
#         event_list.append((INPUT_DIR, keys))

#     # for entry in event_list:
#     #     folder = entry[0]
#     #     event_names = entry[1]

#     #     # Open the Pandora endpoint tree if reading ROOT files
#     #     epTree = None
#     #     if hasattr(folder, 'Get'):
#     #         epTree = folder.Get("pandoraEndpoints")

#     #         print("epTree =", epTree)

#     for entry in event_list:

#         folder = entry["file"]
#         epTree = entry["tree"]
#         event_names = entry["keys"]

#         print("endpoint entries =", epTree.GetEntries())

#         for evname in event_names:
#             # Parse run, subrun, event from evname (format "event_<evt>_run_<run>_subrun_<subrun>_...")
#             m = __import__('re').match(r"event_(\d+)_run_(\d+)_subrun_(\d+)_", evname)
#             if not m:
#                 continue
#             evt_no = int(m.group(1))
#             runNumber = int(m.group(2))
#             subRunNumber = int(m.group(3))
#             fcount += 1
#             print('Process event', fcount, evname, 'NO.', evt_no)

#             # Retrieve data arrays (raw, deposit, pdg, tracks, showers) for this event
#             raw, deposit, pdg, tracks, showers = get_data(folder, evname, PATCH_SIZE_D//2 + 2,
#                                                          crop_event, blur_kernel, white_noise, coherent_noise)
#             if raw is None:
#                 print('Skip empty event...')
#                 continue
#             # Clean NaNs in raw
#             raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

#             print('Tracks', np.sum(tracks), 'showers', np.sum(showers),
#                   'michels', np.count_nonzero((pdg & 0xF000) == 0x2000))

#             # === Replace pixel loops with endpoint-based patch extraction ===
#             if epTree is None:
#                 continue

#             # Collect all endpoints in this event and plane
#             endpoints = []
#             for ep in epTree:
#                 # if ep.run != runNumber or ep.subrun != subRunNumber:
#                 #     continue
#                 if ep.run != runNumber:
#                     continue

#                 if ep.subrun != subRunNumber:
#                     continue

#                 if ep.event != evt_no:
#                     continue

#                 if ep.plane != selected_view_idx:
#                     continue
#                 endpoints.append(ep)

#             if not endpoints:
#                 continue

#             # For each endpoint, create a patch and label it
#             for ep in endpoints:
#                 # Compute patch center (round to nearest integer)
#                 # wire = int(round(ep.center_wire_abs))
#                 # tick = int(round(ep.center_tick_abs))

#                 wire = int(round(ep.center_wire_local))
#                 tick = int(round(ep.center_tick_local))
#                 # Patch boundaries
#                 x_start = wire - PATCH_SIZE_W//2
#                 y_start = tick - PATCH_SIZE_D//2
#                 x_stop  = x_start + PATCH_SIZE_W
#                 y_stop  = y_start + PATCH_SIZE_D
#                 # Check bounds
#                 if x_start < 0 or y_start < 0 or x_stop > raw.shape[0] or y_stop > raw.shape[1]:
#                     continue

#                 # Extract patch from raw data
#                 patch = raw[x_start:x_stop, y_start:y_stop].astype(np.float32)
#                 # One-hot label: [Em,Trk,Michel,None], here only Michel or None
#                 label = np.zeros(4, dtype=np.int32)
#                 if hasattr(ep, 'truth_has_michel') and ep.truth_has_michel == 1:
#                     label[2] = 1  # Michel endpoint
#                 else:
#                     label[3] = 1  # Non-Michel endpoint

#                 db.append(patch)
#                 db_y.append(label)
#                 cnt_ind += 1

#             # ================================================================

#     # Convert lists to numpy arrays
#     if len(db) == 0:
#         print("No patches found.")
#         return

#     X_final = np.stack(db).astype(np.float32)
#     Y_final = np.stack(db_y).astype(np.int32)

#     X_final = np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)
#     print("Final dataset shape: X:", X_final.shape, "Y:", Y_final.shape)

#     # Save to .npy files
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     x_path = os.path.join(OUTPUT_DIR, f"X_view{selected_view_idx}.npy")
#     y_path = os.path.join(OUTPUT_DIR, f"Y_view{selected_view_idx}.npy")
#     np.save(x_path, X_final)
#     np.save(y_path, Y_final)
#     print("Saved X to", x_path)
#     print("Saved Y to", y_path)

# if __name__ == "__main__":
#     main(argv)


from ROOT import TFile
import numpy as np
from sys import argv
import os, argparse, re

from utils import read_config


# -----------------------------------------------------------------------------
# Fast local replacements for the slow helpers in utils.py.
# The original utils.hist2array() loops over every TH2 bin and calls
# GetBinContent() from Python. That is the dominant runtime for full-event
# ROOT histograms. These functions read the ROOT TH2 backing buffer in one
# NumPy operation and keep the same array convention: [wire, tick].
# -----------------------------------------------------------------------------

def fast_hist2array(hist, dtype):
    if hist is None:
        raise RuntimeError("Missing ROOT histogram")

    nx = hist.GetNbinsX()
    ny = hist.GetNbinsY()

    # ROOT stores TH2 bins as (ny+2) rows of (nx+2) x-bins, including
    # underflow/overflow bins. Strip those borders and transpose to preserve
    # the old utils.hist2array() shape: arr[wire, tick].
    raw = np.frombuffer(hist.GetArray(), dtype=dtype, count=hist.GetSize())
    return raw.reshape((ny + 2, nx + 2))[1:ny + 1, 1:nx + 1].T.copy()


def fast_apply_blur(a, kernel):
    if kernel is None or kernel.shape[0] < 2:
        return a

    margin_left = kernel.shape[0] >> 1
    margin_right = kernel.shape[0] - margin_left - 1
    stop = a.shape[0] - margin_right
    if stop <= margin_left:
        return a

    src = a.copy()
    out = src.copy()
    blurred = np.zeros_like(src[margin_left:stop, :])
    for i, weight in enumerate(kernel):
        blurred += weight * src[i:stop + i - margin_left, :]
    out[margin_left:stop, :] = blurred
    return out


def fast_add_white_noise(a, sigma):
    if sigma is None or sigma == 0:
        return a
    a += np.random.normal(0, sigma, a.shape)
    return a


def fast_add_coherent_noise(a, sigma):
    if sigma is None or sigma == 0:
        return a

    a += np.random.normal(0, sigma, a.shape)
    amps1 = np.random.normal(1, 0.1, a.shape[0])
    amps2 = np.random.normal(1, 0.1, 1 + (a.shape[0] >> 5))

    group_noise = None
    group_amp = 1.0
    for w in range(a.shape[0]):
        if (w & 31) == 0:
            group_noise = np.random.normal(0, sigma, a.shape[1])
            group_amp = amps2[w >> 5]
        a[w] += group_amp * amps1[w] * group_noise
    return a


def fast_get_event_bounds(A, drift_margin=0):
    cum = np.cumsum(np.sum(A, axis=0))
    start_ind = np.max([0, np.where(cum > cum[-1] * 0.005)[0][0] - drift_margin])
    end_ind = np.min([A.shape[1], np.where(cum > cum[-1] * 0.995)[0][0] + drift_margin])
    return start_ind, end_ind


def get_data(folder, fname, drift_margin=0, crop=True, blur=None, white_noise=0, coherent_noise=0):
    print('Reading', fname)
    try:
        if isinstance(folder, TFile):
            A_raw = fast_hist2array(folder.Get(fname + '_raw'), np.float32)
            A_deposit = fast_hist2array(folder.Get(fname + '_deposit'), np.float32)
            A_pdg = fast_hist2array(folder.Get(fname + '_pdg'), np.int32)

            A_raw = np.nan_to_num(A_raw).astype(np.float32)
            A_pdg = np.nan_to_num(A_pdg).astype(np.int32)
        else:
            A_raw = np.genfromtxt(folder + '/' + fname + '.raw', delimiter=' ', dtype=np.float32)
            A_deposit = np.genfromtxt(folder + '/' + fname + '.deposit', delimiter=' ', dtype=np.float32)
            A_pdg = np.genfromtxt(folder + '/' + fname + '.pdg', delimiter=' ', dtype=np.int32)
    except Exception as exc:
        print('Bad event, return empty arrays:', exc)
        return None, None, None, None, None

    if A_raw.shape[0] < 8 or A_raw.shape[1] < 8:
        return None, None, None, None, None

    test_pdg = np.sum(A_pdg)
    test_dep = np.sum(A_deposit)
    test_raw = np.sum(A_raw)
    if test_raw == 0.0 or test_dep == 0.0 or test_pdg == 0:
        return None, None, None, None, None

    print(test_raw, test_dep, test_pdg)

    if crop:
        evt_start_ind, evt_stop_ind = fast_get_event_bounds(A_deposit, drift_margin)
        A_raw = A_raw[:, evt_start_ind:evt_stop_ind]
        A_deposit = A_deposit[:, evt_start_ind:evt_stop_ind]
        A_pdg = A_pdg[:, evt_start_ind:evt_stop_ind]
    else:
        evt_start_ind = 0
        evt_stop_ind = A_raw.shape[1]
    print(evt_start_ind, evt_stop_ind)

    A_raw = fast_apply_blur(A_raw, blur)
    A_raw = fast_add_white_noise(A_raw, white_noise)
    A_raw = fast_add_coherent_noise(A_raw, coherent_noise)

    deposit_th_ind = A_deposit < 2.0e-5
    A_pdg[deposit_th_ind] = 0

    tracks = A_pdg.copy()
    showers = A_pdg.copy()
    tracks[(A_pdg & 0x0FFF) == 11] = 0
    tracks[tracks > 0] = 1
    showers[(A_pdg & 0x0FFF) != 11] = 0
    showers[showers > 0] = 1
    return A_raw, A_deposit, A_pdg, tracks, showers


def get_patch(a, wire, drift, wsize, dsize):
    # The main loop already rejects edge-padded endpoint patches, so this can
    # be a direct NumPy slice instead of a Python loop over every patch pixel.
    half_w = wsize // 2
    half_d = dsize // 2
    return a[wire - half_w:wire - half_w + wsize,
             drift - half_d:drift - half_d + dsize].copy()


def main(argv):

    parser = argparse.ArgumentParser(description='Makes endpoint-based Michel training patches')
    parser.add_argument('-c', '--config', help="JSON with script configuration", default='config.json')
    parser.add_argument('-t', '--type', help="Input file format")
    parser.add_argument('-i', '--input', help="Input directory")
    parser.add_argument('-o', '--output', help="Output directory")
    args = parser.parse_args()

    config = read_config(args.config)

    print('#' * 50, '\nPrepare data for CNN')

    if args.type is None: INPUT_TYPE = config['prepare_data_em_track']['input_type']
    else: INPUT_TYPE = args.type

    if args.input is None: INPUT_DIR = config['prepare_data_em_track']['input_dir']
    else: INPUT_DIR = args.input

    if args.output is None: OUTPUT_DIR = config['prepare_data_em_track']['output_dir']
    else: OUTPUT_DIR = args.output

    PATCH_SIZE_W = config['prepare_data_em_track']['patch_size_w']
    PATCH_SIZE_D = config['prepare_data_em_track']['patch_size_d']

    selected_view_idx = config['prepare_data_em_track']['selected_view_idx']
    patch_fraction = config['prepare_data_em_track']['patch_fraction']
    crop_event = config['prepare_data_em_track']['crop_event']

    blur_kernel = np.asarray(config['prepare_data_em_track']['blur'])
    white_noise = config['prepare_data_em_track']['noise']
    coherent_noise = config['prepare_data_em_track']['coherent']

    print('Using', patch_fraction, '% of data from view', selected_view_idx)
    print(f'Noise: blur={blur_kernel}, white={white_noise}, coherent={coherent_noise}')
    print('Patch size:', PATCH_SIZE_W, 'x', PATCH_SIZE_D)

    max_capacity = 17000000

    db = []
    db_y = []

    cnt_michel = 0
    cnt_nonmichel = 0
    cnt_track = 0
    cnt_shower = 0
    cnt_empty = 0
    cnt_zero_patch = 0
    cnt_skipped_bounds = 0
    cnt_no_endpoint_match = 0

    fcount = 0
    rootModule = 'datadump'
    event_list = []

    # ============================================================
    # ROOT INPUT BLOCK: kept close to your old working code
    # ============================================================

    if INPUT_TYPE == "root":

        # Same hard-coded directory as your old working script
        INPUT_DIR = "/exp/dune/app/users/kadhikar/CNNimage"

        # IMPORTANT: only process analyzer output files, not reco2.root
        fnames = [
            f for f in os.listdir(INPUT_DIR)
            if f.endswith(".root") and f.startswith("reco_hist")
        ]

        if len(fnames) == 0:
            print("No reco_hist*.root files found in", INPUT_DIR)
            return

        for n in fnames:

            fullpath = os.path.join(INPUT_DIR, n)
            rootFile = TFile.Open(fullpath)

            if not rootFile or rootFile.IsZombie():
                print("Cannot open", fullpath)
                continue

            print("FILE =", n)

            datadump = rootFile.Get(rootModule)
            if not datadump:
                print("NO datadump directory found in", n)
                continue

            # Try both possible locations for the endpoint tree
            epTree = rootFile.Get("pandoraEndpoints")

            if not epTree:
                epTree = rootFile.Get(rootModule + "/pandoraEndpoints")

            if not epTree:
                print("NO pandoraEndpoints tree found in", n)
                print("File contents are:")
                rootFile.ls()
                print("datadump contents are:")
                datadump.ls()
                continue

            print("endpoint entries =", epTree.GetEntries())

            keys = [
                rootModule + '/' + k.GetName()[:-4]
                for k in datadump.GetListOfKeys()
                if '_raw' in k.GetName()
            ]

            print("number of raw images =", len(keys))

            event_list.append({
                "file": rootFile,
                "tree": epTree,
                "keys": keys
            })

    else:
        keys = [f[:-4] for f in os.listdir(INPUT_DIR) if '.raw' in f]
        event_list.append({
            "file": INPUT_DIR,
            "tree": None,
            "keys": keys
        })

    # ============================================================
    # MAIN EVENT LOOP
    # ============================================================

    for entry in event_list:

        folder = entry["file"]
        epTree = entry["tree"]
        event_names = entry["keys"]

        if epTree is None:
            print("Endpoint tree is missing; cannot make endpoint patches.")
            continue

        print("Looping over endpoint tree with", epTree.GetEntries(), "entries")

        # ------------------------------------------------------------
        # Cache endpoint entries once.
        # Do NOT append the PyROOT ep object directly.
        # Store plain Python values.
        # ------------------------------------------------------------

        endpoint_cache = {}
        has_ancestry_truth = bool(epTree.GetBranch("truth_has_michel_ancestry"))

        for iep in range(epTree.GetEntries()):

            epTree.GetEntry(iep)

            if int(epTree.plane) != int(selected_view_idx):
                continue

            key = (
                int(epTree.run),
                int(epTree.subrun),
                int(epTree.event),
                int(epTree.tpc),
                int(epTree.plane)
            )

            endpoint_info = {
                "track_key": int(epTree.track_key),
                "endpoint": int(epTree.endpoint),
                "wire": int(round(float(epTree.center_wire_local))),
                "tick": int(round(float(epTree.center_tick_local))),
                "patch_fits": int(epTree.patch_fits),
                "truth_has_michel": int(epTree.truth_has_michel),
                "truth_michel_pixels": int(epTree.truth_michel_pixels),
                "truth_muon_pixels": int(epTree.truth_muon_pixels),
                "truth_track_pixels": int(epTree.truth_track_pixels),
                "truth_shower_pixels": int(epTree.truth_shower_pixels),
                "truth_has_michel_ancestry": 0,
                "truth_reco_track_id": 0,
                "truth_reco_track_pdg": 0,
                "truth_reco_track_is_primary_muon": 0,
                "truth_reco_track_has_michel_decay": 0,
                "truth_track_charge": 0.0,
                "truth_shower_charge": 0.0,
                "truth_michel_charge": 0.0,
            }

            if has_ancestry_truth:
                endpoint_info["truth_has_michel_ancestry"] = int(epTree.truth_has_michel_ancestry)
                if epTree.GetBranch("truth_reco_track_id"):
                    endpoint_info["truth_reco_track_id"] = int(epTree.truth_reco_track_id)
                    endpoint_info["truth_reco_track_pdg"] = int(epTree.truth_reco_track_pdg)
                    endpoint_info["truth_reco_track_is_primary_muon"] = int(epTree.truth_reco_track_is_primary_muon)
                    endpoint_info["truth_reco_track_has_michel_decay"] = int(epTree.truth_reco_track_has_michel_decay)
                endpoint_info["truth_track_charge"] = float(epTree.truth_track_charge)
                endpoint_info["truth_shower_charge"] = float(epTree.truth_shower_charge)
                endpoint_info["truth_michel_charge"] = float(epTree.truth_michel_charge)

            endpoint_cache.setdefault(key, []).append(endpoint_info)

        print("Cached endpoint event/TPC/plane keys =", len(endpoint_cache))

        for evname in event_names:

            # evname looks like:
            # datadump/event_55200_run_74496296_subrun_1_tpc_12_view_2
            m = re.search(
                r"event_(\d+)_run_(\d+)_subrun_(\d+)_tpc_(\d+)_view_(\d+)",
                evname
            )

            if not m:
                print("Could not parse event name:", evname)
                continue

            evt_no = int(m.group(1))
            runNumber = int(m.group(2))
            subRunNumber = int(m.group(3))
            tpc_idx = int(m.group(4))
            view_idx = int(m.group(5))

            if view_idx != selected_view_idx:
                continue

            fcount += 1
            print('Process event', fcount, evname, 'NO.', evt_no)

            raw, deposit, pdg, tracks, showers = get_data(
                folder,
                evname,
                PATCH_SIZE_D // 2 + 2,
                crop_event,
                blur_kernel,
                white_noise,
                coherent_noise
            )

            if raw is None:
                print('Skip empty event...')
                continue

            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

            print(
                'Tracks', np.sum(tracks),
                'showers', np.sum(showers),
                'michels', np.count_nonzero((pdg & 0xF000) == 0x2000)
            )

            endpoint_key = (
                runNumber,
                subRunNumber,
                evt_no,
                tpc_idx,
                view_idx
            )

            endpoints = endpoint_cache.get(endpoint_key, [])

            if len(endpoints) == 0:
                cnt_no_endpoint_match += 1
                print("No endpoints matched this event/TPC/view")
                continue

            print("Matched endpoints:", len(endpoints))

            # ========================================================
            # NEW PATCH LOGIC:
            # Instead of looping over every hit/pixel,
            # loop over Pandora track endpoints.
            # This gives 2 patches per track if both endpoints exist.
            # ========================================================

            for ep in endpoints:

                wire = ep["wire"]
                tick = ep["tick"]

                x_start = wire - PATCH_SIZE_W // 2
                x_stop = x_start + PATCH_SIZE_W

                y_start = tick - PATCH_SIZE_D // 2
                y_stop = y_start + PATCH_SIZE_D

                patch_is_edge_padded = (
                    x_start < 0 or
                    y_start < 0 or
                    x_stop > raw.shape[0] or
                    y_stop > raw.shape[1]
                )

                if patch_is_edge_padded:
                    cnt_skipped_bounds += 1
                    # Old endpoint script kept edge-padded patches because get_patch() fills
                    # outside-image pixels with zero:
                    # continue  # was intentionally disabled before this fix
                    # Turn this back on for endpoint training: a patch whose centre is too close
                    # to an image boundary is not a clean ROI around the track extremum and can
                    # look like random detector noise after zero padding.
                    continue

                patch = get_patch(raw, wire, tick, PATCH_SIZE_W, PATCH_SIZE_D).astype(np.float32)

                patch = np.nan_to_num(
                    patch,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )

                scale = np.max(np.abs(patch))
                if scale > 0:
                    patch = patch / scale
                else:
                    cnt_zero_patch += 1
                    continue

                # Binary Michel-tagging target used by the current CNN workflow:
                # [NonMichel, Michel]. The older 4-class target was [Track, Shower, Michel, Empty].
                # target = np.zeros(4, dtype=np.int32)
                target = np.zeros(2, dtype=np.int32)

                x0 = max(x_start, 0)
                x1 = min(x_stop, raw.shape[0])
                y0 = max(y_start, 0)
                y1 = min(y_stop, raw.shape[1])

                # Full-patch truth counts are kept only as diagnostics. Using them for labels
                # can let unrelated charge elsewhere in the 44x48 patch dominate the endpoint.
                full_patch_track_pixels = np.count_nonzero(tracks[x0:x1, y0:y1])
                full_patch_shower_pixels = np.count_nonzero(showers[x0:x1, y0:y1])

                # The analyser stores two endpoint-local truth definitions. New ROOT files
                # contain MCParticle-ancestry charge counters; old files only have pixel counters.
                # Prefer ancestry truth because it follows Geant4 TrackID -> MCParticle mother
                # chains and tags Michel electrons from decaying muons.
                if has_ancestry_truth:
                    track_score = ep["truth_track_charge"]
                    shower_score = ep["truth_shower_charge"]
                    is_michel_endpoint = ep["truth_has_michel_ancestry"] == 1
                else:
                    # Fallback for old ROOT files made before the ancestry branches existed.
                    track_score = ep["truth_track_pixels"]
                    shower_score = ep["truth_shower_pixels"]
                    # if ep["truth_has_michel"] == 1:
                    is_michel_endpoint = (
                        ep["truth_has_michel"] == 1 or
                        ep["truth_michel_pixels"] >= 3
                    )

                if is_michel_endpoint:
                    # Old 4-class label:
                    # target[2] = 1
                    target[1] = 1
                    cnt_michel += 1
                else:
                    # Old 4-class non-Michel handling classified the endpoint as Track/Shower/Empty:
                    # if track_score > 0 and track_score >= shower_score:
                    #     target[0] = 1
                    # elif shower_score > 0:
                    #     target[1] = 1
                    # else:
                    #     target[3] = 1
                    # Current workflow is binary Michel tagging, so all non-Michel endpoint ROIs
                    # are one background class regardless of their track/shower/empty truth content.
                    target[0] = 1
                    if track_score > 0 and track_score >= shower_score:
                        cnt_track += 1
                    elif shower_score > 0:
                        cnt_shower += 1
                    else:
                        cnt_empty += 1
                    cnt_nonmichel += 1

                if len(db) < max_capacity:
                    db.append(patch.astype(np.float32))
                    db_y.append(target.astype(np.int32))
                else:
                    print("MAX CAPACITY REACHED!!!")
                    break

    # ============================================================
    # FINAL SAVE BLOCK: same idea as your old script
    # ============================================================

    print("Added", len(db), "endpoint patches")
    print("Michel endpoint patches:", cnt_michel)
    print("Non-Michel endpoint patches:", cnt_nonmichel)
    print("  Non-Michel truth diagnostic Track-like:", cnt_track)
    print("  Non-Michel truth diagnostic Shower-like:", cnt_shower)
    print("  Non-Michel truth diagnostic Empty-like:", cnt_empty)
    print("Binary target columns: [NonMichel, Michel]")
    print("Edge-padded endpoint patches skipped:", cnt_skipped_bounds)
    print("Skipped zero/empty patches:", cnt_zero_patch)
    print("Events with no endpoint match:", cnt_no_endpoint_match)

    if len(db) == 0:
        print("No patches found.")
        return

    X_final = np.stack(db).astype(np.float32)
    Y_final = np.stack(db_y).astype(np.int32)

    X_final = np.nan_to_num(
        X_final,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    print("Final dataset shape:", X_final.shape)
    print("Final label shape:", Y_final.shape)
    print("NaNs in X:", np.isnan(X_final).sum())
    print("Infs in X:", np.isinf(X_final).sum())

    assert X_final.shape[0] == Y_final.shape[0]
    assert X_final.shape[1:] == (PATCH_SIZE_W, PATCH_SIZE_D)
    assert np.isfinite(X_final).all()

    # Keep your old output location and naming convention
    outdir = "/exp/dune/app/users/kadhikar/CNNimage"

    # Fast-copy output names intentionally do not overwrite the original script outputs.
    # x_path = os.path.join(outdir, f"db_view_{selected_view_idx}_x.npy")
    # y_path = os.path.join(outdir, f"db_view_{selected_view_idx}_y.npy")
    x_path = os.path.join(outdir, f"db_view_{selected_view_idx}_x_fast.npy")
    y_path = os.path.join(outdir, f"db_view_{selected_view_idx}_y_fast.npy")

    tmp_x = x_path + ".tmp.npy"
    tmp_y = y_path + ".tmp.npy"

    np.save(tmp_x, X_final)
    np.save(tmp_y, Y_final)

    os.replace(tmp_x, x_path)
    os.replace(tmp_y, y_path)

    print("Saved:", x_path)
    print("Saved:", y_path)


if __name__ == "__main__":
    main(argv)
