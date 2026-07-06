from ROOT import TFile
import numpy as np
from sys import argv
from os import listdir
from os.path import isfile, join
import os, json
import argparse
#import h5py


from utils import read_config, get_data, get_patch

def main(argv):

    parser = argparse.ArgumentParser(description='Makes training data set for EM vs track separation')
    parser.add_argument('-c', '--config', help="JSON with script configuration", default='config.json')
    parser.add_argument('-t', '--type', help="Input file format")
    parser.add_argument('-i', '--input', help="Input directory")
    parser.add_argument('-o', '--output', help="Output directory")
    args = parser.parse_args()

    config = read_config(args.config)

    print ('#'*50,'\nPrepare data for CNN')
    if args.type is None: INPUT_TYPE   = config['prepare_data_em_track']['input_type']
    else: INPUT_TYPE = args.type
    if args.input is None: INPUT_DIR   = config['prepare_data_em_track']['input_dir']
    else: INPUT_DIR = args.input
    if args.output is None: OUTPUT_DIR = config['prepare_data_em_track']['output_dir']
    else: OUTPUT_DIR = args.output

    PATCH_SIZE_W = config['prepare_data_em_track']['patch_size_w']
    PATCH_SIZE_D = config['prepare_data_em_track']['patch_size_d']

    print ('Using %s as input dir, and %s as output dir' % (INPUT_DIR, OUTPUT_DIR))
    if INPUT_TYPE == 'root': print ('Reading from ROOT file')
    else: print ('Reading from TEXT files')
    print ('#'*50)

    doing_nue = config['prepare_data_em_track']['doing_nue']                       # set to true for nu_e events (will skip more showers)
    selected_view_idx = config['prepare_data_em_track']['selected_view_idx']       # set the view id
    patch_fraction = config['prepare_data_em_track']['patch_fraction']             # percent of used patches
    empty_fraction = config['prepare_data_em_track']['empty_fraction']             # percent of "empty background" patches
    clean_track_fraction = config['prepare_data_em_track']['clean_track_fraction'] # percent of selected patches, where only a clean track is present
    muon_track_fraction = config['prepare_data_em_track']['muon_track_fraction']   # ***** new: preselect muos, they are many *****
    crop_event = config['prepare_data_em_track']['crop_event']                     # use true only if no crop on LArSoft level and not a noise dump

    # =====================================================
    # MICHEL-FOCUSED DATASET CONFIGURATION
    # =====================================================

    michel_focused = True

    # keep all true Michel patches
    keep_all_true_michels = True

    # only keep difficult non-Michel backgrounds
    keep_hard_backgrounds_only = True

    # radius around true Michel pixels
    michel_context_radius = 8

    # keep only small fraction of generic backgrounds
    background_keep_fraction = 2.0

    # minimum muon pixels for endpoint-like topology
    min_muon_pixels = 3

    # avoid gigantic datasets
    max_patches_per_event = 500
    ########################################################

    blur_kernel = np.asarray(config['prepare_data_em_track']['blur'])              # add blur in wire direction with given kernel if it is not empty (only for tests)
    white_noise = config['prepare_data_em_track']['noise']                         # add gauss noise with given sigma if value > 0 (only for tests)
    coherent_noise = config['prepare_data_em_track']['coherent']                   # add coherent (groups of 32 wires) gauss noise with given sigma if value > 0 (only for tests)

    print ('Using', patch_fraction, '% of data from view', selected_view_idx)
    print ('Using', muon_track_fraction, '% of muon points')
    if doing_nue: print ('Neutrino mode, will skip more showers.')

    print ('Blur kernel', blur_kernel, 'noise RMS', white_noise)

    max_capacity = 17000000
    #db = np.zeros((max_capacity, PATCH_SIZE_W, PATCH_SIZE_D), dtype=np.float32)
    #db_y = np.zeros((max_capacity, 4), dtype=np.int32)
    db = []
    db_y = []

    patch_area = PATCH_SIZE_W * PATCH_SIZE_D

    cnt_ind = 0
    cnt_trk = 0
    cnt_sh = 0
    cnt_michel = 0
    cnt_void = 0

    fcount = 0

    rootFile = None
    rootModule = 'datadump'
    event_list = []
    if INPUT_TYPE == "root":
        INPUT_DIR = "/exp/dune/app/users/kadhikar/CNNimage"
        #fnames = [f for f in os.listdir("/exp/dune/app/users/kadhikar/CNNimage") if '.root' in f]
        fnames = [f for f in os.listdir(INPUT_DIR) if f.endswith('.root')]
        for n in fnames:
            rootFile = TFile(os.path.join(INPUT_DIR, n))
            #rootFile = TFile("reco_hisl = np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)t.root")
            keys = [rootModule+'/'+k.GetName()[:-4] for k in rootFile.Get(rootModule).GetListOfKeys() if '_raw' in k.GetName()]
            event_list.append((rootFile, keys))
    else:
        keys = [f[:-4] for f in os.listdir(INPUT_DIR) if '.raw' in f] # only main part of file name, without extension
        event_list.append((INPUT_DIR, keys)) # single entry in the list of txt files

    for entry in event_list:
        folder = entry[0]
        event_names = entry[1]

        #for evname in event_names[:2]: #for only two events
        for evname in event_names:
            finfo = evname.split('_')
            evt_no = finfo[2]
            tpc_idx = int(finfo[8])
            view_idx = int(finfo[10])

            if view_idx != selected_view_idx: continue
            fcount += 1

            print ('Process event', fcount, evname, 'NO.', evt_no)

            # get clipped data, margin depends on patch size in drift direction
            raw, deposit, pdg, tracks, showers = get_data(folder, evname, PATCH_SIZE_D//2 + 2, crop_event, blur_kernel, white_noise, coherent_noise)
            
            if raw is None:
                print ('Skip empty event...')
                continue

            # =====================================================
            # CLEAN RAW ARRAYS
            # =====================================================

            raw = np.nan_to_num(
                raw,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            print(
                'RAW stats:',
                 np.min(raw),
                 np.max(raw),
                 np.isnan(raw).sum()
            )
            #######################################################

            # CRITICAL FIX: clean NaNs immediately
            if np.isnan(raw).any():
                print("WARNING: NaNs in raw -> fixing")
                raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

            # Optional debug
            print("RAW stats:", np.min(raw), np.max(raw), np.isnan(raw).sum())

            # Replace NaNs and infs in raw
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

            pdg_michel = ((pdg & 0xF000) == 0x2000)
            vtx_map = (pdg >> 16)

            # =====================================================
            # BUILD MAP OF REGIONS NEAR TRUE MICHEL PIXELS
            # =====================================================

            near_michel_map = np.zeros_like(
                pdg_michel,
                dtype=bool
            )

            michel_coords = np.argwhere(pdg_michel)

            for wi, di in michel_coords:

                w0 = max(0, wi - michel_context_radius)
                w1 = min(raw.shape[0], wi + michel_context_radius + 1)

                d0 = max(0, di - michel_context_radius)
                d1 = min(raw.shape[1], di + michel_context_radius + 1)

                near_michel_map[w0:w1, d0:d1] = True
            ########################################################

            print ('Tracks', np.sum(tracks), 'showers', np.sum(showers), 'michels', np.sum(pdg_michel))

            sel_trk = 0
            sel_sh = 0
            sel_muon = 0
            sel_mu_near_stop = 0
            sel_michel = 0
            sel_empty = 0
            sel_clean_trk = 0
            sel_near_nu = 0
            
            event_patch_count = 0

            for i in range(raw.shape[0]):
                for j in range(raw.shape[1]):
                    is_raw_zero = (raw[i,j] < 0.01)
                    is_michel = (pdg[i,j] & 0xF000 == 0x2000) # has michel flag set, wont skip it
                    is_muon = (pdg[i,j] & 0xFFF == 13)

                    is_vtx = (vtx_map[i,j] > 0)

                    x_start = np.max([0, i - PATCH_SIZE_W//2])
                    x_stop  = np.min([raw.shape[0], x_start + PATCH_SIZE_W])

                    y_start = np.max([0, j - PATCH_SIZE_D//2])
                    y_stop  = np.min([raw.shape[1], y_start + PATCH_SIZE_D])

                    if x_stop - x_start != PATCH_SIZE_W or y_stop - y_start != PATCH_SIZE_D:
                        continue

                    is_mu_near_stop = False
                    if is_muon:
                        pdg_patch = pdg_michel[x_start+2:x_stop-2, y_start+2:y_stop-2]
                        if np.count_nonzero(pdg_patch) > 2:
                            is_mu_near_stop = True
                            sel_mu_near_stop += 1

                    vtx_patch = vtx_map[x_start+2:x_stop-2, y_start+2:y_stop-2]
                    near_vtx_count = np.count_nonzero(vtx_patch)

                    nuvtx_patch = vtx_patch & 0x4 # any nu primary vtx
                    is_near_nu = (np.count_nonzero(nuvtx_patch) > 0)

                    eff_patch_fraction = patch_fraction
                    if near_vtx_count > 0 and eff_patch_fraction < 40:
                        eff_patch_fraction = 40 # min 40% of points near vertices

                    # randomly skip fraction of patches
                    if not(is_michel | is_mu_near_stop | is_vtx | is_near_nu) and np.random.randint(10000) > int(100*eff_patch_fraction): continue

                    track_pixels = np.count_nonzero(tracks[x_start:x_stop, y_start:y_stop])
                    shower_pixels = np.count_nonzero(showers[x_start:x_stop, y_start:y_stop])

                    # =====================================================
                    # MICHEL-CANDIDATE SELECTION
                    # =====================================================

                    near_true_michel = np.count_nonzero(
                        near_michel_map[x_start:x_stop, y_start:y_stop]
                    ) > 0

                    muon_pixels = np.count_nonzero(
                        ((pdg[x_start:x_stop, y_start:y_stop] & 0xFFF) == 13)
                    )

                    has_muon_context = muon_pixels >= min_muon_pixels
                    #######################################################

                    target = np.zeros(4, dtype=np.int32)
                    if tracks[i,j] == 1:
                        target[0] = 1 # label as a track-like
                        if is_raw_zero: continue
                        # skip fraction of almost-track-only patches
                        if shower_pixels < 8 and near_vtx_count == 0 and not(is_near_nu):
                            if np.random.randint(10000) > int(100*clean_track_fraction): continue
                            else: sel_clean_trk += 1
                        else:
                            if is_muon:
                                if not(is_mu_near_stop) and np.random.randint(10000) > int(100*muon_track_fraction): continue
                                sel_muon += 1
                        cnt_trk += 1
                        sel_trk += 1
                    elif showers[i,j] == 1:
                        target[1] = 1 # label as a em-like
                        if doing_nue and not(is_near_nu): # for nu_e events (lots of showers) skip some fraction of shower patches
                            if near_vtx_count == 0 and np.random.randint(100) < 40: continue  # skip 40% of any shower
                            if shower_pixels > 0.05*patch_area and np.random.randint(100) < 50: continue
                            if shower_pixels > 0.20*patch_area and np.random.randint(100) < 90: continue
                        if is_raw_zero: continue
                        if is_michel:
                            target[2] = 1 # additionally label as a michel electron
                            cnt_michel += 1
                            sel_michel += 1
                        cnt_sh += 1
                        sel_sh += 1
                    else: # use small fraction of empty-but-close-to-something patches
                        target[3] = 1 # label an empty pixel
                        if np.random.randint(10000) < int(100*empty_fraction):
                            nclose = np.count_nonzero(showers[i-2:i+3, j-2:j+3])
                            nclose += np.count_nonzero(tracks[i-2:i+3, j-2:j+3])
                            if nclose == 0:
                                npix = shower_pixels + track_pixels
                                if npix > 6:
                                    cnt_void += 1
                                    sel_empty += 1
                                else: continue # completely empty patch
                            else: continue # too close to trk/shower
                        else: continue # not selected randomly

                    if is_near_nu:
                        sel_near_nu += 1

                    if np.count_nonzero(target) == 0:
                        print ('NEED A LABEL IN THE TARGET!!!')
                        continue

#                    if cnt_ind < max_capacity:
#                        #db[cnt_ind] = get_patch(raw, i, j, PATCH_SIZE_W, PATCH_SIZE_D)
#                        #db_y[cnt_ind] = target
#                        #cnt_ind += 1
#
#                        patch = get_patch(raw, i, j, PATCH_SIZE_W, PATCH_SIZE_D)
#                        # ===== FIX : Remove NaNs in patch =====
#                        if np.isnan(patch).any() or np.isinf(patch).any():
#                            # Skip corrupted patches
#                            continue
#
#                        # ===== FIX : Safe normalization =====
#                        max_val = np.max(patch)
#                        if max_val > 0:
#                            patch = patch / max_val
#                        else:
#                            # skip empty patch
#                            continue
#
#                        # Final safety (paranoid but safe)
#                        patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
#
#                        #db[cnt_ind] = patch
#                        #db_y[cnt_ind] = target
#                        #cnt_ind += 1
#
#                        db.append(patch)
#                        db_y.append(target)
#                        #print ('hello!')
###################################################################################################
                   # if len(db) < max_capacity:
                   #     patch = get_patch(raw, i, j, PATCH_SIZE_W, PATCH_SIZE_D)

                   #     patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

                   #     scale = np.max(np.abs(patch))
                   #     if scale > 0:
                   #         patch = patch / scale

                   #     db.append(patch.astype(np.float32))
                   #     db_y.append(target.astype(np.int32))

                   # else:
                   #     print ('MAX CAPACITY REACHED!!!')
                   #     break
###################################################################################################
                   if michel_focused:

                       is_signal_patch = (
                            target[2] == 1
                       )
                       
                       is_hard_background = (
                            target[2] == 0
                            and has_muon_context
                            and (track_pixels + shower_pixels > 6)
                       )

                       if is_signal_patch:
                            pass

                       elif is_hard_background:

                            if np.random.randint(10000) > int(100 * background_keep_fraction):
                                  continue

                       else:
                            continue

                       # =====================================================
                       # EVENT PATCH LIMIT
                       # =====================================================

                       if event_patch_count >= max_patches_per_event:
                            continue

                       # =====================================================
                       # BUILD PATCH
                       # =====================================================

                       patch = get_patch(
                             raw,
                             i,
                             j,
                             PATCH_SIZE_W,
                             PATCH_SIZE_D
                       )

                       # =====================================================
                       # REMOVE NaNs / infs
                       # =====================================================

                       patch = np.nan_to_num(
                             patch,
                             nan=0.0,
                             posinf=0.0,
                             neginf=0.0
                       )

                       # =====================================================
                       # SAFE NORMALIZATION
                       # =====================================================

                       scale = np.max(np.abs(patch))

                       if scale > 0:
                            patch = patch / scale
                       else:
                            continue

                       # =====================================================
                       # APPEND TO DATASET
                       # =====================================================

                       db.append(
                            patch.astype(np.float32)
                       )

                       db_y.append(
                                   target.astype(np.int32)
                       )
                       event_patch_count += 1
###################################################################################################
            print ('Selected: tracks', sel_trk, 'showers', sel_sh, 'empty', sel_empty, '/// muons', sel_muon, 'michel', sel_michel, 'clean trk', sel_clean_trk, 'near nu', sel_near_nu)

    # print ('Added', cnt_ind, 'tracks:', cnt_trk, 'showers:', cnt_sh, 'michels:', cnt_michel, 'empty:', cnt_void)
    print('Added', len(db), 'patches | tracks:', cnt_trk, 'showers:', cnt_sh, 'michels:', cnt_michel, 'empty:', cnt_void)

    #np.save('/exp/dune/app/users/kadhikar/CNNimage'+'/db_view_'+str(selected_view_idx)+'_x', db[:cnt_ind])
    #np.save('/exp/dune/app/users/kadhikar/CNNimage'+'/db_view_'+str(selected_view_idx)+'_y', db_y[:cnt_ind])

    # ===== FINAL CLEANUP =====
    #X_final = db[:cnt_ind]
    #Y_final = db_y[:cnt_ind]

    # Absolute safety
    #X_final = np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)

    ##np.save('/exp/dune/app/users/kadhikar/CNNimage'+'/db_view_'+str(selected_view_idx)+'_x_new', X_final)
    ##np.save('/exp/dune/app/users/kadhikar/CNNimage'+'/db_view_'+str(selected_view_idx)+'_y_new', Y_final)
    X_final = np.stack(db).astype(np.float32)
    Y_final = np.stack(db_y).astype(np.int32)

    X_final = np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)

    print("Final dataset shape:", X_final.shape)
    print("Final label shape:", Y_final.shape)
    print("NaNs in X:", np.isnan(X_final).sum())
    print("Infs in X:", np.isinf(X_final).sum())

    assert X_final.shape[0] == Y_final.shape[0]
    assert X_final.shape[1:] == (PATCH_SIZE_W, PATCH_SIZE_D)
    assert np.isfinite(X_final).all()

    outdir = "/exp/dune/app/users/kadhikar/CNNimage"
    x_path = os.path.join(outdir, f"db_view_{selected_view_idx}_x.npy")
    y_path = os.path.join(outdir, f"db_view_{selected_view_idx}_y.npy")

    tmp_x = x_path + ".tmp.npy"
    tmp_y = y_path + ".tmp.npy"

    np.save(tmp_x, X_final)
    np.save(tmp_y, Y_final)

    os.replace(tmp_x, x_path)
    os.replace(tmp_y, y_path)

    print("Saved:", x_path)
    print("Saved:", y_path)

    # VERIFY FILE INTEGRITY
    #X_test = np.load('/exp/dune/app/users/kadhikar/CNNimage'+'/db_view_'+str(selected_view_idx)+'_x_new')
    #Y_test = np.load('/exp/dune/app/users/kadhikar/CNNimage'+'/db_view_'+str(selected_view_idx)+'_y_new')

    #print("VERIFY SHAPE:", X_test.shape, Y_test.shape)
    #print("VERIFY NaNs:", np.isnan(X_test).sum())
if __name__ == "__main__":
    main(argv)

