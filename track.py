#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FOD tractography -- RAS space, target streamline count mode (-select style).

Similar to MRtrix tckgen -select N:
  - Set target streamline count TARGET_STREAMLINES
  - Start from density=1, double seed density each round
  - Stop immediately once target streamline count is reached
  - Print progress between rounds for efficiency estimation
  - Support reading pre-computed directions from peaks.nii.gz (preferred)

Usage:
    python fod_tractography.py \\
        --fodf fod.nii.gz \\
        --fa fa.nii.gz \\
        --mask mask.nii.gz \\
        --seed-mask seed_mask.nii.gz \\
        --output output.tck \\
        [--peaks peaks.nii.gz] \\
        [--asi asi_map.nii.gz] \\
        [--delta delta.nii.gz] \\
        [--target 50000] \\
        [--max-density 16] \\
        [--probabilistic] \\
        [--use-fa-mod] \\
        [--use-asi-mod] \\
        [--use-delta-mod]
"""

import numpy as np
import nibabel as nib
from dipy.tracking import utils
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from dipy.data import default_sphere
from dipy.reconst.shm import sh_to_sf
from multiprocessing import Pool, cpu_count
import os, time, argparse, warnings
warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="FOD-based tractography with target streamline count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (using FOD)
  python fod_tractography.py \\
      --fodf fod.nii.gz --fa fa.nii.gz --mask mask.nii.gz \\
      --seed-mask mask.nii.gz --output output.tck

  # Using pre-computed peaks directions
  python fod_tractography.py \\
      --fodf fod.nii.gz --fa fa.nii.gz --mask mask.nii.gz \\
      --seed-mask mask.nii.gz --peaks peaks.nii.gz --output output.tck

  # Using advanced modulation
  python fod_tractography.py \\
      --fodf fod.nii.gz --fa fa.nii.gz --mask mask.nii.gz \\
      --seed-mask mask.nii.gz --asi asi_map.nii.gz --delta delta.nii.gz \\
      --use-fa-mod --use-asi-mod --use-delta-mod --target 100000 \\
      --output output.tck
        """
    )
    
    # Required arguments
    parser.add_argument('--fodf', required=True, 
                        help='Path to FOD file (nifti)')
    parser.add_argument('--fa', required=True, 
                        help='Path to FA file (nifti)')
    parser.add_argument('--mask', required=True, 
                        help='Path to brain mask file (nifti)')
    parser.add_argument('--seed-mask', required=True, 
                        help='Path to seed mask file (nifti)')
    parser.add_argument('--output', required=True, 
                        help='Output .tck file path')
    
    # Optional arguments - additional input files
    parser.add_argument('--peaks', default=None, 
                        help='Path to pre-computed peaks directions file (nifti, optional)')
    parser.add_argument('--asi', default=None, 
                        help='Path to ASI map file (nifti, optional)')
    parser.add_argument('--delta', default=None, 
                        help='Path to delta map file (nifti, optional)')
    
    # Optional arguments - tracking settings
    parser.add_argument('--target', type=int, default=50000, 
                        help='Target number of streamlines (default: 50000)')
    parser.add_argument('--max-density', type=int, default=16, 
                        help='Maximum seed density (default: 16)')
    parser.add_argument('--step-size', type=float, default=1.0, 
                        help='Step size in mm (default: 1.0)')
    parser.add_argument('--min-length', type=float, default=20.0, 
                        help='Minimum streamline length in mm (default: 20.0)')
    parser.add_argument('--max-length', type=float, default=200.0, 
                        help='Maximum streamline length in mm (default: 200.0)')
    parser.add_argument('--max-angle', type=float, default=70.0, 
                        help='Maximum turning angle in degrees (default: 70.0)')
    parser.add_argument('--fa-stop', type=float, default=0.05, 
                        help='FA stopping threshold (default: 0.05)')
    parser.add_argument('--delta-stop', type=float, default=0.05, 
                        help='Delta stopping threshold (default: 0.05)')
    
    # Optional arguments - modulation settings
    parser.add_argument('--probabilistic', action='store_true', default=False,
                        help='Use probabilistic tracking (default: deterministic)')
    parser.add_argument('--use-peaks-first', action='store_true', default=False,
                        help='Prioritize using peaks.nii.gz directions')
    parser.add_argument('--use-fa-mod', action='store_true', default=False,
                        help='Enable FA-modulated turning angle')
    parser.add_argument('--use-asi-mod', action='store_true', default=False,
                        help='Enable ASI modulation')
    parser.add_argument('--use-delta-mod', action='store_true', default=False,
                        help='Enable delta stopping condition')
    
    # FA modulation parameters
    parser.add_argument('--fa-high', type=float, default=0.7, 
                        help='FA high threshold (default: 0.7)')
    parser.add_argument('--fa-low', type=float, default=0.2, 
                        help='FA low threshold (default: 0.2)')
    parser.add_argument('--max-angle-low-fa', type=float, default=40.0, 
                        help='Maximum turning angle in low FA regions in degrees (default: 40.0)')
    
    # ASI parameters
    parser.add_argument('--asi-weight-scale', type=float, default=2.0, 
                        help='ASI weight scaling factor (default: 2.0)')
    
    # Performance parameters
    parser.add_argument('--num-processes', type=int, default=max(1, cpu_count()-1),
                        help=f'Number of parallel processes (default: {max(1, cpu_count()-1)})')
    parser.add_argument('--chunk-size', type=int, default=200,
                        help='Number of seeds per batch (default: 200)')
    
    return parser.parse_args()


def load_peaks(peaks_path):
    """Load peaks direction data"""
    if not peaks_path or not os.path.exists(peaks_path):
        return False, None
    
    try:
        peaks_img = nib.load(peaks_path)
        peaks_data = peaks_img.get_fdata()
        X, Y, Z, C = peaks_data.shape
        
        if C == 3:
            peaks_dir_cache = peaks_data.reshape(X, Y, Z, 3)
            print(f"[OK] Loaded peaks.nii.gz (single direction): {peaks_data.shape}")
            return True, peaks_dir_cache
        elif C == 9:
            peaks_data_reshaped = peaks_data.reshape(X, Y, Z, 3, 3)
            peaks_dir_cache = peaks_data_reshaped[..., 0, :]
            print(f"[OK] Loaded peaks.nii.gz (dual direction, using first): {peaks_data.shape}")
            return True, peaks_dir_cache
        else:
            print(f"[WARNING] peaks.nii.gz last dimension is {C}, not supported, skipping")
            return False, None
    except Exception as e:
        print(f"[WARNING] Failed to load peaks.nii.gz: {e}")
        return False, None


# =========================================================
# Helper functions
# =========================================================
def ras_to_vox(pos_ras, affine_inv):
    p = np.array([pos_ras[0], pos_ras[1], pos_ras[2], 1.0])
    return (affine_inv @ p)[:3]


def get_peak_direction_at_voxel(i, j, k, shape, peaks_dir_cache):
    """Get direction from peaks cache"""
    if peaks_dir_cache is not None and 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
        dir_vec = peaks_dir_cache[i, j, k]
        if np.linalg.norm(dir_vec) > 0.01:
            return dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
    return None


def get_odf_at_voxel(i, j, k, odf_cache):
    """Get ODF from cache (fallback)"""
    return odf_cache.get((i, j, k), None)


def get_context_at_ras(pos_ras, shape, mask_data, fa_data, delta_data, asi_data,
                      peaks_available, peaks_dir_cache, odf_cache,
                      USE_DELTA_MOD, USE_ASI_MOD, FA_STOP, DELTA_STOP):
    vox = ras_to_vox(pos_ras, affine_inv)
    i, j, k = np.round(vox).astype(int)
    
    if not (0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]):
        return None
    if not mask_data[i, j, k]:
        return None
    
    fa = fa_data[i, j, k]
    if fa < FA_STOP:
        return None
    
    if USE_DELTA_MOD and delta_data is not None and delta_data[i, j, k] < DELTA_STOP:
        return None
    
    asi = asi_data[i, j, k] if USE_ASI_MOD and asi_data is not None else 0.0
    
    # Prioritize peaks directions
    if peaks_available:
        peak_dir = get_peak_direction_at_voxel(i, j, k, shape, peaks_dir_cache)
        if peak_dir is not None:
            return peak_dir, fa, asi, 'peak'
    
    # Fallback: use ODF
    odf = get_odf_at_voxel(i, j, k, odf_cache)
    if odf is not None:
        return odf, fa, asi, 'odf'
    
    return None


def compute_dynamic_angle(fa, USE_FA_MOD, max_angle, FA_LOW, FA_HIGH, MAX_ANGLE_LOW_FA):
    if not USE_FA_MOD:
        return np.deg2rad(max_angle)
    t = np.clip((fa - FA_LOW) / (FA_HIGH - FA_LOW + 1e-8), 0.0, 1.0)
    return np.deg2rad(MAX_ANGLE_LOW_FA + t * (max_angle - MAX_ANGLE_LOW_FA))


def apply_asi_boost(odf, asi, sphere, USE_ASI_MOD, ASI_WEIGHT_SCALE):
    if not USE_ASI_MOD or asi < 0.01:
        return odf
    peak_idx = np.argmax(odf)
    dots = np.clip(np.abs(np.dot(sphere.vertices, sphere.vertices[peak_idx])), 0, 1)
    return odf * (1.0 + ASI_WEIGHT_SCALE * asi * dots)


def get_next_direction_from_odf(odf, fa, asi, prev_dir, sphere, USE_FA_MOD, 
                                 max_angle, FA_LOW, FA_HIGH, MAX_ANGLE_LOW_FA,
                                 USE_ASI_MOD, ASI_WEIGHT_SCALE, PROBABILISTIC):
    """Get next direction from ODF (fallback)"""
    if odf is None or np.sum(odf) < 1e-8:
        return None
    
    odf_w = apply_asi_boost(odf.copy(), asi, sphere, USE_ASI_MOD, ASI_WEIGHT_SCALE)
    max_angle_rad = compute_dynamic_angle(fa, USE_FA_MOD, max_angle, FA_LOW, FA_HIGH, MAX_ANGLE_LOW_FA)
    
    if prev_dir is not None:
        angles = np.arccos(np.clip(np.abs(np.dot(sphere.vertices, prev_dir)), 0, 1))
        odf_w[angles > max_angle_rad] = 0
        if np.sum(odf_w) < 1e-8:
            return None
    
    odf_w = np.maximum(odf_w, 0)
    total = odf_w.sum()
    if total < 1e-8:
        return None
    
    if PROBABILISTIC:
        probs = odf_w / total
        if np.any(np.isnan(probs)):
            return None
        idx = np.random.choice(len(sphere.vertices), p=probs)
    else:
        idx = np.argmax(odf_w)
    
    new_dir = sphere.vertices[idx].copy()
    if prev_dir is not None and np.dot(new_dir, prev_dir) < 0:
        new_dir = -new_dir
    return new_dir / (np.linalg.norm(new_dir) + 1e-12)


def get_next_direction_from_peak(peak_dir, fa, prev_dir, USE_FA_MOD, 
                                  max_angle, FA_LOW, FA_HIGH, MAX_ANGLE_LOW_FA):
    """Get next direction from peak direction (preferred)"""
    if peak_dir is None:
        return None
    
    new_dir = peak_dir.copy()
    max_angle_rad = compute_dynamic_angle(fa, USE_FA_MOD, max_angle, FA_LOW, FA_HIGH, MAX_ANGLE_LOW_FA)
    
    if prev_dir is not None:
        dot = np.dot(new_dir, prev_dir)
        angle = np.arccos(np.clip(dot, -1, 1))
        if angle > max_angle_rad:
            new_dir = -new_dir
            dot = np.dot(new_dir, prev_dir)
            angle = np.arccos(np.clip(dot, -1, 1))
            if angle > max_angle_rad:
                return None
    
    if prev_dir is not None and np.dot(new_dir, prev_dir) < 0:
        new_dir = -new_dir
    
    return new_dir / (np.linalg.norm(new_dir) + 1e-12)


def track_one_direction(start_pos, start_dir, step_size, max_length, 
                       use_peak_mode, track_params):
    pos, direction = start_pos.copy(), start_dir.copy()
    pts = [pos.copy()]
    max_steps = int(max_length / step_size)
    
    for _ in range(max_steps):
        ctx = get_context_at_ras(pos, **track_params)
        if ctx is None:
            break
        
        if use_peak_mode:
            peak_dir, fa, asi, mode = ctx
            new_dir = get_next_direction_from_peak(peak_dir, fa, direction, 
                                                   track_params['USE_FA_MOD'],
                                                   track_params['max_angle'],
                                                   track_params['FA_LOW'],
                                                   track_params['FA_HIGH'],
                                                   track_params['MAX_ANGLE_LOW_FA'])
        else:
            odf, fa, asi, mode = ctx
            new_dir = get_next_direction_from_odf(odf, fa, asi, direction,
                                                  track_params['sphere'],
                                                  track_params['USE_FA_MOD'],
                                                  track_params['max_angle'],
                                                  track_params['FA_LOW'],
                                                  track_params['FA_HIGH'],
                                                  track_params['MAX_ANGLE_LOW_FA'],
                                                  track_params['USE_ASI_MOD'],
                                                  track_params['ASI_WEIGHT_SCALE'],
                                                  track_params['PROBABILISTIC'])
        
        if new_dir is None:
            break
        direction = new_dir
        pos = pos + step_size * direction
        pts.append(pos.copy())
    
    return pts


def track_streamline(seed_ras, track_params):
    ctx = get_context_at_ras(seed_ras, **track_params)
    if ctx is None:
        return None
    
    if len(ctx) == 4 and ctx[3] == 'peak':
        peak_dir, fa, asi, _ = ctx
        init_dir = peak_dir.copy()
        init_dir /= (np.linalg.norm(init_dir) + 1e-12)
        use_peak_mode = True
    else:
        odf, fa, asi, _ = ctx
        if odf is None:
            return None
        odf_mod = apply_asi_boost(odf.copy(), asi, track_params['sphere'],
                                  track_params['USE_ASI_MOD'],
                                  track_params['ASI_WEIGHT_SCALE'])
        init_dir = track_params['sphere'].vertices[np.argmax(odf_mod)].copy()
        init_dir /= (np.linalg.norm(init_dir) + 1e-12)
        use_peak_mode = False
    
    step_size = track_params['step_size']
    max_length = track_params['max_length']
    min_length = track_params['min_length']
    
    fwd = track_one_direction(seed_ras, init_dir, step_size, max_length, 
                             use_peak_mode, track_params)
    bwd = track_one_direction(seed_ras, -init_dir, step_size, max_length, 
                             use_peak_mode, track_params)
    
    streamline = np.array(bwd[::-1][:-1] + fwd) if len(bwd) > 1 else np.array(fwd)
    if len(streamline) < 2:
        return None
    
    total_len = np.sum(np.linalg.norm(np.diff(streamline, axis=0), axis=1))
    if total_len < min_length or total_len > max_length:
        return None
    
    return streamline.astype(np.float32)


def process_seed(args):
    _, seed, track_params = args
    return track_streamline(seed, track_params)


# =========================================================
# Adaptive seeding loop
# =========================================================
def run_adaptive(target, max_density, seed_mask, affine, NUM_PROCESSES, CHUNK_SIZE, track_params):
    """
    Generate seeds round by round, doubling density each round,
    until cumulative streamline count reaches target.
    """
    all_streamlines = []
    used_seed_count = 0
    density = 1

    print(f"\nTarget streamlines: {target}")
    print(f"Maximum seed density: {max_density}")
    print(f"Direction source: {'Peaks.nii.gz (preferred)' if track_params['peaks_available'] else 'FOD (on-the-fly)'}")
    print("-" * 50)

    while len(all_streamlines) < target and density <= max_density:
        seeds_all = utils.seeds_from_mask(seed_mask, affine=affine, density=density)
        new_seeds = seeds_all[used_seed_count:]

        if len(new_seeds) == 0:
            print(f"density={density}: No new seeds, skipping")
            density *= 2
            continue

        t0 = time.time()
        needed = target - len(all_streamlines)
        print(f"density={density:2d} | New seeds this round: {len(new_seeds):6d} | "
              f"Still needed: {needed}")

        seed_args = [(i, seed, track_params) for i, seed in enumerate(new_seeds)]
        round_streamlines = []

        with Pool(processes=NUM_PROCESSES) as pool:
            for i, result in enumerate(
                pool.imap_unordered(process_seed, seed_args, chunksize=CHUNK_SIZE)
            ):
                if result is not None:
                    round_streamlines.append(result)
                    if len(all_streamlines) + len(round_streamlines) >= target:
                        pool.terminate()
                        break

                if (i + 1) % 2000 == 0:
                    elapsed = time.time() - t0
                    speed = (i + 1) / elapsed
                    print(f"  Processed {i+1}/{len(new_seeds)} | "
                          f"Round streamlines: {len(round_streamlines)} | "
                          f"{speed:.0f} seeds/sec")

        all_streamlines.extend(round_streamlines)
        used_seed_count = len(seeds_all)

        elapsed = time.time() - t0
        success_rate = len(round_streamlines) / max(len(new_seeds), 1) * 100
        print(f"  -> Round obtained: {len(round_streamlines)} streamlines | "
              f"Success rate: {success_rate:.1f}% | "
              f"Accumulated: {len(all_streamlines)}/{target} | "
              f"Time: {elapsed:.1f}s")

        if len(all_streamlines) >= target:
            break

        density *= 2

    result = all_streamlines[:target]

    if len(result) < target:
        print(f"\n[WARNING] Reached maximum density {max_density}, only obtained {len(result)} streamlines "
              f"(target: {target})")
        print(f"   Suggestions: increase --max-density or relax --min-length / --fa-stop")
    else:
        print(f"\n[OK] Reached target of {target} streamlines")

    return result


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    start_time = time.time()
    args = parse_arguments()
    
    # Set parameters from command line arguments
    TARGET_STREAMLINES = args.target
    MAX_SEED_DENSITY = args.max_density
    step_size = args.step_size
    min_length = args.min_length
    max_length = args.max_length
    max_angle = args.max_angle
    FA_STOP = args.fa_stop
    DELTA_STOP = args.delta_stop
    PROBABILISTIC = args.probabilistic
    USE_PEAKS_FIRST = args.use_peaks_first
    USE_FA_MOD = args.use_fa_mod
    USE_ASI_MOD = args.use_asi_mod
    USE_DELTA_MOD = args.use_delta_mod
    FA_HIGH = args.fa_high
    FA_LOW = args.fa_low
    MAX_ANGLE_LOW_FA = args.max_angle_low_fa
    ASI_WEIGHT_SCALE = args.asi_weight_scale
    NUM_PROCESSES = args.num_processes
    CHUNK_SIZE = args.chunk_size
    
    print("=" * 60)
    print("Loading data...")
    
    # Load required data
    fodf_img = nib.load(args.fodf)
    fodf_data = fodf_img.get_fdata()
    affine = fodf_img.affine
    affine_inv = np.linalg.inv(affine)
    shape = fodf_data.shape[:3]

    mask_data = nib.load(args.mask).get_fdata().astype(bool)
    seed_mask = nib.load(args.seed_mask).get_fdata().astype(bool)
    fa_data = nib.load(args.fa).get_fdata()
    
    asi_data = nib.load(args.asi).get_fdata() if args.asi and USE_ASI_MOD else None
    delta_data = nib.load(args.delta).get_fdata() if args.delta and USE_DELTA_MOD else None

    # Load peaks data
    peaks_available, peaks_dir_cache = load_peaks(args.peaks if USE_PEAKS_FIRST else None)
    
    # Prepare sphere and ODF
    sphere = default_sphere
    n_coeffs = fodf_data.shape[-1]
    sh_order = int((np.sqrt(1 + 8 * n_coeffs) - 3) / 2)
    print(f"SH order: {sh_order}")
    print(f"Voxel size: {np.abs(np.diag(affine)[:3])} mm")

    # Pre-compute ODF cache if peaks are not available
    odf_cache = {}
    if not peaks_available:
        print("Pre-computing ODF cache (fallback)...")
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if mask_data[i, j, k]:
                        sh = fodf_data[i, j, k, :]
                        if np.sum(np.abs(sh)) > 0.01:
                            odf = sh_to_sf(sh, sphere, sh_order_max=sh_order)
                            odf_cache[(i, j, k)] = np.maximum(odf, 0)
        print(f"Cached {len(odf_cache)} voxels")

    # Package tracking parameters
    track_params = {
        'shape': shape,
        'mask_data': mask_data,
        'fa_data': fa_data,
        'delta_data': delta_data,
        'asi_data': asi_data,
        'peaks_available': peaks_available,
        'peaks_dir_cache': peaks_dir_cache,
        'odf_cache': odf_cache,
        'affine_inv': affine_inv,
        'USE_DELTA_MOD': USE_DELTA_MOD,
        'USE_ASI_MOD': USE_ASI_MOD,
        'USE_FA_MOD': USE_FA_MOD,
        'FA_STOP': FA_STOP,
        'DELTA_STOP': DELTA_STOP,
        'max_angle': max_angle,
        'FA_LOW': FA_LOW,
        'FA_HIGH': FA_HIGH,
        'MAX_ANGLE_LOW_FA': MAX_ANGLE_LOW_FA,
        'ASI_WEIGHT_SCALE': ASI_WEIGHT_SCALE,
        'sphere': sphere,
        'PROBABILISTIC': PROBABILISTIC,
        'step_size': step_size,
        'min_length': min_length,
        'max_length': max_length,
    }

    print("=" * 60)
    print(f"Target streamlines:   {TARGET_STREAMLINES}")
    print(f"Maximum seed density: {MAX_SEED_DENSITY}")
    print(f"Step size:            {step_size} mm")
    print(f"Length range:         {min_length}-{max_length} mm")
    print(f"Base angle:           {max_angle} deg")
    print(f"Tracking mode:        {'Probabilistic' if PROBABILISTIC else 'Deterministic'}")
    print(f"Modulation switches:  FA={USE_FA_MOD}, ASI={USE_ASI_MOD}, delta={USE_DELTA_MOD}")
    print(f"Direction source:     {'Peaks.nii.gz (preferred)' if peaks_available else 'FOD (on-the-fly)'}")
    print(f"Output:               {args.output}")
    print(f"Number of processes:  {NUM_PROCESSES}")
    print("=" * 60)

    # Run tracking
    final_streamlines = run_adaptive(TARGET_STREAMLINES, MAX_SEED_DENSITY, 
                                    seed_mask, affine, NUM_PROCESSES, CHUNK_SIZE, 
                                    track_params)

    # Save results
    if final_streamlines:
        print(f"\nSaving {len(final_streamlines)} streamlines...")
        sft = StatefulTractogram(final_streamlines, fodf_img, Space.RASMM)
        sft.remove_invalid_streamlines()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        save_tractogram(sft, args.output)
        print(f"[OK] Saved to {args.output}")

        # Statistics
        lengths = [
            np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))
            for s in final_streamlines
        ]
        print(f"\nStatistics:")
        print(f"  Total streamlines:  {len(final_streamlines)}")
        print(f"  Mean length:        {np.mean(lengths):.1f} mm")
        print(f"  Length range:       {np.min(lengths):.1f} - {np.max(lengths):.1f} mm")
        
        # Direction verification
        if len(final_streamlines) >= 5:
            print("\nStreamline direction verification (first 5):")
            for idx, sl in enumerate(final_streamlines[:5]):
                if len(sl) >= 2:
                    start = sl[0]
                    end = sl[-1]
                    direction = end - start
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    print(f"  Streamline {idx}: X={direction[0]:.3f}, Y={direction[1]:.3f}, Z={direction[2]:.3f}")
    else:
        print("[ERROR] No streamlines generated")

    print(f"\nTotal time: {time.time()-start_time:.1f} seconds")
