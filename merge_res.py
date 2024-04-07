# -*- coding: utf-8 -*-
"""Merge files with different resolutions
"""
import h5py
import tools


SPEC_PREF = "eigenspec_res"

def merge_resolutions(target_fname: str, *merge_fnames, verbose: int = 0):
    """Append resolutions in merge_fnames to target_fname
    """
    with h5py.File(target_fname, 'a') as target_f:
        
        n_res = 0
        for obj_name in target_f.keys():
            if obj_name[:len(SPEC_PREF)] == SPEC_PREF:
                n_res += 1
        
        for fname in merge_fnames:
            if verbose > 0:
                tools.print_heading("Merging {} to {}".format(fname, target_fname), prefix='\n', suffix='\n', lines="over")
            with h5py.File(fname, 'r') as merge_f:
                for obj_name in merge_f.keys():
                    if obj_name[:len(SPEC_PREF)] == SPEC_PREF:
                        write_name = f"{SPEC_PREF}{n_res}"
                        merge_f.copy(merge_f[obj_name], target_f, name=write_name)
                        if verbose > 1:
                            print("\t{}:/{} copied to {}:/{}".format(fname, obj_name, target_fname, write_name))
                        n_res += 1
            if verbose > 0:
                tools.print_heading("{} merged to {}".format(fname, target_fname), prefix='\n', lines="under")


if __name__ == "__main__":
    merge_resolutions(
        # S1pi - E_eta = 1e-6
        "./runs/S1pi_1e-6_m1_galerkin_30-40_parity-oppo/eigenspectra_all.h5",
        "./runs/S1pi_1e-6_m1_galerkin_30-40_parity-oppo/eigenspectra_backup.h5",
        "./runs/S1pi_1e-6_m1_galerkin_42_parity-oppo/eigenspectra.h5",
        "./runs/S1pi_1e-6_m1_galerkin_60_parity-oppo/eigenspectra.h5",
        verbose=2
    )
