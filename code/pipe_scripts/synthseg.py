import argparse
import os
import nibabel as nib
import subprocess
import numpy as np

def load_mri(mri_path):
    print("Load MRI.")

    mri_d = nib.load(mri_path)
    mri = mri_d.get_fdata()

    return mri, mri_d

def seg(mri, output_path, threads, gpu):
    
    try:
        print(f"Esecution of mri_synthseg on {mri}") 
        if gpu == "--cpu":
            subprocess.run(['mri_synthseg', '--i', mri, '--o', output_path, '--threads', str(threads), '--robust', "--cpu"],
                    check=True)
        else:
            subprocess.run(['mri_synthseg', '--i', mri, '--o', output_path, '--threads', str(threads), '--robust'],
                    check=True)

    except subprocess.CalledProcessError as e:
        print(f"Error during command execution: {e}")
        exit()

def get_grayMatter(mri):
    print("Extraction Gray Matter")
    mri_GM = np.where((mri == 3) | (mri == 10)| (mri == 11) | (mri == 12) | (mri == 13)| (mri == 17) | (mri == 18) |
                    (mri == 42) | (mri == 49)| (mri == 50) | (mri == 51) | (mri == 52)| (mri == 53) | (mri == 54) |
                    (mri == 8) | (mri == 26)| (mri == 28) | (mri == 47) | (mri == 58) | (mri == 60)
                    , 1, 0)
    
    return mri_GM

def get_whiteMatter(mri):
    print("Extraction White Matter")
    mri_WM = np.where((mri == 2) | (mri == 7) | (mri == 16) | (mri == 41) | (mri == 46), 1, 0)

    return mri_WM

def get_csf(mri):
    print("Extraction CSF")
    mri_CSF = np.where((mri == 24) | (mri == 4)| (mri == 5) | (mri == 14) | (mri == 15)| (mri == 43) | (mri == 44), 1, 0)

    return mri_CSF

def get_union(csf, gm, wm):
    print("Union masks")

    gm = np.where((gm == 1), 2, 0)
    wm = np.where((wm == 1), 3, 0)

    seg = csf + gm + wm

    return seg

def save(mri, d, n, output_path, mri_name):
    mri_ni = nib.Nifti1Image(mri, d.affine, d.header)
    name = mri_name.replace(".nii.gz", f"_{n}.nii.gz")
    nib.save(mri_ni, os.path.join(output_path, name))
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help="The path to MRI.")
    parser.add_argument('--threads', type=int, default=os.cpu_count(), help="The numbers of utilize threads. Default 1, if you want to use all the threads of your system, set it to 0.")
    parser.add_argument('--output_path', type=str, default="./", help="The path where the segmetation will be saved, if don't work give the absolute path.")
    parser.add_argument('--s', type=int, default="3", help="The number of tissues to be segmented. (default 3, it can be 2 or 3)")
    parser.add_argument('--gpu', type=int, required=True, help="Use GPU")

   
    args = parser.parse_args()
    mri = args.input
    output_path = args.output_path
    threads = args.threads
    s = args.s
    gpu = args.gpu

    assert os.path.exists(mri), 'file path doesn\'t exist'
    assert os.path.exists(output_path), 'file path doesn\'t exist'
    assert 0 <= threads <= os.cpu_count(), 'The number of threads must be between 0 and max thread of your system'
    assert s in (2,3),  "The number must be 2 or 3"
    assert gpu in (0,1), "The gpu argument must be 0 or 1"
    
    if threads == 0:
        threads = os.cpu_count()

    if gpu == 0:
        print("GPU not used")
        gpu = "--cpu"
    elif gpu == 1:
        print("GPU used")
        gpu = ""
    else:
        Warning("Not correct gpu argument, GPU not used")
        gpu = "--cpu"
        
    
    print("Segmentation phase")
    seg_path_out = os.path.join(output_path, os.path.basename(mri)).replace(".nii.gz", "_seg_tot.nii.gz")
    seg(mri, seg_path_out, threads, gpu)
    mri_seg, mri_d = load_mri(seg_path_out)
    pve_0 = get_csf(mri_seg)
    pve_1 = get_grayMatter(mri_seg)
    pve_2 = get_whiteMatter(mri_seg)
    pve_seg = get_union(pve_0, pve_1, pve_2)

    print("Save phase")
    save(pve_0, mri_d, "pve_0", output_path, os.path.basename(mri))
    save(pve_1, mri_d, "pve_1", output_path, os.path.basename(mri))
    save(pve_2, mri_d, "pve_2", output_path, os.path.basename(mri))
    save(pve_seg, mri_d, "pveseg", output_path, os.path.basename(mri))
    save(pve_seg, mri_d, "seg", output_path, os.path.basename(mri))
    