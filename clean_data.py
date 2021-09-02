# {{{ Library imports
from hmi_clean import HmiClass
from globalvars import DopplerVars
import argparse
import glob
import time
import os
# }}} imports

# {{{ argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--gnup', help="Argument for gnuParallel",
                    default=1, type=int)
ARGS = parser.parse_args()

GVAR = DopplerVars(ARGS.gnup)
# }}} argument parser


if __name__ == "__main__":
    # {{{ directories
    hmi_data_dir = GVAR.get_dir("hmidata")
    hmi_files = glob.glob(f"{hmi_data_dir}/*.fits")
    print("Program started -- reading files")
    # }}} dirs

    # 1. Reading HMI Dopplergrams
    t1 = time.time()

    total_days = len(hmi_files)
    print(f"Total days = {total_days}")

    # 2. Creating SunPy maps
    for day in daylist:
        print("### 1. Initializing")
        dop_img = HmiClass(hmi_data_dir, hmi_files[day], day)
        print("### 2. Getting satellite velocity")
        dop_img.get_sat_vel()
        print("### 3. Removing effect of satellite velocity")
        dop_img.remove_sat_vel()
        print("### 4. Removing gravitational redshift")
        dop_img.remove_grav_redshift()
        print("### 5. Removing large scale features")
        dop_img.remove_large_features()
        print("### 6. Saving processed data")

        dop_img.save_theta_phi()
        dop_img.save_theta_phi_DC()
        dop_img.save_map_data()
    t2 = time.time()
    print(f"Total time taken = {(t2-t1)/60:5.2f} minutes")
