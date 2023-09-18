import numpy as np
from constants import CUBIC_TC, specific_heat, susceptibility
from model import Model
import time

# EXP_NAME = "cconv_periodic"
# EXP_DIR = "__cooling_exps/cconv_periodic/"
EXP_NAME = "cconv_periodic"
EXP_DIR = "__cooling_exps/cconv_periodic/"

CRIT_TEMP = CUBIC_TC
STEPS = 1000
MAX_STEPS = 10000
HIS_SAVE_FREQ = 2000

def explore_conv(model,
             t,
             ns_per_temp,
             max_steps,
             exp_dir,
             exp_name,
            his_save_freq):
    avg_ene_arr, avg_mag_arr, sh_arr, sus_arr =  [], [], [], []
    file_prefix = exp_dir + exp_name+"_"
    his_prefix = file_prefix+"his_"
    steps = 0
    en_his, mag_his = [], []
    his_counts = 0

    while steps <= max_steps:
        start_time = time.time_ns()
        am, en, mag = model.sweep(ns_per_temp, ns_per_temp*2)
        # Save the model
        steps += ns_per_temp

        en_his += en
        mag_his += mag
        ac_rate = np.round(am / (
                    ns_per_temp * m.lattice.num_sites) * 100, 2)

        spheat = specific_heat(en, t, m.size())
        sus = susceptibility(mag, t, m.size())
        avg_ene_arr.append(np.average(en))
        sh_arr.append(spheat)
        avg_mag_arr.append(np.average(mag))
        sus_arr.append(sus)

        # Time the section
        end_time = time.time_ns()
        time_used = np.round((end_time - start_time) * 1e-9,
                             3)
        print(
            f"specific heat; susceptibility; ac_rate; time:" +
            f" {np.round(spheat, 5)}; {np.round(sus, 5)}; {ac_rate}%; {time_used} per {ns_per_temp} sweeps")

        model.save(fileprefix=file_prefix, time=False, sweep_num=steps)
        if steps % his_save_freq == 0:
            his = np.round([en_his, mag_his], 5).T
            np.savetxt(his_prefix + f"#{his_counts}_", his, delimiter=',')
            en_his, mag_his = [], []
            his_counts += 1


    res = np.round(np.vstack((avg_ene_arr, sh_arr,
                              avg_mag_arr, sus_arr)).T, 5)
    print(res)
    np.savetxt(file_prefix, res, delimiter=',')

if __name__ == "__main__":
    m = Model(shape=(16, 16, 16),
              lattice_structure="cubic",
              temperature=CRIT_TEMP,
              periodic=(0,1,2),
              build=True)
    explore_conv(m, CRIT_TEMP, STEPS, MAX_STEPS,
             EXP_DIR, EXP_NAME, HIS_SAVE_FREQ)
    # explore_conv(m, CRIT_TEMP, 2, 20,
    #          EXP_DIR, EXP_NAME, 10)