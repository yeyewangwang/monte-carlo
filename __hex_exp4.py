import numpy as np
from constants import CUBIC_TC, specific_heat, susceptibility
from model import Model
import time

# EXP_NAME = "cooling1"
# EXP_DIR = "__cooling_exps/"
EXP_NAME = "hex5"
EXP_DIR = "__hex_exps/hex5_long/"

TEMP_RANGE = np.arange(3.5, 7, 0.1)
# print("temp range:", TEMP_RANGE)
# Number of steps per temperature level used to
# achieve convergence
NS_PER_TEMP = 5000
# Number of steps per temperature level used to
# measure specific heat and susceiptibility.
MNS_PER_TEMP = 5000


def run(model,
        temp_range,
        ns_per_temp,
        mns_per_temp,
        exp_dir,
        exp_name):

    ts, avg_ene_arr, avg_mag_arr, sh_arr, sus_arr = [], [], [], [], []
    file_prefix = exp_dir + exp_name+"_"
    for t in temp_range:
        start_time = time.time_ns()

        model.raise_to(temp=t)
        model.init_simulation_state()

        am, en, mag = model.sweep(ns_per_temp, 2000)
        ac_rate = np.round(am / (ns_per_temp * m.lattice.num_sites) * 100, 2)

        # print(f"t={t}, ac_rate={ac_rate}% convergence round")
        spheat = specific_heat(en, t, m.size())
        sus = susceptibility(mag, t, m.size())
        # avg_ene_arr.append(np.average(en))
        # avg_mag_arr.append(np.average(mag))
        # ts.append(t)
        # sh_arr.append(spheat)
        # sus_arr.append(sus)
        # print(f"specific heat; susceptibility: {np.round(spheat, 5)}; {np.round(sus, 5)}")

        steps = ns_per_temp
        model.save(fileprefix=file_prefix, time=False, sweep_num=steps)
        time_used = np.round((time.time_ns() - start_time) * 1e-9, 3)
        print(
            f"init:temp;ac_rate;sheat;sus;time: {t} ;{ac_rate} ;" +
            f"{np.round(spheat, 5)} ;{np.round(sus, 5)} ;{time_used}")

        am, en, mag = model.sweep(mns_per_temp, 2000)
        ac_rate = np.round(am / (
                    mns_per_temp * m.lattice.num_sites) * 100, 2)

        # print(f"t={t}, ac_rate={ac_rate}% stable round")
        spheat = specific_heat(en, t, m.size())
        sus = susceptibility(mag, t, m.size())
        ts.append(t)
        avg_ene_arr.append(np.average(en))
        sh_arr.append(spheat)
        avg_mag_arr.append(np.average(mag))
        sus_arr.append(sus)

        # print(f"specific heat; susceptibility: {np.round(spheat, 5)}; {np.round(sus, 5)}")

        steps += mns_per_temp
        model.save(fileprefix=file_prefix, time=False, sweep_num=steps)

        end_time = time.time_ns()
        time_used = np.round((end_time - start_time) * 1e-9, 3)

        print(f"meas:temp;ac_rate;sheat;sus;time: {t} ;{ac_rate} ;"+
              f"{np.round(spheat, 5)} ;{np.round(sus, 5)} ;{time_used}")
    res = np.round(np.vstack((temp_range, avg_ene_arr, sh_arr, avg_mag_arr, sus_arr)).T, 5)
    print(res)
    np.savetxt(file_prefix +"data", res, delimiter=',')

if __name__ == "__main__":
    m = Model(shape=(100, 100),
              lattice_structure="hexagon",
              temperature=0,
              build=True)
    run(m, TEMP_RANGE, NS_PER_TEMP, MNS_PER_TEMP,
             EXP_DIR, EXP_NAME)
    # run(m, [10, 5, 2], 2, 2,
    #          EXP_DIR, EXP_NAME)