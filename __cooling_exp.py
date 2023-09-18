import numpy as np
from constants import CUBIC_TC, specific_heat, susceptibility
from model import Model
import time

# EXP_NAME = "cooling1"
# EXP_DIR = "__cooling_exps/"
EXP_NAME = "c1_perio"
EXP_DIR = "__cooling_exps/c1_perio/"

CRIT_TEMP = CUBIC_TC
DISORDER_RANGE_2 = np.arange(0.1 + CRIT_TEMP, 0.501 + CRIT_TEMP, 0.05)
DISORDER_RANGE_1 = np.arange(0.05 + CRIT_TEMP, 0.1 + CRIT_TEMP, 0.01)
CRIT_RANGE = np.arange(-0.05 + CRIT_TEMP, 0.05 + CRIT_TEMP, 0.005)
ORDER_RANGE_1 = np.arange(-0.1 + CRIT_TEMP, -0.05 + CRIT_TEMP, 0.01)
ORDER_RANGE_2 = np.arange(-0.5 + CRIT_TEMP, -0.11 + CRIT_TEMP, 0.05)
ORDER_RANGE_3 = np.arange(-1.5 + CRIT_TEMP, -0.51 + CRIT_TEMP, 0.1)
TEMP_RANGE = -np.sort(-np.hstack((DISORDER_RANGE_2, DISORDER_RANGE_1,
                        CRIT_RANGE, ORDER_RANGE_1, ORDER_RANGE_2, ORDER_RANGE_3)))

# print("temp range:", TEMP_RANGE)
# Number of steps per temperature level used to
# achieve convergence
NS_PER_TEMP = 1000
# Number of steps per temperature level used to
# measure specific heat and susceiptibility.
MNS_PER_TEMP = 1000


def schedule(model,
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
    m = Model(shape=(16, 16, 16),
              lattice_structure="cubic",
              periodic=(0, 1, 2),
              temperature=0,
              build=True)
    schedule(m, TEMP_RANGE, NS_PER_TEMP, MNS_PER_TEMP,
             EXP_DIR, EXP_NAME)
    # schedule(m, [10], 2, 2,
    #          EXP_DIR, EXP_NAME)