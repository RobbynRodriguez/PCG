import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


for j in range(1, 4):
    patient_num = j
    peak_loc = np.load("Labeled CPSC-2020 data/RP" + str(patient_num) + ".npy").tolist()

    if sys.argv[1] == "ECG":
        y_m = -1.5
        y_p = 1.5
        p = 211
        gan_outputs = sio.loadmat("all_test_outputs_ECG/" + str(patient_num) + "/gan_outputs.mat")
        real_sig = sio.loadmat("all_test_outputs_ECG/" + str(patient_num) + "/real_sig.mat")
    elif sys.argv[1] == "PCG":
        p = 311
        y_m = -1.0
        y_p = 1.0
        gan_outputs = sio.loadmat("all_test_outputs_PCG/" + str(patient_num) + "/gan_outputs.mat")
        real_sig = sio.loadmat("all_test_outputs_PCG/" + str(patient_num) + "/real_sig.mat")
        clean_sig = sio.loadmat("all_test_outputs_PCG/" + str(patient_num) + "/S" + str(patient_num) + "_clean.mat")

    else:
        print("L'argument du programme doit être ECG ou PCG")
        sys.exit(1)

    gan_outputs = gan_outputs["gan_outputs"]
    real_sig = real_sig["real_sig"]
    ab_beats = sio.loadmat("Labeled CPSC-2020 data/R" + str(patient_num) + ".mat")
    S = ab_beats["S"]
    V = ab_beats["V"]

    S = pd.DataFrame(data=S)
    V = pd.DataFrame(data=V)

    win_size = int(np.floor(len(gan_outputs) / 4000))

    gan_outputs = gan_outputs[:win_size * 4000]

    real_sig = real_sig[:win_size * 4000]

    gan_outputs1 = gan_outputs.reshape(win_size, 4000)

    real_sig1 = real_sig.reshape(win_size, 4000)

    if sys.argv[1] == "PCG":
        clean_sig = clean_sig["PCG"]
        clean_sig = clean_sig[:win_size * 4000]
        clean_sig1 = clean_sig.reshape(win_size, 4000)
    # S_arr = np.zeros(((win_size * 4000), 1))
    # V_arr = np.zeros(((win_size * 4000), 1))
    # S_arr[S.values] = 1
    # V_arr[V.values] = 1
    # S_arr = S_arr.reshape(win_size, 4000)
    # V_arr = V_arr.reshape(win_size, 4000)
    for i in range(0, win_size):
        print(i)
        gan_outputs = gan_outputs1[i, :]

        real_sig = real_sig1[i, :]

        # V = V_arr[i, :]
        # S = S_arr[i, :]

        time_axis = np.arange(i * 4000, (i + 1) * 4000) / 400
        a = plt.figure()
        a.set_size_inches(12, 10)
        ax = plt.subplot(p)
        major_ticksx = np.arange(10 * i, 10 * (i + 1), 1)
        minor_ticksx = np.arange(10 * i, 10 * (i + 1), 0.25)
        major_ticksy = np.arange(-1.5, 1.5, 0.3)
        minor_ticksy = np.arange(-1.5, 1.5, 0.075)
        ax.set_xticks(major_ticksx)
        ax.set_xticks(minor_ticksx, minor=True)
        ax.set_yticks(major_ticksy)
        ax.set_yticks(minor_ticksy, minor=True)

        plt.plot(time_axis, real_sig, linewidth=0.7, color='k')
        # plt.scatter(time_axis[real_sig * S != 0], real_sig[real_sig * S != 0], c='#bcbd22', s=100, marker=(5, 1),
        #             alpha=0.5)
        # plt.scatter(time_axis[real_sig * V != 0], real_sig[real_sig * V != 0], c='#2ca02c', s=100, marker=(5, 1),
        #             alpha=0.5)
        ax.grid(which='minor', alpha=0.2, color='r')
        ax.grid(which='major', alpha=0.5, color='r')
        plt.title("Original ECG Segment", fontsize=15)
        plt.axis([10 * i, 10 * (i + 1), y_m, y_p])
        plt.xlabel('Time (seconds)', fontsize=13)
        plt.ylabel('Amplitude', fontsize=13)
        ax2 = plt.subplot(p+1, sharex=ax)
        # Major ticks every 20, minor ticks every 5
        major_ticksx = np.arange(10 * i, 10 * (i + 1), 1)
        minor_ticksx = np.arange(10 * i, 10 * (i + 1), 0.25)
        major_ticksy = np.arange(-1.5, 1.5, 0.3)
        minor_ticksy = np.arange(-1.5, 1.5, 0.075)
        ax2.set_xticks(major_ticksx)
        ax2.set_xticks(minor_ticksx, minor=True)
        ax2.set_yticks(major_ticksy)
        ax2.set_yticks(minor_ticksy, minor=True)
        plt.plot(time_axis, gan_outputs, linewidth=0.7, color='k')
        # plt.scatter(time_axis[gan_outputs * S != 0], gan_outputs[gan_outputs * S != 0], c='#bcbd22', s=100,
        #             marker=(5, 1), alpha=0.5)
        # plt.scatter(time_axis[gan_outputs * V != 0], gan_outputs[gan_outputs * V != 0], c='#2ca02c', s=100,
        #             marker=(5, 1), alpha=0.5)
        ax2.grid(which='minor', alpha=0.2, color='r')
        ax2.grid(which='major', alpha=0.5, color='r')
        plt.title("Operational Cycle-GAN", fontsize=15)
        plt.xlabel('Time (seconds)', fontsize=13)
        plt.ylabel('Amplitude', fontsize=13)
        plt.axis([10 * i, 10 * (i + 1), y_m, y_p])
        if sys.argv[1] == "PCG":
            clean_sig = clean_sig1[i, :]
            ax3 = plt.subplot(p+2)
            major_ticksx = np.arange(10 * i, 10 * (i + 1), 1)
            minor_ticksx = np.arange(10 * i, 10 * (i + 1), 0.25)
            major_ticksy = np.arange(-1.5, 1.5, 0.3)
            minor_ticksy = np.arange(-1.5, 1.5, 0.075)
            ax3.set_xticks(major_ticksx)
            ax3.set_xticks(minor_ticksx, minor=True)
            ax3.set_yticks(major_ticksy)
            ax3.set_yticks(minor_ticksy, minor=True)

            plt.plot(time_axis, clean_sig, linewidth=0.7, color='k')
            # plt.scatter(time_axis[real_sig * S != 0], real_sig[real_sig * S != 0], c='#bcbd22', s=100, marker=(5, 1),
            #             alpha=0.5)
            # plt.scatter(time_axis[real_sig * V != 0], real_sig[real_sig * V != 0], c='#2ca02c', s=100, marker=(5, 1),
            #             alpha=0.5)
            ax3.grid(which='minor', alpha=0.2, color='r')
            ax3.grid(which='major', alpha=0.5, color='r')
            plt.title("Clean PCG Segment (Ground truth)", fontsize=15)
            plt.axis([10 * i, 10 * (i + 1), y_m, y_p])
            plt.xlabel('Time (seconds)', fontsize=13)
            plt.ylabel('Amplitude', fontsize=13)

        plt.tight_layout(pad=1.0)
        if sys.argv[1] == "ECG":
            fname = "deneme_ECG/" + str(patient_num) + "/pt_" + str(patient_num) + "_" + str(i)
        elif sys.argv[1] == "PCG":
            fname = "deneme_PCG/" + str(patient_num) + "/pt_" + str(patient_num) + "_" + str(i)

        plt.savefig(fname, dpi=200)
        plt.close()
