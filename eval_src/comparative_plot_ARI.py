### you can plot transition of phoneme/word ARI and mean ARI at the last iteration
### you can also conduct Welchi's t test
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
import scipy.stats as st

Path("figures").mkdir(exist_ok=True)

trial = 20  # num of trial
npbdaa_dir = "./RESULTS/1209_npbdaa/"   # directory path of NPB-DAA's result
comp_dir = ["./RESULTS/0217_real_0_40_200/"]    # directory path of comparative method's result. you can sprcify it multiplly
label_list = ["Proposed method"]    # display label list excluded NPB-DAA

def trans_plot():
    ### Word plot
    plt.clf()
    ### plot npb-daa's result
    word_ARIs = np.load(npbdaa_dir+"summary_files/word_ARI.npy")
    plt.errorbar(range(len(word_ARIs[0])), word_ARIs.mean(axis=0), yerr=word_ARIs.std(axis=0), label="NPB-DAA", alpha=0.5)

    for dir,l in zip(comp_dir, label_list):
        word_ARIs = np.load(dir+"summary_files/word_ARI.npy")
        plt.errorbar(range(len(word_ARIs[0])), word_ARIs.mean(axis=0), yerr=word_ARIs.std(axis=0), label=l, alpha=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Segmentation ARI")
    plt.ylim(0, 1.1)
    plt.legend(loc='best')
    plt.savefig("figures/comp_trans_word_ARI_ex1.eps")
    plt.savefig("figures/comp_trans_word_ARI_ex1.png")

    ### Phoneme plot
    plt.clf()
    ### plot npb-daa's result
    phoneme_ARIs = np.load(npbdaa_dir+"summary_files/letter_ARI.npy")
    plt.errorbar(range(len(word_ARIs[0])), phoneme_ARIs.mean(axis=0), yerr=phoneme_ARIs.std(axis=0), label="NPB-DAA", alpha=0.5)

    for dir,l in zip(comp_dir, label_list):
        phoneme_ARIs = np.load(dir+"summary_files/letter_ARI.npy")
        plt.errorbar(range(len(word_ARIs[0])), phoneme_ARIs.mean(axis=0), yerr=phoneme_ARIs.std(axis=0), label=l, alpha=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Segmentation ARI")
    plt.ylim(0, 1.1)
    plt.legend(loc='best')
    plt.savefig("figures/comp_trans_pho_ARI_ex1.eps")
    plt.savefig("figures/comp_trans_pho_ARI_ex1.png")

def last_plot():
    x = np.array([1,2])
    x_label = ["Word ARI", "Phoneme ARI"]   # display label name

    plt.clf()
    ### path of npbdaa's result
    npbdaa_word = np.load(npbdaa_dir+"summary_files/word_ARI.npy")
    npbdaa_pho = np.load(npbdaa_dir+"summary_files/letter_ARI.npy")
    npbdaa_last_word = []
    npbdaa_last_pho = []

    for i in range(trial):
        npbdaa_last_word.append(npbdaa_word[i][-1])
        npbdaa_last_pho.append(npbdaa_pho[i][-1])
    npbdaa_heights = [np.array(npbdaa_last_word).mean(), np.array(npbdaa_last_pho).mean()]
    npbdaa_yerr = [np.array(npbdaa_last_word).std(), np.array(npbdaa_last_pho).std()]
    print("NPB-DAA")
    print(f"Phoneme ARI: {np.array(npbdaa_last_pho).mean():.3f}\pm{np.array(npbdaa_last_pho).std():.3f}")
    print(f"Word ARI: {np.array(npbdaa_last_word).mean():.3f}\pm{np.array(npbdaa_last_word).std():.3f}")

    height_data = [npbdaa_heights]
    yerr_data = [npbdaa_yerr]
    labels = ["NPB-DAA"]

    for dir,l in zip(comp_dir, label_list):
        word_ARI = np.load(dir+"summary_files/word_ARI.npy")
        phoneme_ARI = np.load(dir+"summary_files/letter_ARI.npy")
        last_word = []
        last_pho = []
        for i in range(trial):
            last_word.append(word_ARI[i][-1])
            last_pho.append(phoneme_ARI[i][-1])
        height_list = [np.array(last_word).mean(), np.array(last_pho).mean()]
        yerr_list = [np.array(last_word).std(), np.array(last_pho).std()]
        height_data.append(height_list)
        yerr_data.append(yerr_list)
        labels.append(l)
        print(l)
        print(f"Phoneme ARI: {np.array(last_pho).mean():.3f}\pm{np.array(last_pho).std():.3f}")
        print(f"Word ARI: {np.array(last_word).mean():.3f}\pm{np.array(last_word).std():.3f}")
    margin = 0.2
    total_width = 1 - margin

    for i, h in enumerate(height_data):
        x_pos = x - total_width *( 1- (2*i+1)/len(height_data) )/2
        plt.bar(x_pos, h, yerr=yerr_data[i], width=total_width/len(height_data),label=labels[i])
    plt.xticks(x, x_label)
    plt.ylabel("Segmentation ARI")
    plt.ylim(0, 1.1)
    plt.legend(loc='best')
    plt.savefig("figures/comp_last_seg_ARI_ex1.eps")
    plt.savefig("figures/comp_last_seg_ARI_ex1.png")

# ###This function only applied if len(comp_dir)==1
def t_test():
    a_list = np.load(npbdaa_dir+"summary_files/word_ARI.npy", 'r')
    b_list = np.load(comp_dir[0]+"summary_files/word_ARI.npy", 'r')
    xa = []
    xb = []
    for i in range(trial):
        xa.append(a_list[i][-1])
        xb.append(b_list[i][-1])

    t, p = st.ttest_ind(xa, xb, equal_var=False)
    MU = abs(np.array(xa).mean()-np.array(xb).mean())
    SE =  MU/t
    DF = len(xa)+len(xb)-2
    CI = st.t.interval( alpha=0.95, loc=MU, scale=SE, df=DF )   # Welchi's t test

    # print(f"Comparing weight value {a_weight} and {b_weight}")
    print(f"p value = {p:.5f}")
    print(f"t value = {t:.2f}")

print("Plot transition of ARI")
trans_plot()
print("Plot ARI of last iteration")
last_plot()
print("Welchi\'s t test")
t_test()
