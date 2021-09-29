### you can display the mean and variance of phoneme/word ARI in command prompt
### It is useful for make table
# python ./eval_src/ARI_analysis.py -t ./RESULTS/tcds0412/
import numpy as np
import argparse

default_tar = "./RESULTS/1209_npbdaa/"  # path of target result
trial = 20  # num of trial
print("trial",trial)

parser = argparse.ArgumentParser()
parser.add_argument("-t",default=default_tar)
args = parser.parse_args()
target = args.t

# added by akira 
all_pho_NMI = np.load(target+"summary_files/letter_NMI.npy")
all_word_NMI = np.load(target+"summary_files/word_NMI.npy")

last_pho_list = []
last_word_list = []
for i in range(trial):
    last_pho_list.append(all_pho_NMI[i][-1])
    last_word_list.append(all_word_NMI[i][-1])
print(f"Phoneme NMI:\t{np.array(last_pho_list).mean():.3f} \pm {np.array(last_pho_list).std():.3f}")
print(f"Word NMI:\t{np.array(last_word_list).mean():.3f} \pm {np.array(last_word_list).std():.3f}")
#

all_pho_ARI = np.load(target+"summary_files/letter_ARI.npy")
all_word_ARI = np.load(target+"summary_files/word_ARI.npy")

last_pho_list = []
last_word_list = []
for i in range(trial):
    last_pho_list.append(all_pho_ARI[i][-1])
    last_word_list.append(all_word_ARI[i][-1])
print(f"Phoneme ARI:\t{np.array(last_pho_list).mean():.3f} \pm {np.array(last_pho_list).std():.3f}")
print(f"Word ARI:\t{np.array(last_word_list).mean():.3f} \pm {np.array(last_word_list).std():.3f}")

