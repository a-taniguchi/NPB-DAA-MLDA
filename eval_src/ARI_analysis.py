### you can display the mean and variance of phoneme/word ARI in command prompt
### It is useful for make table
import numpy as np
import argparse

default_tar = "./RESULTS/1209_npbdaa/"  # path of target result
trial = 20  # num of trial

parser = argparse.ArgumentParser()
parser.add_argument("-t",default=default_tar)
args = parser.parse_args()
target = args.t

all_pho_ARI = np.load(target+"summary_files/letter_ARI.npy")
all_word_ARI = np.load(target+"summary_files/word_ARI.npy")

last_pho_list = []
last_word_list = []
for i in range(trial):
    last_pho_list.append(all_pho_ARI[i][-1])
    last_word_list.append(all_word_ARI[i][-1])
print(f"Phoneme ARI: {np.array(last_pho_list).mean():.3f}\pm{np.array(last_pho_list).std():.3f}")
print(f"Word ARI: {np.array(last_word_list).mean():.3f}\pm{np.array(last_word_list).std():.3f}")
