#%%
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
import re

# added by akira
def zigzag(seq):  return seq[::2], seq[1::2]

#%%
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--result_dir", type=Path, required=True)
args = parser.parse_args()

#%%
dirs = [dir for dir in args.result_dir.iterdir() if dir.is_dir() and re.match(r"^[0-9]+$", dir.stem)]
dirs.sort(key=lambda dir: dir.stem)

#%%
Path("figures").mkdir(exist_ok=True)
Path("summary_files").mkdir(exist_ok=True)

#%%
print("Initialize variables....")
N = len(dirs)
#tmp = np.loadtxt(dirs[0] / "summary_files/resample_times.txt")
#T = tmp.shape[0]
T = 100

#resample_times = np.empty((N, T))
log_likelihoods = np.empty((N, T)) # [akira] T+1 -> T

letter_ARIs = np.empty((N, T))
letter_macro_f1_scores = np.empty((N, T))
letter_micro_f1_scores = np.empty((N, T))
letter_NMIs = np.empty((N, T)) # added by akira

word_ARIs = np.empty((N, T))
word_macro_f1_scores = np.empty((N, T))
word_micro_f1_scores = np.empty((N, T))
word_NMIs = np.empty((N, T)) # added by akira

print("Done!")

#%%
print("Loading results....")
for i, dir in enumerate(dirs):
    #resample_times[i] = np.loadtxt(dir / "summary_files/resample_times.txt")
    log_likelihood = np.loadtxt(dir / "summary_files/log_likelihood.txt")
    letter_ARIs[i] = np.loadtxt(dir / "summary_files/Letter_ARI.txt")
    letter_NMIs[i] = np.loadtxt(dir / "summary_files/Letter_NMI.txt")
    letter_macro_f1_scores[i] = np.loadtxt(dir / "summary_files/Letter_macro_F1_score.txt")
    letter_micro_f1_scores[i] = np.loadtxt(dir / "summary_files/Letter_micro_F1_score.txt")
    word_ARIs[i] = np.loadtxt(dir / "summary_files/Word_ARI.txt")
    word_NMIs[i] = np.loadtxt(dir / "summary_files/Word_NMI.txt")
    word_macro_f1_scores[i] = np.loadtxt(dir / "summary_files/Word_macro_F1_score.txt")
    word_micro_f1_scores[i] = np.loadtxt(dir / "summary_files/Word_micro_F1_score.txt")

    # akira
    print(i, "len:log_likelihood",len(log_likelihood))
    #train_iter = T
    if (len(log_likelihood) == T):
        log_likelihoods[i] = log_likelihood
    elif (len(log_likelihood) == T+1): # yousosu ga 101 no toki
        log_likelihoods[i] =  np.delete(log_likelihood,0)
    elif ( len(log_likelihood) == 2*T ): # yousosuu ga 200 no toki
        log_likelihood_zig,log_likelihood_zag = zigzag(log_likelihood)
        log_likelihoods[i] = log_likelihood_zig
        #log_likelihoods[i] = np.append(log_likelihood[0], log_likelihood_zig)
        #np.savetxt("summary_files/log_likelihood.txt",log_likelihood_zig)

print("Done!")



#%%
print("Ploting...")
#plt.clf()
#plt.errorbar(range(T), resample_times.mean(axis=0), yerr=resample_times.std(axis=0))
#plt.xlabel("Iteration")
#plt.ylabel("Execution time [sec]")
#plt.title("Transitions of the execution time")
#plt.savefig("figures/summary_of_execution_time.png")

plt.clf()
plt.errorbar(range(T), log_likelihoods.mean(axis=0), yerr=log_likelihoods.std(axis=0))
# [akira] T+1 -> T
plt.xlabel("Iteration")
plt.ylabel("Log likelihood")
plt.title("Transitions of the log likelihood")
plt.savefig("figures/summary_of_log_likelihood.png")

plt.clf()
plt.errorbar(range(T), word_ARIs.mean(axis=0), yerr=word_ARIs.std(axis=0), label="Word ARI")
plt.errorbar(range(T), letter_ARIs.mean(axis=0), yerr=letter_ARIs.std(axis=0), label="Letter ARI")
plt.xlabel("Iteration")
plt.ylabel("ARI")
plt.ylim(0.0,1.0)
plt.title("Transitions of the ARI")
plt.legend()
plt.savefig("figures/summary_of_ARI.png")

# akira
plt.clf()
plt.errorbar(range(T), word_NMIs.mean(axis=0), yerr=word_NMIs.std(axis=0), label="Word NMI")
plt.errorbar(range(T), letter_NMIs.mean(axis=0), yerr=letter_NMIs.std(axis=0), label="Letter NMI")
plt.xlabel("Iteration")
plt.ylabel("NMI")
plt.ylim(0.0,1.0)
plt.title("Transitions of the NMI")
plt.legend()
plt.savefig("figures/summary_of_NMI.png")
#

plt.clf()
plt.errorbar(range(T), word_macro_f1_scores.mean(axis=0), yerr=word_macro_f1_scores.std(axis=0), label="Word macro F1")
plt.errorbar(range(T), letter_macro_f1_scores.mean(axis=0), yerr=letter_macro_f1_scores.std(axis=0), label="Letter macro F1")
plt.xlabel("Iteration")
plt.ylabel("Macro F1 score")
plt.ylim(0.0,1.0)
plt.title("Transitions of the macro F1 score")
plt.legend()
plt.savefig("figures/summary_of_macro_F1_score.png")

plt.clf()
plt.errorbar(range(T), word_micro_f1_scores.mean(axis=0), yerr=word_micro_f1_scores.std(axis=0), label="Word micro F1")
plt.errorbar(range(T), letter_micro_f1_scores.mean(axis=0), yerr=letter_micro_f1_scores.std(axis=0), label="Letter micro F1")
plt.xlabel("Iteration")
plt.ylabel("Micro F1 score")
plt.ylim(0.0,1.0)
plt.title("Transitions of the micro F1 score")
plt.legend()
plt.savefig("figures/summary_of_micro_F1_score.png")
print("Done!")

#%%
print("Save npy files...")
#np.save("summary_files/resample_times.npy", resample_times)
np.save("summary_files/log_likelihoods.npy", log_likelihoods)

np.save("summary_files/letter_ARI.npy", letter_ARIs)
np.save("summary_files/letter_macro_F1.npy", letter_macro_f1_scores)
np.save("summary_files/letter_micro_F1.npy", letter_micro_f1_scores)
np.save("summary_files/letter_NMI.npy", letter_NMIs)
np.save("summary_files/word_ARI.npy", word_ARIs)
np.save("summary_files/word_macro_F1.npy", word_macro_f1_scores)
np.save("summary_files/word_micro_F1.npy", word_micro_f1_scores)
np.save("summary_files/word_NMI.npy", word_NMIs)
print("Done!")
