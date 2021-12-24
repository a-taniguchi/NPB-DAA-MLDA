### you can plot the result of word segmentation, ground truth label and its waveform in the same figure
import sys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap
import wave
import struct
from pathlib import Path
from tqdm import trange, tqdm

args = sys.argv # recive commadn line 

global wav_target
global data_target
global result_target
wav_target = "./cutted_mod/"    # path of wav files
data_target = "./cutted_dataset/"   # path of MFCC's data files
result_target = args[1]+"/20/"  #"./0128_real_200/wav_lab_figs/" # path of result files

def _boundary(label):
    diff = np.diff(label)
    diff[diff!=0] = 1
    return np.concatenate((diff, [0]))

def get_labels(names):
    # letter_labels = [np.loadtxt("LABEL/" + name + ".lab") for name in names]
    word_labels = [np.loadtxt(data_target+"LABEL/" + name + ".lab2") for name in names]
    return word_labels

def get_datas_and_length(names):
    datas = [np.loadtxt(data_target+"DATA/" + name + ".txt") for name in names]
    length = [len(d) for d in datas]
    # print("length",length)
    return length

def _joblib_get_results(names, lengths, c):
    from joblib import Parallel, delayed
    def _component(name, length, c):
        return np.loadtxt(result_target+"results/" + name + "_" + c + ".txt").reshape((-1, length))
    return Parallel(n_jobs=-1)([delayed(_component)(n, l, c) for n, l in zip(names, lengths)])

def get_idx(names, lengths):
    wrd_idxes = list([np.loadtxt(result_target+"results/" + name + "_s.txt").reshape((-1, l))[-1] for name, l in zip(names, lengths)])
    wrd_idxes_unique = []
    for i in range(len(wrd_idxes)):
        wrd_idxes_unique.append(sorted(set(wrd_idxes[i]), key=list(wrd_idxes[i]).index))
    # for i in range(len(wrd_idxes)):
    #     print(f"before: {wrd_idxes[i]}")
    #     print(f"after: {wrd_idxes_unique[i]}")
    return wrd_idxes_unique

def _plot_discreate_sequence(true_data, title, sample_data, wrd_idx, cmap=None, cmap2=None, label_cmap=None):
    mpl.rcParams['axes.xmargin'] = 0

    # plot label data
    ax = plt.subplot2grid((10, 1), (1, 0))
    plt.sca(ax)
    if label_cmap is None:
        label_cmap = cmap
    ax.matshow([true_data], aspect='auto', cmap=label_cmap)
    plt.ylabel('Truth Label')

    # plot waveform
    ax = plt.subplot2grid((10, 1), (2, 0), rowspan = 8)
    plt.suptitle(title)
    plt.sca(ax)
    wf = wave.open(wav_target+title+".wav", mode='rb')
    buf = wf.readframes(-1)
    if wf.getsampwidth() == 2:
        data = np.frombuffer(buf, dtype='int16')
    elif wf.getsampwidth() == 4:
        data = np.frombuffer(buf, dtype='int32')
    if wf.getnchannels()==2:
        data_l = data[::2]
        data_r = data[1::2]
        plt.plot(data_l, color="black", alpha=0.5)
        plt.plot(data_r, color="black", alpha=0.5)
    else:
        plt.plot(data, color="black", alpha=0.5)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    #plot segmentation result
    if cmap2 is not None:
        cmap = cmap2
    ax.matshow(sample_data, extent=[*xlim, *ylim], aspect='auto', cmap=cmap, alpha=0.5)
    wrd_idx_int = [int(i) for i in wrd_idx]
    plt.xlabel(str(wrd_idx_int))

def _get_results(target, names, lengths, c):
    return [np.loadtxt(target+"results/" + name + "_" + c + ".txt").reshape((-1, l)) for name, l in zip(names, lengths)]
    # return np.loadtxt(target + "results/" + names + "_" + c + ".txt").reshape((-1, lengths))

Path("figures").mkdir(exist_ok=True)
names = np.loadtxt(data_target+"files.txt", dtype=str)
lengths = get_datas_and_length(names)
w_labels = get_labels(names)
w_results = _joblib_get_results(names, lengths, "s")
w_idxes = get_idx(names, lengths)
word_num = 50   # num of weak-limit latent words
wcolors = ListedColormap([cm.tab20(float(i)/word_num) for i in range(word_num)])

for i, name in enumerate(tqdm(names)):
    plt.clf()
    _plot_discreate_sequence(_boundary(w_labels[i]), name, w_results[i], w_idxes[i], cmap=wcolors, label_cmap=cm.binary)
    plt.savefig(result_target+"figures/" + name + "_wav_lab.eps", bbox_inches="tight")
    plt.savefig(result_target+"figures/" + name + "_wav_lab.png", bbox_inches="tight")
    plt.clf()

print("Done!")
