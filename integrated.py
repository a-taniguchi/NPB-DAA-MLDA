import os
import shutil
import glob
import re
import numpy as np
import random
import math
import pyhsmm
import warnings
import time
import itertools
import sys
import pickle
from subprocess import Popen
from subprocess import call
from pathlib import Path
from matplotlib import pyplot as plt
from pyhlm.model import WeakLimitHDPHLM, WeakLimitHDPHLMPython
from pyhlm.internals.hlm_states import WeakLimitHDPHLMStates
from pyhlm.word_model import LetterHSMM, LetterHSMMPython
from tqdm import trange
from collections import Counter
from joblib import Parallel, delayed
from multiprocessing import Array
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from util.config_parser import ConfigParser_with_eval
import copy

warnings.filterwarnings('ignore')
utr_num = [3, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 4]  #num of utterance for each object
obj_idxes = [i for (i, v) in enumerate(utr_num) for _ in range(v)]        #utterance index keeping its ubject index
K = 7       #num of categories for MLDA
N = len(utr_num)    #num of objects
L = 10      #num of candidate word sequences
S = 0      #num of utterance
for n in utr_num:
    S += n
word_weight_setting = "const"		# setting: "const" or "vary", value: "const"=200 "vary"=0~200, you can change values in word_weight_set()

def load_config(filename):
    cp = ConfigParser_with_eval()
    cp.read(filename)
    return cp

#load MFCC data
def load_datas():
    data = []
    names = np.loadtxt("files.txt", dtype=str)
    files = names
    for name in names:
        static = np.loadtxt("DATA/" + name + ".txt")
        delta = np.loadtxt("DATA/" + name + "_d.txt")
        delta_delta = np.loadtxt("DATA/" + name + "_dd.txt")
        data.append(np.concatenate((static, delta, delta_delta), axis = 1))
        # data.append(np.loadtxt("DATA/"+name+".txt"))
    return data

def unpack_durations(dur):
    unpacked = np.zeros(dur.sum())
    d = np.cumsum(dur)
    unpacked[d-1] = 1.0
    return unpacked

def save_stateseq(model):
    # Save sampled states sequences.
    names = np.loadtxt("files.txt", dtype=str)
    for i, s in enumerate(model.states_list):
        with open("results/" + names[i] + "_s.txt", "a") as f:
            np.savetxt(f, s.stateseq, fmt="%d")
        with open("results/" + names[i] + "_l.txt", "a") as f:
            np.savetxt(f, s.letter_stateseq, fmt="%d")
        with open("results/" + names[i] + "_d.txt", "a") as f:
            np.savetxt(f, unpack_durations(s.durations_censored), fmt="%d")

def save_cand_stateseq(model, l, iter):
    # Save candidate sampled states sequences.
    names = np.loadtxt("files.txt", dtype=str)
    Path("cand_results").mkdir(exist_ok=True)
    Path("cand_results/time"+str(iter)+"_"+str(l)+"-th_results").mkdir(exist_ok=True)
    for i, s in enumerate(model.states_list):
        with open("cand_results/time"+str(iter)+"_"+str(l)+"-th_results/" + names[i] + "_s.txt", "a") as f:
            np.savetxt(f, s.stateseq, fmt="%d")
        with open("cand_results/time"+str(iter)+"_"+str(l)+"-th_results/" + names[i] + "_l.txt", "a") as f:
            np.savetxt(f, s.letter_stateseq, fmt="%d")
        with open("cand_results/time"+str(iter)+"_"+str(l)+"-th_results/" + names[i] + "_d.txt", "a") as f:
            np.savetxt(f, unpack_durations(s.durations_censored), fmt="%d")

def save_params_as_text(itr_idx, model):
    with open("parameters/ITR_{0:04d}.txt".format(itr_idx), "w") as f:
        f.write(str(model.params))

def save_params_as_file(iter_idx, model):
    params = model.params
    root_dir = Path("parameters/ITR_{0:04d}".format(iter_idx))
    root_dir.mkdir(exist_ok=True)
    save_json(root_dir, params)

def save_json(root_dir, json_obj):
    for keyname, subjson in json_obj.items():
        type_of_subjson = type(subjson)
        if type_of_subjson == dict:
            dir = root_dir / keyname
            dir.mkdir(exist_ok=True)
            save_json(dir, json_obj[keyname])
        else:
            savefile = root_dir / f"{keyname}.txt"
            if type_of_subjson == np.ndarray:
                if subjson.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
                    np.savetxt(savefile, subjson, fmt="%d")
                else:
                    np.savetxt(savefile, subjson)
            else:
                savefile.write_text(str(subjson))

def save_params_as_npz(iter_idx, model):
    params = model.params
    flatten_params = flatten_json(params)
    # flatten_params = copy_flatten_json(flatten_params)
    np.savez(f"parameters/ITR_{iter_idx:04d}.npz", **flatten_params)

def flatten_json(json_obj, keyname_prefix=None, dict_obj=None):
    if dict_obj is None:
        dict_obj = {}
    if keyname_prefix is None:
        keyname_prefix = ""
    for keyname, subjson in json_obj.items():
        if type(subjson) == dict:
            prefix = f"{keyname_prefix}{keyname}/"
            flatten_json(subjson, keyname_prefix=prefix, dict_obj=dict_obj)
        else:
            dict_obj[f"{keyname_prefix}{keyname}"] = subjson
    return dict_obj

def unflatten_json(flatten_json_obj):
    dict_obj = {}
    for keyname, value in flatten_json_obj.items():
        current_dict = dict_obj
        splitted_keyname = keyname.split("/")
        for key in splitted_keyname[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        current_dict[splitted_keyname[-1]] = value
    return dict_obj

def copy_flatten_json(json_obj):
    new_json = {}
    for keyname, subjson in json_obj.items():
        type_of_subjson = type(subjson)
        if type_of_subjson in [int, float, complex, bool]:
            new_json[keyname] = subjson
        elif type_of_subjson in [list, tuple]:
            new_json[keyname] = subjson[:]
        elif type_of_subjson == np.ndarray:
            new_json[keyname] = subjson.copy()
        else:
            raise NotImplementedError(f"type :{type_of_subjson} can not copy. Plz implement here!")
    return new_json

def save_loglikelihood(model):
    with open("summary_files/log_likelihood.txt", "a") as f:
        f.write(str(model.log_likelihood()) + "\n")

def save_resample_times(resample_time):
    with open("summary_files/resample_times.txt", "a") as f:
        f.write(str(resample_time) + "\n")

### setting weight value for word cue
def word_weight_set(flag):
    if flag == "vary":
        if iter <= 10:
            word_weight = 0
        elif iter >= 11 and word_weight <= 190:   # until weight value is 200
            word_weight = 40+((iter-10)*10)
        else:
            word_weight = 200
    elif flag == "const": word_weight = 200
    else:
        print("word weight setting invalid")
        exit(0)
    return word_weight

#%% parse arguments
#####
hypparams_model = "hypparams/model.config"
hypparams_letter_duration = "hypparams/letter_duration.config"
hypparams_letter_hsmm = "hypparams/letter_hsmm.config"
hypparams_letter_observation = "hypparams/letter_observation.config"
hypparams_pyhlm = "hypparams/pyhlm.config"
hypparams_word_length = "hypparams/word_length.config"
hypparams_superstate = "hypparams/superstate.config"

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--model", default=hypparams_model, help="hyper parameters of model")
parser.add_argument("--letter_duration", default=hypparams_letter_duration, help="hyper parameters of letter duration")
parser.add_argument("--letter_hsmm", default=hypparams_letter_hsmm, help="hyper parameters of letter HSMM")
parser.add_argument("--letter_observation", default=hypparams_letter_observation, help="hyper parameters of letter observation")
parser.add_argument("--pyhlm", default=hypparams_pyhlm, help="hyper parameters of pyhlm")
parser.add_argument("--word_length", default=hypparams_word_length, help="hyper parameters of word length")
parser.add_argument("--superstate", default=hypparams_superstate, help="hyper parameters of superstate")
parser.add_argument("--cont", default=0, help="iteration of previous trial")
args = parser.parse_args()
hypparams_model = args.model
hypparams_letter_duration = args.letter_duration
hypparams_letter_hsmm = args.letter_hsmm
hypparams_letter_observation = args.letter_observation
hypparams_pyhlm = args.pyhlm
hypparams_word_length = args.word_length
hypparams_superstate = args.superstate
cont = int(args.cont)
#####
#####
Path("results").mkdir(exist_ok=True)
Path("parameters").mkdir(exist_ok=True)
Path("summary_files").mkdir(exist_ok=True)
Path("CAND"+str(L)).mkdir(exist_ok=True)
Path("CAND"+str(L)+"/Candidate").mkdir(exist_ok=True)
Path("CAND"+str(L)+"/Chosen").mkdir(exist_ok=True)
Path("Saved").mkdir(exist_ok=True)
Path("MLDA_result").mkdir(exist_ok=True)
Path("word_hist_result").mkdir(exist_ok=True)
Path("sampled_z_lnsj").mkdir(exist_ok=True)
Path("word_hist_candies").mkdir(exist_ok=True)
Path("model").mkdir(exist_ok=True)
for l in range(L):
    Path(f"model/{l}").mkdir(exist_ok=True)

#%% config parse
config_parser = load_config(hypparams_model)
section         = config_parser["model"]
thread_num      = section["thread_num"]
pretrain_iter   = section["pretrain_iter"]
train_iter      = section["train_iter"]
word_num        = section["word_num"]
letter_num      = section["letter_num"]
observation_dim = section["observation_dim"]

hlm_hypparams = load_config(hypparams_pyhlm)["pyhlm"]
config_parser = load_config(hypparams_letter_observation)
obs_hypparams = [config_parser[f"{i+1}_th"] for i in range(letter_num)]
config_parser = load_config(hypparams_letter_duration)
dur_hypparams = [config_parser[f"{i+1}_th"] for i in range(letter_num)]
len_hypparams = load_config(hypparams_word_length)["word_length"]
letter_hsmm_hypparams = load_config(hypparams_letter_hsmm)["letter_hsmm"]
superstate_config = load_config(hypparams_superstate)

#####
#%% make instance of distributions and model
# pre_trial_iter = 0
if cont != 0:
    saved_file_name = glob.glob("Saved/*")
    # pre_trial_iter = re.sub(r'\D', '', saved_file_name)
    for s in saved_file_name:
        with open(s, mode="rb") as f:
            model = pickle.load(f)
else:
    letter_obs_distns = [pyhsmm.distributions.Gaussian(**hypparam) for hypparam in obs_hypparams]
    letter_dur_distns = [pyhsmm.distributions.PoissonDuration(**hypparam) for hypparam in dur_hypparams]
    dur_distns = [pyhsmm.distributions.PoissonDuration(lmbda=20) for _ in range(word_num)]
    length_distn = pyhsmm.distributions.PoissonDuration(**len_hypparams)

#load data
files = np.loadtxt("files.txt", dtype=str)
mfcc_data = load_datas()

#Pretraining
print("Pre-training")
if cont == 0:
    cand_models = []
    for _ in trange(L): #copy L candidates
        letter_hsmm = LetterHSMM(**letter_hsmm_hypparams, obs_distns=letter_obs_distns, dur_distns=letter_dur_distns)
        model = WeakLimitHDPHLM(**hlm_hypparams, letter_hsmm=letter_hsmm, dur_distns=dur_distns, length_distn=length_distn)
        for data in mfcc_data:
            letter_hsmm.add_data(data, **superstate_config["DEFAULT"])
        for t in range(pretrain_iter):
            letter_hsmm.resample_model(num_procs=0)
        letter_hsmm.states_list = []
        for name, data in zip(files, mfcc_data):
            model.add_data(data, **superstate_config[name], generate=False)

        model.resample_states(num_procs=thread_num)     #Update HDPHLM
        cand_models.append(copy.deepcopy(model))
print("Done it")

saved_model = []

devnull = open("/dev/null", "w")    #Not showing the result of MLDA in command line
#Repeat following procedure (train_iter times)
for iter in trange(cont, train_iter):
    st = time.time()
    print(f"{iter+1}-th inference")
    print("sample candidate word seq")
    if iter >= 1:
        cand_models = [copy.deepcopy(pre_cand_models[np.random.choice(L, p=pre_weight)]) for _ in range(L)]    # sample candidates in proportion to pre-weight
    mlda_phi = []
    mlda_theta = []
    mlda_N_mz = []
    word_weight = word_weight_set(word_weight_setting)
    print(f"word cue weight: {word_weight}")

    for l in range(L):      # object categorization using each candidates
        cand_models[l].resample_model(num_procs=thread_num)
        hist = np.zeros((N, word_num), dtype=int)
        for obj_idx, state in zip(obj_idxes, cand_models[l].states_list):
            hist[obj_idx] += np.histogram(state.stateseq_norep, bins=word_num, range=(0, word_num))[0]
        np.savetxt(f"./mlda_data/word_hist_candies/{l}-th_word_hist.txt", hist, fmt="%d", delimiter='\t')
        p = Popen(["./mlda", "-learn", "-config", "lda_config.json", "-data0", f"./mlda_data/word_hist_candies/{l}-th_word_hist.txt", "-weight0", f"{word_weight}", "-save_dir", f"model/{l}"], stdout=devnull, stderr=devnull)       # conduct MLDA
        r = p.wait()
        if r != 0:
            print(f"MLDA finished with return code {r}")
            sys.exit(1)
        #Read the output from MLDA
        mlda_phi.append(np.loadtxt(f"model/{l}/phi000.txt"))#*番目のモダリティにおいて，カテゴリkで特徴oが発生する確率 theta^w_k [K, word_num]
        mlda_theta.append(np.loadtxt(f"model/{l}/theta.txt"))#n番目の物体にカテゴリkが割り当てられる確率 pi_n [obj_num, k]
        mlda_N_mz.append(np.loadtxt(f"model/{l}/Nmz.txt"))#学習の結果，モダリティmにカテゴリzが割り当てられた回数 [Modal, K]
        # print(f"{l+1}-th cand is finished.")

    #%%
    MLDA_path = "MLDA_result/"+str(f"{iter+1:0>3}")
    shutil.copytree("model", MLDA_path, dirs_exist_ok=True)

    print("Choose one set of plausible global parameters")
    #Using unigram rescaling method, get word sequences \hat(w)^(w)_(di)
    #that considering both of estimation results NPB-DAA and MLDA

    ### sampling object category
    z_lnsj = []     #topic of n-th obj, s-th utter, j-th word generated from l-th candidate model
    for l in range(L):
        z_nsj = []
        for ns_idx, n in zip(range(S), obj_idxes):
            z_j = []
            len_of_lns = len(cand_models[l].states_list[ns_idx].stateseq_norep)
            for j in range(len_of_lns):
                Pz_lnsj = mlda_phi[l][:, cand_models[l].states_list[ns_idx].stateseq_norep[j]] * mlda_theta[l][n, :]
                Pz_lnsj /= Pz_lnsj.sum()
                sampled_z = np.random.choice(K, p=Pz_lnsj)
                z_j.append(sampled_z)
            z_nsj.append(z_j)
        z_lnsj.append(z_nsj)
    with open("./sampled_z_lnsj/"+str(f"{iter+1:0>3}")+".pkl", "wb") as file:
        pickle.dump(z_lnsj, file)

    weight_l = np.zeros(L)
    for l in range(L):
        weights = np.zeros(S)       #weight of utterance s-th utterance
        for ns_idx in range(S):
            len_of_lns = len(cand_models[l].states_list[ns_idx].stateseq_norep)       #length of word seq generated by l-th set of GP, s-th otterance
            table = np.empty((K, len_of_lns))     #represent the topic of s-th utterance, j-th word
            for k in range(K):
                for j in range(len_of_lns):
                    # table[k, j] = Pz_lnsj[k] * mlda_phi[l][k, cand_models[l].states_list[ns_idx].stateseq_norep[j]]      #table of topics of each words
                    table[k, j] = mlda_phi[l][k, cand_models[l].states_list[ns_idx].stateseq_norep[j]]      #table of topics of each words
            nume = 1
            for idx, z in enumerate(z_lnsj[l][ns_idx]):
                nume *= table[z][idx]       #calculate prob of topic of s-th utterance generated by l-th set of GP (prod_(j})
            deno = table.sum(axis=0).prod()     #calculate prob of all word seq (sum_(k)->prod_(j))
            weight_ns = nume / deno
            weights[ns_idx] = weight_ns
        weight_l[l] = weights.prod()        #calculate weight of l-th set of word seq (prod_(n,s))
    weight_l /= weight_l.sum()      #normalization
    choiced_idx = np.random.choice(L, p=weight_l)
    resample_model_time = time.time() - st

    # Save the file of word sequence (candidates and chosen one) at an iteration
    if iter<10 or (iter+1)%25 == 0:
        super_o_nsj = []
        for states in cand_models[choiced_idx].states_list:
            super_o_nsj.append(states.stateseq_norep.copy())
        with open("./CAND"+str(L)+"/Chosen/"+str(f"{iter+1:0>3}")+".pkl", "wb") as file:
            pickle.dump(super_o_nsj, file)

        cand_o = []
        for l in range(L):
            l_cand_o = []
            for states in cand_models[l].states_list:
                l_cand_o.append(states.stateseq_norep.copy())
            cand_o.append(l_cand_o)
        for l in range(L):
            with open("./CAND"+str(L)+"/Candidate/"+str(f"{iter+1:0>3}")+"_"+str(f"{l+1:0>2}")+".pkl", "wb") as file:
                pickle.dump(cand_o[l], file)
            save_cand_stateseq(cand_models[l], l, iter)

    # save most likely HDPHLM and save candidates' weight
    print(f"{choiced_idx}-th candy is chosen")
    model = None
    model = cand_models[choiced_idx]
    saved_model.append(cand_models[choiced_idx])
    pre_weight = weight_l
    pre_cand_models = cand_models
    cand_models = None

    save_resample_times(resample_model_time)
    shutil.rmtree("Saved")
    os.mkdir("Saved")
    with open("Saved/"+str(f"{iter+1:0>2}")+".pkl", "wb") as f:
        pickle.dump(model, f)

#save parameters of each iterations
for i in range(train_iter):
    save_stateseq(saved_model[i])
    save_loglikelihood(saved_model[i])
    save_params_as_npz(i, saved_model[i])
    save_loglikelihood(saved_model[i])

print("saved parameters removed")
shutil.rmtree("Saved")
