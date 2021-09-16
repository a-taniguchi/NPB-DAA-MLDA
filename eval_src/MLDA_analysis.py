### you can plot the categorization ARI comparing multiple method
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import seaborn as sns

target = "./MLDA_test_dir/1209_npbdaa/" # path of target dir
save_dir = "./MLDA_test_dir/comparative_ex/tcds_1/" # path of save dir
trial = 20  # num of trial
iter = 100  # num of training iteration
weight_list = []

Path(target).mkdir(exist_ok=True)
Path(save_dir).mkdir(exist_ok=True)

### save ARI values
weight_list = []
for i in range(0,310,10):
    weight_list.append(i)
for weight in weight_list:
    overall_ARI = np.zeros((trial, iter+1), dtype=float)
    for i in range(trial):
        for j in range(iter+1):
            with open(target+str(weight)+"_weight/"+str(f"{i+1:0>2}")+"/"+str(j)+"/confutionmat.txt", encoding='utf-8') as f:
                overall_ARI[i][j] = float(f.readline())
    np.save(target+"ARI_"+str(weight), np.array(overall_ARI))

weight_list = []
for i in range(20,320,20):  # setting of word weight range
    weight_list.append(i)

def trial_iter_ARI(target_path, flag):
    overall_ARI = np.zeros((trial, iter), dtype=float)
    for i in range(trial):
        for j in range(iter):
            if flag==1:
                with open(target_path+str(f"{i+1:0>2}")+"/MLDA_result/"+str(f"{j+1:0>3}")+"/confutionmat.txt", encoding='utf-8') as f:
                    overall_ARI[i][j] = float(f.readline())
            else:
                with open(target_path+str(f"{i+1:0>2}")+"/MLDA_result/"+str(j)+"/confutionmat.txt", encoding='utf-8') as f:
                    overall_ARI[i][j] = float(f.readline())
    return overall_ARI


### plot overall result
def plot_weight_sorted_cat_trans_ARI():
    plt.clf()
    plt.ylim(0, 1.1)
    for s, weight in enumerate(weight_list):
        overall_ARI = np.zeros((trial, iter+1), dtype=float)
        # ground_truth_ARI = np.full((iter), 0.875, dtype=float)
        for i in range(trial):
            for j in range(iter+1):
                with open(f"{target}{weight}_weight/{i+1:0>2}/{j}/confutionmat.txt", encoding='utf-8') as f:
                    overall_ARI[i][j] = float(f.readline())
        plt.errorbar(range(iter+1), overall_ARI.mean(axis=0), yerr=overall_ARI.std(axis=0), label=weight, color=cm.jet(s/len(weight_list)),alpha=0.5)
        if weight <= 200:
            np.savetxt(f"{target}csv_files/{weight}_cat_mean_ARI.csv", overall_ARI.mean(axis=0)[0:10], delimiter=',', fmt="%.3f")
            np.savetxt(f"{target}csv_files/{weight}_cat_std_ARI.csv", overall_ARI.std(axis=0)[0:10], delimiter=',', fmt="%.3f")
    # plt.errorbar(range(iter), ground_truth_ARI, label="Ground truth", color="gray")
    plt.xlabel("Iteration of word acquisition")
    plt.ylabel("Categorization ARI")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(target+"MLDA_ARI.eps", bbox_inches="tight")
    plt.savefig(target+"MLDA_ARI.png", bbox_inches="tight")

### plot each weight result
# for weight in weight_list:
#     overall_ARI = np.zeros((trial, iter+1), dtype=float)
#     for i in range(trial):
#         for j in range(iter+1):
#             with open(target+str(weight)+"_weight/"+str(f"{i+1:0>2}")+"/"+str(j)+"/confutionmat.txt", encoding='utf-8') as f:
#                 overall_ARI[i][j] = float(f.readline())
#     # np.save("MLDA_result_1128", np.array(overall_ARI))
#     plt.clf()
#     plt.ylim(0, 1.1)
#     plt.errorbar(range(iter+1), overall_ARI.mean(axis=0), yerr=overall_ARI.std(axis=0), label=weight)
#     plt.xlabel("Iteration of word segmentation")
#     plt.ylabel("Categorization ARI")
#     plt.title(f"Transition of the categorization ARI (weight: {weight})")
#     plt.savefig(target+"MLDA_ARI_"+str(weight)+".png")

### Compare consist and variable weight values
# weight_change = trial_iter_ARI(target,1)
# weight_consist = trial_iter_ARI("./RESULTS/1128/",0)
# npbdaa_target = "./MLDA_test_dir/1209_npbdaa/"
# plt.clf()
# plt.ylim(0, 1.1)
# ground_truth_ARI = np.full((iter), 0.875, dtype=float)
# ex_lang = np.full((iter), 0.625, dtype=float)
# npbdaa_ARI = np.zeros((trial, iter), dtype=float)
# for i in range(trial):
#     for j in range(iter):
#         with open(npbdaa_target+"100_weight/"+str(f"{i+1:0>2}")+"/"+str(j)+"/confutionmat.txt", encoding='utf-8') as f:
#             npbdaa_ARI[i][j] = float(f.readline())
# # print(f"{npbdaa_ARI}\nlen: {len(npbdaa_ARI)}\nlen2: {len(npbdaa_ARI[0])}")
# plt.errorbar(range(iter), npbdaa_ARI.mean(axis=0), yerr=npbdaa_ARI.std(axis=0), label="NPB-DAA", alpha=0.5)
# plt.errorbar(range(iter), weight_change.mean(axis=0), yerr=weight_change.std(axis=0), label="Variable weight", alpha=0.5)
# plt.errorbar(range(iter), weight_consist.mean(axis=0), yerr=weight_consist.std(axis=0), label="Consistent weight", alpha=0.5)
# plt.errorbar(range(iter), ground_truth_ARI, label="Ground truth", color="red")
# plt.errorbar(range(iter), ex_lang, label="excluded word cue", color="gray")
# plt.legend(loc='best')
# plt.xlabel("Iteration of word segmentation")
# plt.ylabel("Categorization ARI")
# plt.title("Transition of the categorization ARI")
# plt.savefig(save_dir+"MLDA_ARI.png")

### Plot mean ARI of last iteration (NPB-DAA based)
def plot_last_ARI_npbdaa_base():
    plt.clf()
    plt.ylim(0,1.1)
    bar_weight = []
    for i in range(0,310,10):   # setting of word weight value
        bar_weight.append(i)
    # print(f"Bar\'s weight: {bar_weight}")
    mean_ARI = []
    std_ARI = []
    for weight in bar_weight:
        ARIs = np.load(target+"ARI_"+str(weight)+".npy","r")
        last_ARI = np.zeros((trial),dtype=float)
        for i in range(trial):
            last_ARI[i] = ARIs[i][-1]
        mean_ARI.append(last_ARI.mean())
        std_ARI.append(last_ARI.std())
        print(f"{weight} & {last_ARI.mean():.3f} pm {last_ARI.std():.3f}")

    plt.bar(bar_weight, mean_ARI, yerr=std_ARI, color="lightblue", width=10, edgecolor="gray")
    plt.xlabel("Weight value of language cue")
    plt.ylabel("Categorization ARI")
    plt.savefig(target+"last_ARI.eps", bbox_inches="tight")
    plt.savefig(target+"last_ARI.png", bbox_inches="tight")

### Plot ARI of last iteration (only MLDA)
def plot_last_ARI_MLDA_base():
    plt.clf()
    bar_weight = []
    for i in range(0,310,10):   # setting of word weight value
        bar_weight.append(i)
    ARI = np.zeros((len(bar_weight)))
    for i, w in enumerate(bar_weight):
        with open(target+str(w)+"_weight/model/confutionmat.txt", encoding='utf-8') as f:
            ARI[i] = float(f.readline())
    plt.bar(bar_weight, ARI, color="lightblue", width=10, edgecolor="gray")
    plt.xlabel("Weight value of language cue")
    plt.ylabel("Categorization ARI")
    plt.ylim(0, 1.1)
    plt.savefig(target+"last_ARI.eps")
    plt.savefig(target+"last_ARI.png")
    np.save(target+"last_ARIs", np.array(ARI))

### you can set multiple comparative method as you like
def proposed_based_comp_last_ARI():
    height_data = []
    yerr_data = []

    # ###ground truth + syn (weight: 100)
    # with open("./MLDA_test_dir/truth_word_synco/100_weight/model/confutionmat.txt", encoding='utf-8') as f:
    #     ARI = float(f.readline())
    # height_data.append(ARI)
    # yerr_data.append(float(0.0))
    # ### NPB-DAA + syn (weight: 100)
    # all_ARI = np.load("./MLDA_test_dir/1209_npbdaa_synco/ARI_100.npy", 'r')
    # ARI = []
    # for i in range(20):
    #     ARI.append(all_ARI[i][-1])
    # height_data.append(np.array(ARI).mean())
    # yerr_data.append(np.array(ARI).std())
    # ### Proposed + syn (weight: 100)    
    # ARI = []
    # for i in range(20):
    #     with open("./RESULTS/0107_synco/"+str(f"{i+1:0>2}")+"/MLDA_result/099/confutionmat.txt", encoding='utf-8') as f:
    #         ARI.append(float(f.readline()))
    # height_data.append(np.array(ARI).mean())
    # yerr_data.append(np.array(ARI).std())

    ### Ground truth + real (weight:  200)
    with open("./MLDA_test_dir/truth_word_real/200_weight/model/confutionmat.txt", encoding='utf-8') as f:
        ARI = float(f.readline())
    height_data.append(ARI)
    yerr_data.append(0.0)
    ### NPB-DAA + real (weight: 200)
    all_ARI = np.load("./MLDA_test_dir/1209_npbdaa/ARI_200.npy", 'r')
    ARI = []
    for i in range(trial):
        ARI.append(all_ARI[i][-1])
    height_data.append(np.array(ARI).mean())
    yerr_data.append(np.array(ARI).std())
    ### Proposed + real (weight: 200)
    # ARI = []
    # for i in range(trial):
    #     with open(f"./EX_workspace/RESULTS/0128_real_200/{i+1:0>2}/log.txt", encoding='utf-8') as f:
    #         log_list = f.readlines()
    #     with open(f"./EX_workspace/RESULTS/0128_real_200/{i+1:0>2}/MLDA_result/100/{log_list[-2][0]}/confutionmat.txt", encoding='utf-8') as f:
    #         ARI.append(float(f.readline()))
    # height_data.append(np.array(ARI).mean())
    # yerr_data.append(np.array(ARI).std())
    ### Proposed + real (weight: 10)
    # ARI = []
    # for i in range(trial):
    #     with open(f"./RESULTS/0128_real_10/{i+1:0>2}/log.txt", encoding='utf-8') as f:
    #         log_list = f.readlines()
    #     with open(f"./RESULTS/0128_real_10/{i+1:0>2}/MLDA_result/100/{log_list[-2][0]}/confutionmat.txt", encoding='utf-8') as f:
    #         ARI.append(float(f.readline()))
    # height_data.append(np.array(ARI).mean())
    # yerr_data.append(np.array(ARI).std())
    ### Proposed + real (weight: 0-200)
    ARI = []
    for i in range(trial):
        with open(f"./RESULTS/0217_real_0_40_200/{i+1:0>2}/log.txt", encoding='utf-8') as f:
            log_list = f.readlines()
        with open(f"./RESULTS/0217_real_0_40_200/{i+1:0>2}/MLDA_result/100/{log_list[-2][0]}/confutionmat.txt", encoding='utf-8') as f:
            ARI.append(float(f.readline()))
    height_data.append(np.array(ARI).mean())
    yerr_data.append(np.array(ARI).std())

    # ### 1 modal version
    # x_label = ["vision", "haptic", "audio"]
    # height_data = []
    # for l in x_label:
    #     with open(f"./MLDA_test_dir/comparative_ex/1modal/{l}/model/confutionmat.txt", encoding='utf-8') as f:
    #         height_data.append(float(f.readline()))

    x = np.array(range(len(height_data)))
    # x_label = ["Vision", "Haptic", "Audio"]
    x_label = ["Ground truth\n(200)", "NPB-DAA\n(200)", "Proposed\n(0~200)"]
    for i in range(len(x)):
        print(f"{x_label[i]}: {height_data[i]:.3f} pm {yerr_data[i]:.3f}")
    plt.clf()
    plt.ylim(0,1.1)
    plt.bar(x, height_data, yerr=yerr_data)
    plt.xticks(x,x_label)
    plt.ylim(0, 1.1)
    plt.ylabel("Categorization ARI")
    plt.savefig(save_dir+"comp_last_cat_ARI_0301.eps", bbox_inches="tight")
    plt.savefig(save_dir+"comp_last_cat_ARI_0301.png", bbox_inches="tight")

# plot_weight_sorted_cat_trans_ARI()
# plot_last_ARI_npbdaa_base()
# plot_last_ARI_MLDA_base()
proposed_based_comp_last_ARI()
