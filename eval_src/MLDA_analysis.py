### you can plot the categorization accuracy and ARI comparing multiple method
### "confutionmat.txt" shows categorization accuracy (ACC).
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from pathlib import Path
import seaborn as sns

args = sys.argv # recive commadn line 

target   = args[1] #"./RESULTS/" + args[1]  # "./MLDA_test_dir/1209_npbdaa/" # path of target dir
save_dir = "./RESULTS/figures/" # "./MLDA_test_dir/comparative_ex/tcds_1/" # path of save dir
trial = 20  # num of trial
iter = 100  # num of training iteration
print("trial",trial)
weight_list = []

Path(target).mkdir(exist_ok=True)
Path(save_dir).mkdir(exist_ok=True)

""" # deleted by akira
### save ACC values
weight_list = []
for i in range(0,310,10):
    weight_list.append(i)
for weight in weight_list:
    overall_ACC = np.zeros((trial, iter+1), dtype=float)
    for i in range(trial):
        for j in range(iter+1):
            with open(target+str(weight)+"_weight/"+str(f"{i+1:0>2}")+"/"+str(j)+"/confutionmat.txt", encoding='utf-8') as f:
                overall_ACC[i][j] = float(f.readline())
    np.save(target+"ACC_"+str(weight), np.array(overall_ACC))

weight_list = []
for i in range(20,320,20):  # setting of word weight range
    weight_list.append(i)
"""

def trial_iter_ACC(target_path, flag):
    overall_ACC = np.zeros((trial, iter), dtype=float)
    for i in range(trial):
        for j in range(iter):
            if flag==1:
                with open(target_path+str(f"{i+1:0>2}")+"/MLDA_result/"+str(f"{j+1:0>3}")+"/confutionmat.txt", encoding='utf-8') as f:
                    overall_ACC[i][j] = float(f.readline())
            else:
                with open(target_path+str(f"{i+1:0>2}")+"/MLDA_result/"+str(j)+"/confutionmat.txt", encoding='utf-8') as f:
                    overall_ACC[i][j] = float(f.readline())
    return overall_ACC


### plot overall result
def plot_weight_sorted_cat_trans_ACC():
    plt.clf()
    plt.ylim(0, 1.1)
    for s, weight in enumerate(weight_list):
        overall_ACC = np.zeros((trial, iter+1), dtype=float)
        # ground_truth_ACC = np.full((iter), 0.875, dtype=float)
        for i in range(trial):
            for j in range(iter+1):
                with open(f"{target}{weight}_weight/{i+1:0>2}/{j}/confutionmat.txt", encoding='utf-8') as f:
                    overall_ACC[i][j] = float(f.readline())
        plt.errorbar(range(iter+1), overall_ACC.mean(axis=0), yerr=overall_ACC.std(axis=0), label=weight, color=cm.jet(s/len(weight_list)),alpha=0.5)
        if weight <= 200:
            np.savetxt(f"{target}csv_files/{weight}_cat_mean_ACC.csv", overall_ACC.mean(axis=0)[:], delimiter=',', fmt="%.3f")
            np.savetxt(f"{target}csv_files/{weight}_cat_std_ACC.csv", overall_ACC.std(axis=0)[:], delimiter=',', fmt="%.3f")
    # plt.errorbar(range(iter), ground_truth_ACC, label="Ground truth", color="gray")
    plt.xlabel("Iteration of word acquisition")
    plt.ylabel("Categorization ACC")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.savefig(target+"MLDA_ACC.eps", bbox_inches="tight")
    plt.savefig(target+"MLDA_ACC.png", bbox_inches="tight")

### plot each weight result
# for weight in weight_list:
#     overall_ACC = np.zeros((trial, iter+1), dtype=float)
#     for i in range(trial):
#         for j in range(iter+1):
#             with open(target+str(weight)+"_weight/"+str(f"{i+1:0>2}")+"/"+str(j)+"/confutionmat.txt", encoding='utf-8') as f:
#                 overall_ACC[i][j] = float(f.readline())
#     # np.save("MLDA_result_1128", np.array(overall_ACC))
#     plt.clf()
#     plt.ylim(0, 1.1)
#     plt.errorbar(range(iter+1), overall_ACC.mean(axis=0), yerr=overall_ACC.std(axis=0), label=weight)
#     plt.xlabel("Iteration of word segmentation")
#     plt.ylabel("Categorization ACC")
#     plt.title(f"Transition of the categorization ACC (weight: {weight})")
#     plt.savefig(target+"MLDA_ACC_"+str(weight)+".png")

### Compare consist and variable weight values
# weight_change = trial_iter_ACC(target,1)
# weight_consist = trial_iter_ACC("./RESULTS/1128/",0)
# npbdaa_target = "./MLDA_test_dir/1209_npbdaa/"
# plt.clf()
# plt.ylim(0, 1.1)
# ground_truth_ACC = np.full((iter), 0.875, dtype=float)
# ex_lang = np.full((iter), 0.625, dtype=float)
# npbdaa_ACC = np.zeros((trial, iter), dtype=float)
# for i in range(trial):
#     for j in range(iter):
#         with open(npbdaa_target+"100_weight/"+str(f"{i+1:0>2}")+"/"+str(j)+"/confutionmat.txt", encoding='utf-8') as f:
#             npbdaa_ACC[i][j] = float(f.readline())
# # print(f"{npbdaa_ACC}\nlen: {len(npbdaa_ACC)}\nlen2: {len(npbdaa_ACC[0])}")
# plt.errorbar(range(iter), npbdaa_ACC.mean(axis=0), yerr=npbdaa_ACC.std(axis=0), label="NPB-DAA", alpha=0.5)
# plt.errorbar(range(iter), weight_change.mean(axis=0), yerr=weight_change.std(axis=0), label="Variable weight", alpha=0.5)
# plt.errorbar(range(iter), weight_consist.mean(axis=0), yerr=weight_consist.std(axis=0), label="Consistent weight", alpha=0.5)
# plt.errorbar(range(iter), ground_truth_ACC, label="Ground truth", color="red")
# plt.errorbar(range(iter), ex_lang, label="excluded word cue", color="gray")
# plt.legend(loc='best')
# plt.xlabel("Iteration of word segmentation")
# plt.ylabel("Categorization ACC")
# plt.title("Transition of the categorization ACC")
# plt.savefig(save_dir+"MLDA_ACC.png")

### Plot mean ACC of last iteration (NPB-DAA based)
def plot_last_ACC_npbdaa_base():
    plt.clf()
    plt.ylim(0,1.1)
    bar_weight = []
    for i in range(0,310,10):   # setting of word weight value
        bar_weight.append(i)
    # print(f"Bar\'s weight: {bar_weight}")
    mean_ACC = []
    std_ACC = []
    for weight in bar_weight:
        ACCs = np.load(target+"ARI_"+str(weight)+".npy","r")
        last_ACC = np.zeros((trial),dtype=float)
        for i in range(trial):
            last_ACC[i] = ACCs[i][-1]
        mean_ACC.append(last_ACC.mean())
        std_ACC.append(last_ACC.std())
        print(f"{weight} & {last_ACC.mean():.3f} pm {last_ACC.std():.3f}")

    plt.bar(bar_weight, mean_ACC, yerr=std_ACC, color="lightblue", width=10, edgecolor="gray")
    plt.xlabel("Weight value of language cue")
    plt.ylabel("Categorization ACC")
    plt.savefig(target+"last_ACC.eps", bbox_inches="tight")
    plt.savefig(target+"last_ACC.png", bbox_inches="tight")

### Plot ACC of last iteration (only MLDA)
def plot_last_ACC_MLDA_base():
    plt.clf()
    bar_weight = []
    for i in range(0,310,10):   # setting of word weight value
        bar_weight.append(i)
    ACC = np.zeros((len(bar_weight)))
    for i, w in enumerate(bar_weight):
        with open(target+str(w)+"_weight/model/confutionmat.txt", encoding='utf-8') as f:
            ACC[i] = float(f.readline())
    plt.bar(bar_weight, ACC, color="lightblue", width=10, edgecolor="gray")
    plt.xlabel("Weight value of language cue")
    plt.ylabel("Categorization ACC")
    plt.ylim(0, 1.1)
    plt.savefig(target+"last_ACC.eps")
    plt.savefig(target+"last_ACC.png")
    np.save(target+"last_ACCs", np.array(ACC))

### you can set multiple comparative method as you like
def proposed_based_comp_last_ACC():
    height_data = []
    yerr_data = []

    # ###ground truth + syn (weight: 100)
    # with open("./MLDA_test_dir/truth_word_synco/100_weight/model/confutionmat.txt", encoding='utf-8') as f:
    #     ACC = float(f.readline())
    # height_data.append(ACC)
    # yerr_data.append(float(0.0))
    # ### NPB-DAA + syn (weight: 100)
    # all_ACC = np.load("./MLDA_test_dir/1209_npbdaa_synco/ACC_100.npy", 'r')
    # ACC = []
    # for i in range(20):
    #     ACC.append(all_ACC[i][-1])
    # height_data.append(np.array(ACC).mean())
    # yerr_data.append(np.array(ACC).std())
    # ### Proposed + syn (weight: 100)    
    # ACC = []
    # for i in range(20):
    #     with open("./RESULTS/0107_synco/"+str(f"{i+1:0>2}")+"/MLDA_result/099/confutionmat.txt", encoding='utf-8') as f:
    #         ACC.append(float(f.readline()))
    # height_data.append(np.array(ACC).mean())
    # yerr_data.append(np.array(ACC).std())

    ### Ground truth + real (weight:  200)
    with open("./MLDA_test_dir/truth_word_real/200_weight/model/confutionmat.txt", encoding='utf-8') as f:
        ACC = float(f.readline())
    height_data.append(ACC)
    yerr_data.append(0.0)
    ### NPB-DAA + real (weight: 200)
    all_ACC = np.load("./MLDA_test_dir/1209_npbdaa/ARI_200.npy", 'r')
    ACC = []
    for i in range(trial):
        ACC.append(all_ACC[i][-1])
    height_data.append(np.array(ACC).mean())
    yerr_data.append(np.array(ACC).std())
    ### Proposed + real (weight: 200)
    # ACC = []
    # for i in range(trial):
    #     with open(f"./EX_workspace/RESULTS/0128_real_200/{i+1:0>2}/log.txt", encoding='utf-8') as f:
    #         log_list = f.readlines()
    #     with open(f"./EX_workspace/RESULTS/0128_real_200/{i+1:0>2}/MLDA_result/100/{log_list[-2][0]}/confutionmat.txt", encoding='utf-8') as f:
    #         ACC.append(float(f.readline()))
    # height_data.append(np.array(ACC).mean())
    # yerr_data.append(np.array(ACC).std())
    ### Proposed + real (weight: 10)
    # ACC = []
    # for i in range(trial):
    #     with open(f"./RESULTS/0128_real_10/{i+1:0>2}/log.txt", encoding='utf-8') as f:
    #         log_list = f.readlines()
    #     with open(f"./RESULTS/0128_real_10/{i+1:0>2}/MLDA_result/100/{log_list[-2][0]}/confutionmat.txt", encoding='utf-8') as f:
    #         ACC.append(float(f.readline()))
    # height_data.append(np.array(ACC).mean())
    # yerr_data.append(np.array(ACC).std())
    ### Proposed + real (weight: 0-200)
    ACC = []
    for i in range(trial):
        with open(f"./RESULTS/0217_real_0_40_200/{i+1:0>2}/log.txt", encoding='utf-8') as f:
            log_list = f.readlines()
        with open(f"./RESULTS/0217_real_0_40_200/{i+1:0>2}/MLDA_result/100/{log_list[-2][0]}/confutionmat.txt", encoding='utf-8') as f:
            ACC.append(float(f.readline()))
    height_data.append(np.array(ACC).mean())
    yerr_data.append(np.array(ACC).std())

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
    plt.ylabel("Categorization ACC")
    plt.savefig(save_dir+"comp_last_cat_ACC.eps", bbox_inches="tight")
    plt.savefig(save_dir+"comp_last_cat_ACC.png", bbox_inches="tight")

# added by akira
def print_ACC():
    ACC = []
    for i in range(trial):
        with open(target+f"/{i+1:0>2}/log.txt", encoding='utf-8') as f:
            log_list = f.readlines()
        with open(target+f"/{i+1:0>2}/MLDA_result/100/{log_list[-2][0]}/confutionmat.txt", encoding='utf-8') as f:
            ACC.append(float(f.readline()))
    mean_ACC = np.array(ACC).mean()
    std_ACC  = np.array(ACC).std()

    # Print terminal
    #print("ACC:",mean_ACC,std_ACC)
    print(f"ACC: {mean_ACC:.3f} \pm {std_ACC:.3f}")

# added by akira
def calc_last_category_ARI():
    # READ truth categorization file
    truth = np.loadtxt("./Category.txt")
    #with open("./Category.txt", encoding='utf-8') as f:
    #        truth.append(int(f.readline()))

    obj_ARI = np.zeros(trial)
    obj_NMI = np.zeros(trial)

    for i in range(trial):
        # READ chosen index in last iterarion
        with open(target+f"/{i+1:0>2}/log.txt", encoding='utf-8') as f:
            log_list = f.readlines()
        chosen_idx = log_list[-2][0]
        #print(i+1,"chosen_idx",chosen_idx)

        # READ object categorization results (chosen ones) in last iteration
        objcat = np.loadtxt(target+f"/{i+1:0>2}/MLDA_result/100/{chosen_idx}/ClassResult.txt") #[]
        #with open(target+f"/{i+1:0>2}/MLDA_result/100/{chosen_idx}/ClassResult.txt", encoding='utf-8') as f:
        #    objcat.append(int(f.readline()))
        
        # calculate ARI and NMI
        #for t in trange(trial):
        #print(truth)
        #print(objcat)
        obj_ARI[i] = adjusted_rand_score(truth, objcat)
        obj_NMI[i] = normalized_mutual_info_score(truth, objcat)

    # mean, std
    mean_ARI = np.array(obj_ARI).mean()
    std_ARI  = np.array(obj_ARI).std()
    mean_NMI = np.array(obj_NMI).mean()
    std_NMI  = np.array(obj_NMI).std()

    # Print terminal
    #print(target)
    print(f"NMI: {mean_NMI:.3f} \pm {std_NMI:.3f}")
    print(f"ARI: {mean_ARI:.3f} \pm {std_ARI:.3f}")
    
    # Write txt file
    #ARI = [mean_ARI,std_ARI]
    #NMI = [mean_NMI,std_NMI]
    np.savetxt(target+"/summary_files/Obj_ARI.txt", obj_ARI)
    np.savetxt(target+"/summary_files/Obj_NMI.txt", obj_NMI)


# plot_weight_sorted_cat_trans_ACC()
# plot_last_ACC_npbdaa_base()
# plot_last_ACC_MLDA_base()
# proposed_based_comp_last_ACC()
print_ACC() # added by akira
calc_last_category_ARI() # added by akira

