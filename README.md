# HDP-HLM+MLDA (Co-occurrence DAA)

## Docker image
https://drive.google.com/drive/folders/1uDxyXEvgv8C7I4jOJ9vF5Vtm9Htms8CN?usp=sharing


# Unsupervised word/phoneme discovery method using co-occurrence cue integrated by NPB-DAA and MLDA

The execution environment is saved as a Docker image.
First, please make sure that Docker is available.

```
docker help
```
If there is no error, Docker should be available.


# Building the execution environment
1. Import a docker image in tar.gz format
```
docker load < execution_env_docker_image.tar.gz
```

2. Create a container from an imported image
```
docker run -d -it --name $(container_name) hiroaki_murakami_npbdaa-mlda
```

3. Placement of sample data required for execution
Copy the entire contents of the "sample_data" directory to the "int" directory in the container.
```
docker cp sample_data/* $(container_name):root/int/
```
From here on, please work in the directory "int" in the container you created.
```
docker attach $(container_name)
cd ~/int
```

4. Compiling MLDA
```
make
```
note: You may get a warning message, but please proceed.

Testing MLDA
```
./mlda -learn -config lda_config.json
```

5. Hyperparameter application of NPP-DAA
```
python unroll_default_config.py
```

# Execution Procedure
1. Cleaning up the previous execution results, etc.
```
bash clean.sh
```

2. Running a shell script
```
bash runner.sh -l $(directory_name)
```
The results of the experiment are stored in ". /RESULTS/<directory_name>".

note: In the case of the default of 20 attempts, it will take about 3 days to complete the execution, so it is recommended to run it on a virtual terminal such as tmux.

+ If you want to run the NPB-DAA by itself
```
bash runner_py.sh -l $(directory_name)
```

# Change parameter settings and data
+ Hyperparameters of NPB-DAA  
See "~/int/hypparams/defaults.config"  
+ Setting of MLDA  
See "~/int/lda_config.json". The details are described in Nakamura's github. "https://github.com/naka-tomo/LightMLDA"  
+ Other parameters (number of HDP-HLM candidates, number of categories in MLDA, number of utterances for each object, etc.)  
See "~/int/integrated.py". The details are described in the comments.
+ Setting word modality weights for categorization  
Can be set by the function "word_weight_set()" in "integrated.py".
+ Textfile of MFCC  
Place in the "~/int/DATA" directory
+ Label files for phonemes and words  
Place the file in the "~/int/LABEL" directory.
+ Name of each file  
It is described in "~/int/files.txt".

# Directory structure and its contents
Directories other than int under root are used for building the environment and are not directly related to this program, so they are omitted.

The components of MLDA are also omitted.

+ CAND*: Directory to store information on * HDP-HLM candidates.
    + Candidates: Directory where all * candidates are stored as pickle files.
    + Chosen: Directory where the selected HDP-HLM candidate is saved as a pickle file when one of the * candidates is selected according to the respective weight.
+ DATA: Directory where the MFCC files are located.
+ LABEL: Directory where phoneme and word label files are placed.
+ MLDA_result: Directory that stores the results of MLDA runs using each HDP-HLM word sequence candidate for each iteration.
+ RESULTS: Directory where the experimental results of each trial are saved when run with "runner.sh".
+ Saved: Directory to store data during execution when "integrated.py" stops in the middle of execution.
+ cand_results: Directory for storing the results of splitting each HDP-HLM candidate in each iteration.
+ hypparams: Directory where the files describing the hyperparameters of the NPB-DAA are placed.
+ mlda_data: Directory where each histogram used in MLDA is placed.
    + word_hist_candies: Directory that stores the word sequences estimated by each HDP-HLM candidate for each iteration.
+ model: Directory where MLDA execution results are saved.
+ sampled_z_lnsj: Directory containing a pickle file that stores the category assigned to the j-th word in the s-th utterance of the n-th object in the l-th candidate in each iteration.
+ eval_src: Directory with source code for evaluating the results of the experiment
    + The content of each source code is described in comments.
    + summary_runner.sh: A shell script to calculate word segmentation ARI, etc. on the NPB-DAA side. The details are described in Ozaki's git "https://github.com/EmergentSystemLabStudent/NPB_DAA".
        + summary_and_plot.py, summary_and_plot_light.py, summary_summary.py

# Reference
Akira Taniguchi, Hiroaki Murakami, Ryo Ozaki, Tadahiro Taniguchi, "Unsupervised Multimodal Word Discovery based on Double Articulation Analysis with Co-occurrence cues", arXiv, 2022. 
https://arxiv.org/abs/2201.06786
