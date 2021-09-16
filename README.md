# NPB-DAA-MLDA

## Docker image
https://drive.google.com/drive/folders/1uDxyXEvgv8C7I4jOJ9vF5Vtm9Htms8CN?usp=sharing

# Unsupervised word/phoneme discovery method using co-occurrence cue integrated by NPB-DAA and MLDA

実行環境はDockerイメージとして保存してあります．
まずはDockerが使用可能であることを確認してください．
```
docker help
```
エラーが出なければDockerは使用可能のはずです．

# 実行環境構築
1. tar.gz形式のdockerイメージを取り込む
```
docker load < execution_env_docker_image.tar.gz
```
2. 取り込んだイメージからコンテナを作成
```
docker run -d -it --name $(container_name) hiroaki_murakami_npbdaa-mlda
```
3. 実行に必要なサンプルデータの配置
"sample_data"ディレクトリの中身を全てコンテナ内の"int"ディレクトリにコピーする．
```
docker cp sample_data/* $(container_name):root/int/
```
ここから先は作成したコンテナ内のディレクトリ"int"で作業してください．
```
docker attach $(container_name)
cd ~/int
```
4. MLDAのコンパイル
```
make
```
note: warning表示が出ます．
MLDAのテスト
```
./mlda -learn -config lda_config.json
```
5. NPB-DAAのハイパーパラメータ適用
```
python unroll_default_config.py
```

# 実行手順
1. 前回の実行結果等の掃除
```
bash clean.sh
```
2. シェルスクリプトの実行
```
bash runner.sh -l $(directory_name)
```
実験結果は"./RESULTS/directory_name"内に保存されます．

note: デフォルトの20回試行の場合，実行完了まで3日程度かかるので，tmux等の仮想ターミナル上で実行することをオススメします．

+ NPB-DAA単体で実行したい場合
```
bash runner_py.sh -l $(directory_name)
```

# パラメータ設定やデータの変更
+ NPB-DAAのハイパーパラメータ<br>
"~/int/hypparams/defaults.config"を参照
+ MLDAの設定<br>
"~/int/lda_config.json"を参照．詳細はなかとも先生のgitに記述してあります．"https://github.com/naka-tomo/LightMLDA"
+ その他のパラメータ（HDP-HLM候補の数，MLDAにおけるカテゴリ数，各物体に対する発話数など）<br>
"~/int/integrated.py"中に記述．詳細はコメントとして記述してあります．
+ カテゴリ分類における単語キューの重み設定<br>
"integrated.py"中の関数"word_weight_set()"で設定可能
+ MFCCのテキストファイル<br>
"~/int/DATA"ディレクトリ内に配置
+ 音素・単語のラベルファイル<br>
"~/int/LABEL"ディレクトリ内に配置．
+ 各ファイルの名称<br>
"~/int/files.txt"に記述

# ディレクトリ構造とその内容
root直下のint以外のディレクトリについては環境構築のために用いているもので，本プログラムと直接関係はないため省略する．

また，MLDAの構成要素についても省略する．

+ CAND*: *個のHDP-HLM候補の情報を保存するディレクトリ
    + Candidates: *個の候補全てがpickleファイルとして保存されているディレクトリ
    + Chosen: *個の候補から，それぞれの重みに従って1つ選択したときに選ばれたHDP-HLM候補がpickleファイルとして保存されているディレクトリ
+ DATA: MFCCファイルを配置するディレクトリ
+ LABEL: 音素・単語ラベルファイルを配置するディレクトリ
+ MLDA_result: 各イテレーションの各HDP-HLMの単語列候補を用いたMLDAの実行結果を保存するディレクトリ
+ RESULTS: "runner.sh"で実行した場合に，各試行の実験結果を保存するディレクトリ
+ Saved: "integrated.py"が実行途中で停止した場合に実行途中のデータを保存しておくディレクトリ
+ cand_results: 各HDP-HLM候補の各イテレーションでの分割結果を保存するディレクトリ
+ hypparams: NPB-DAAのハイパーパラメータを記述するファイルを配置するディレクトリ
+ mlda_data: MLDAで用いる各ヒストグラムを配置するディレクトリ
    + word_hist_candies: 各HDP-HLM候補が推定した単語列をイテレーションごとに保存するディレクトリ
+ model: MLDAの実行結果が保存されるディレクトリ
+ sampled_z_lnsj: 各イテレーションでのl番目の候補におけるn番目の物体のs番目の発話中のj番目の単語に割り当てられたカテゴリを保存したpickleファイルが配置されるディレクトリ．
+ eval_src: 実験結果を評価するためのソースコードがあるディレクトリ
    + 各ソースコードの内容はコメントで記述してあります
    + summary_runner.sh: NPB-DAA側の単語分割ARIなどを算出するシェルスクリプト．尾崎さんのgit "https://github.com/EmergentSystemLabStudent/NPB_DAA" に詳細が記述されています．
        + summary_and_plot.py, summary_and_plot_light.py, summary_summary.pyを用いる
