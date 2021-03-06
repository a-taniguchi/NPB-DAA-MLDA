#!/bin/bash

label=sample_results
begin=1
end=20

while getopts l:b:e: OPT
do
  case $OPT in
    "l" ) label="${OPTARG}" ;;
    "b" ) begin="${OPTARG}" ;;
    "e" ) end="${OPTARG}" ;;
  esac
done

mkdir -p ${label}
cp -ru DATA/ ${label}/
cp -ru LABEL/ ${label}/
cp -ru hypparams/ ${label}/

mkdir -p results
mkdir -p parameters
mkdir -p summary_files

cp hypparams/defaults.config RESULTS/${label}

for i in `seq ${begin} ${end}`
do
  echo ${i}

  i_str=$( printf '%02d' $i )
  rm -f results/*
  rm -rf parameters/*
  rm -f summary_files/*
  rm -f log.txt
  touch log.txt

  echo "#!/bin/bash" > continue.sh
  echo "bash runner_py.sh -l ${label} -b ${i} -e ${end}" >> continue.sh

  python3 pyhlm_sample.py | tee log.txt

  mkdir -p ${label}/${i_str}/
  cp -r results/ ${label}/${i_str}/
  cp -r parameters/ ${label}/${i_str}/
  cp -r summary_files/ ${label}/${i_str}/
  cp log.txt ${label}/${i_str}/

done

rm -f results/*
rm -rf parameters/*
rm -f summary_files/*
rm -f log.txt
