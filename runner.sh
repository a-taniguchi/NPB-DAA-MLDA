#!/bin/bash

label=TEST
begin=1
end=20
cand=10
pre=0
CONTINUE=false

while getopts l:b:e:c:p: OPT
do
  case $OPT in
    "l" ) label="${OPTARG}" ;;
    "b" ) begin="${OPTARG}" ;;
    "e" ) end="${OPTARG}" ;;
    "c" ) cand="${OPTARG}" ;;
    "p" ) pre="${OPTARG}" ;;
  esac
done

if [ ${pre} -ne 0 ] ; then
  echo "continue mode"
  CONTINUE=true
fi

mkdir -p RESULTS
mkdir -p RESULTS/${label}
mkdir -p results
mkdir -p parameters
mkdir -p summary_files
mkdir -p CAND${cand}
mkdir -p model
mkdir -p cand_results

cp hypparams/defaults.config RESULTS/${label}

for i in `seq ${begin} ${end}`
do
  echo ${i}

  i_str=$( printf '%02d' $i )
  rm -f results/*
  rm -f parameters/*
  rm -f summary_files/*
  rm -rf CAND${cand}/*
  rm -rf cand_results/*
  rm -rf MLDA_result/*
  rm -rf sampled_z_lnsj/*

  cat <<EOF >continue.sh
  #!/bin/bash"
  pre=0
  while getopts p: OPT
  do
    case \$OPT in
      "p" ) pre="\${OPTARG}" ;;
    esac
  done
EOF

  if "${CONTINUE}" ; then
    echo "bash runner.sh -l ${label} -b ${i} -e ${end} -p \${pre}" >>continue.sh
    python3 integrated.py --cont ${pre} | tee log.txt
  else
    echo "bash runner.sh -l ${label} -b ${i} -e ${end}" >>continue.sh
    python3 integrated.py | tee log.txt
  fi

  mkdir -p RESULTS/${label}/${i_str}/
  cp -r results/ RESULTS/${label}/${i_str}/
  cp -r parameters/ RESULTS/${label}/${i_str}/
  cp -r summary_files/ RESULTS/${label}/${i_str}/
  cp log.txt RESULTS/${label}/${i_str}/
  cp -r CAND${cand}/ RESULTS/${label}/${i_str}/
  cp -r cand_results/ RESULTS/${label}/${i_str}/
  cp -r mlda_data/word_hist_candies/ RESULTS/${label}/${i_str}/
  cp -r MLDA_result/ RESULTS/${label}/${i_str}/
  cp -r word_hist_result/ RESULTS/${label}/${i_str}/
  cp -r sampled_z_lnsj/ RESULTS/${label}/${i_str}/

  CONTINUE=false
done

cp -f hypparams/defaults.config RESULTS/${label}
rm -f results/*
rm -f parameters/*
rm -f summary_files/*
rm -rf CAND${cand}/*
rm -rf cand_results/*
rm -rf sampled_z_lnsj/*
rm -rf MLDA_result/*
rm -rf word_hist_result/*
rm -rf mlda_data/word_hist_candies/*
