  #!/bin/bash"
  pre=0
  while getopts p: OPT
  do
    case $OPT in
      "p" ) pre="${OPTARG}" ;;
    esac
  done
bash runner.sh -l log1009 -b 20 -e 20
