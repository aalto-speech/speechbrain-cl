#!/bin/bash

# Credits Tamas Grosz: https://github.com/GrosyT/ASR-significance-test/blob/master/run_sign_tests.sh
# Modified by Georgios Karakasidis to cover our speechbrain recipes.

# [ -f ./path.sh ] && . ./path.sh

if [ $# -le 2 ]; then
  echo "Usage: $0 ref-text hyp-text-1 hyp-text-2 ... hyp-text-N <-o OUTPUT_DIR>"
  echo "Performs significance tests using the transcript (ref-text) and the decoder outputs (hyp-text-*)"
  echo "All text files must be in trn format"
  echo "Example line from a trn file:  my favourite sport is volleyball and judo @em @voice and @m @ns  (1010103_en_22_20_102)"
  exit 1;
fi

POSITIONAL_ARGS=()
OUT_ARGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -o|--output-dir)
      OUT_ARGS="$OUT_ARGS $2"
      shift
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters  

ref="$1"
shift
files=""
for var in "${@}"
do
   #run sclite
   sclite -i swb -r ${ref}  -h ${var} -o sgml
   files="$files ${var}.sgml"
done
cat ${files} | sc_stats -p -t std4 -v -u -n significance_report $OUT_ARGS
