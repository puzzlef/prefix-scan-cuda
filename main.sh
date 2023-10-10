#!/usr/bin/env bash
src="prefix-scan-cuda"
out="$HOME/Logs/$src$1.log"
ulimit -s unlimited
printf "" > "$out"

# Download program
if [[ "$DOWNLOAD" != "0" ]]; then
  rm -rf $src
  git clone https://github.com/puzzlef/$src
  cd $src
fi

# Fixed config
: "${MAX_THREADS:=64}"
: "${REPEAT_METHOD:=1}"
# Define macros (dont forget to add here)
DEFINES=(""
"-DMAX_THREADS=$MAX_THREADS"
"-DREPEAT_METHOD=$REPEAT_METHOD"
)

# Run
nvcc ${DEFINES[*]} -x cu -std=c++17 -O3 -Xcompiler -fopenmp main.cxx
stdbuf --output=L ./a.out 2>&1 | tee -a "$out"

# Signal completion
curl -X POST "https://maker.ifttt.com/trigger/puzzlef/with/key/${IFTTT_KEY}?value1=$src$1"
