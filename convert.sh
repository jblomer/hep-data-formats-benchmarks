#!/usr/bin/env sh

set -e

rm -f data/*.orc
rm -f data/*.parquet

############################# LHCB #############################

# Without default write opts
python convert.py DecayTree data/B2HHH.root data/B2HHH.parquet -o parquet
python convert.py DecayTree data/B2HHH.root data/B2HHH.orc -o orc

# With RNTuple write opts mirrored
python convert.py DecayTree data/B2HHH.root data/B2HHH_ntplcfg.parquet -o parquet -m
python convert.py DecayTree data/B2HHH.root data/B2HHH_ntplcfg.orc -o orc -m

############################# CMS ##############################

# Without default write opts
python convert.py Events data/ttjet_signed.root data/ttjet_signed.parquet -o parquet
python convert.py Events data/ttjet_signed.root data/ttjet_signed.orc -o orc

# With RNTuple write opts mirrored
python convert.py Events data/ttjet_signed.root data/ttjet_signed_ntplcfg.parquet -o parquet -m
python convert.py Events data/ttjet_signed.root data/ttjet_signed_ntplcfg.orc -o orc -m
