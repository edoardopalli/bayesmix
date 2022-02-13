#!/usr/bin/env bash

build/run_telescop\
 --algo-params-file resources/prova1_tele/algo.asciipb --hier-type NNW --hier-args resources/prova1_tele/nnw_nniw.asciipb --mix-type MFM --mix-args resources/prova1_tele/mfm_bnb.asciipb --coll-name resources/prova1_tele/out/chains.recordio --data-file resources/prova1_tele/faithful.csv --grid-file resources/prova1_tele/faithful_grid.csv --dens-file resources/prova1_tele/out/density_file.csv --n-cl-file resources/prova1_tele/out/numclust.csv --clus-file resources/prova1_tele/out/clustering.csv --best-clus-file resources/prova1_tele/out/best_clustering.csv
