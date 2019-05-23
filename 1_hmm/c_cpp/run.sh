#!/bin/bash

make clean
make all
for i in `seq 1 5`;do ./train 10 ../model_init.txt ../seq_model_0$i.txt model_0$i.txt;done
./test modellist.txt ../testing_data1.txt result.txt -a
