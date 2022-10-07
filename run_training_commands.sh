#!/bin/bash

echo "Running script 1 of 2"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/ffhq/preprocessed/2x -p batches_per_epoch=20 -p epochs=500 -p model=rdn -p C=1 -p D=2 -p G=4 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=2 -p dataset=ffhq
echo "Running script 2 of 2"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-flickr2k/preprocessed/2x -p batches_per_epoch=20 -p epochs=500 -p model=rdn -p C=1 -p D=2 -p G=4 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=2 -p dataset=div2k-flickr2k