#!/bin/bash

echo "Running script 1 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/2x -p batches_per_epoch=20 -p epochs=500 -p model=rdn -p C=1 -p D=2 -p G=4 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=2 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 2 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/3x -p batches_per_epoch=20 -p epochs=500 -p model=rdn -p C=1 -p D=2 -p G=4 -p G0=64 -p lr=0.0004 -p hr_patch_size=129 -p scale=3 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 3 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/4x -p batches_per_epoch=20 -p epochs=500 -p model=rdn -p C=1 -p D=2 -p G=4 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=4 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 4 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/8x -p batches_per_epoch=20 -p epochs=500 -p model=rdn -p C=1 -p D=2 -p G=4 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=8 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 5 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/2x -p batches_per_epoch=20 -p epochs=500 -p model=rdn -p C=1 -p D=10 -p G=64 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=2 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 6 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/3x -p batches_per_epoch=20 -p epochs=500 -p model=rdn -p C=1 -p D=10 -p G=64 -p G0=64 -p lr=0.0004 -p hr_patch_size=129 -p scale=3 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 7 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/4x -p batches_per_epoch=20 -p epochs=500 -p model=rdn -p C=1 -p D=10 -p G=64 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=4 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 8 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/8x -p batches_per_epoch=20 -p epochs=500 -p model=rdn -p C=1 -p D=10 -p G=64 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=8 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 9 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/2x -p batches_per_epoch=20 -p epochs=500 -p model=rrdn -p C=4 -p D=3 -p G=32 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=2 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 10 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/3x -p batches_per_epoch=20 -p epochs=500 -p model=rrdn -p C=4 -p D=3 -p G=32 -p G0=64 -p lr=0.0004 -p hr_patch_size=129 -p scale=3 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 11 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/4x -p batches_per_epoch=20 -p epochs=500 -p model=rrdn -p C=4 -p D=3 -p G=32 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=4 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1
echo "Running script 12 of 12"
../../bronco/bin/dev train --showLogs -a -e local --verbose -i /mnt/castellan/data/div2k-processed/preprocessed/8x -p batches_per_epoch=20 -p epochs=500 -p model=rrdn -p C=4 -p D=3 -p G=32 -p G0=64 -p lr=0.0004 -p hr_patch_size=128 -p scale=8 -p dataset=div2k -p compression_quality=50 -p vary_compression_quality=1