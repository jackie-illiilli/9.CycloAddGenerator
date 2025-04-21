#!/bin/bash
#SBATCH -p amd_512
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 28

export PGI_FASTMATH_CPU=sandybridge
export COM_DIR=~/software-scg9909/g16
export g16root=$COM_DIR
export PATH=$g16root:$PATH
source $g16root/bsd/g16.profile
export GAUSS_SCRDIR=/public3/home/scg9909/tmp
export GAUSS_EXEDIR=$g16root

root_dir=`pwd`
echo $root_dir
cd $root_dir
files=`ls $root_dir/*.gjf`
for file in $files
do
    input=$file
    output=${file::(-4)}".log"
    g16 <$input >$output
done