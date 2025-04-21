#!/bin/bash
#SBATCH -J g16
#SBATCH -N 1
#SBATCH --ntasks-per-node=28
#SBATCH -p hfacnormal01

export GAUSS_SCRDIR=/public/home/jackie_lijiaqi/g16/scratch
export g16root=/public/home/jackie_lijiaqi
source /public/home/jackie_lijiaqi/g16/bsd/g16.profile

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
