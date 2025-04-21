#PBS -l nodes=1:ppn=32
#PBS -j oe
#PBS -q Frog
#PBS -l walltime=168:00:00
nprocs=`cat $PBS_NODEFILE | wc -l`

source ~/g16-c01avx2-env.sh
#source /etc/profile.d/dawning.sh
cd $PBS_O_WORKDIR
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

