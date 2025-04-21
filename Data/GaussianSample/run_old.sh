#PBS -l nodes=1:ppn=28
#PBS -j oe
#PBS -q gamma
#PBS -l walltime=672:00:00
nprocs=`cat $PBS_NODEFILE | wc -l`

#source /etc/profile.d/dawning.sh
source ~/g16-env.sh
#mpirun -np $nprocs -machinefile $PBS_NODEFILE ./SCFT >>log.dat
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

