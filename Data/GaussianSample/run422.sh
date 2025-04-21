#PBS -l nodes=1:ppn=40
#PBS -j oe
#PBS -q amd
#PBS -l walltime=168:00:00
nprocs=40

cd $PBS_O_WORKDIR
#source /etc/profile.d/dawning.sh
module load g16/C01

#mpirun -np $nprocs -machinefile $PBS_NODEFILE ./SCFT >>log.dat

files=`ls *.gjf`
for file in $files
do
    input=$file
    output=${file::(-4)}".log"
    g16 <$input >$output
done

module unload g16/C01
