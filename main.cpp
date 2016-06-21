#include "mpi.h"
#include "ffm.h"


int main(int argc, char *argv[]) {


	FFMParameter parameter;
	parameter.num_factors = 3,
	parameter.init_learning_rate = 0.01,
	parameter.num_iter = 50,
	parameter.mini_batch_size = 1024;


	int group_size;
	int rank;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&group_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	if(group_size < 2){
		fprintf(stderr,"You need at least two server to run.\n");
		return 1;
	}
	MPIConf conf;
	conf.server_id = 0;
	conf.group_size = group_size;
	conf.rank = rank;
	//MPI_COMM_WORLD
	//conf.mpi_world = MPI_COMM_WORLD;

	if(rank == 0){
		DataSet dataSet;
		dataSet.read("input.txt");
		printf("Read Finished\n");
		FFMScheduler scheduler(parameter,conf, &dataSet);
		scheduler.partition();
		scheduler.server_serve();
	}else{
		FFMScheduler scheduler(parameter, conf,NULL);
		scheduler.recv_partition();
		scheduler.run();
	}

	MPI_Finalize();

	return 0;
}
