#include "mpi.h"
#include "ffm.h"
#include <sys/time.h>
#include <sys/sysinfo.h>

int main(int argc, char *argv[]) {





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
	FFMParameter parameter;
	parameter.num_factors = 5,
	parameter.init_learning_rate = 0.01,
	parameter.num_iter = 50 / (group_size -1),
	parameter.mini_batch_size = 1024;


	MPIConf conf;
	conf.server_id = 0;
	conf.group_size = group_size;
	conf.rank = rank;
	//MPI_COMM_WORLD
	//conf.mpi_world = MPI_COMM_WORLD;
	struct timeval start,end,result;
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
		gettimeofday(&start,NULL);
		scheduler.run();
	}

	MPI_Finalize();
	gettimeofday(&end,NULL);
	if(rank!=0) {
		timersub(&end,&start,&result);
		printf("My rank:%d. Time cost:%d.%ds\n", rank, result.tv_sec, (result.tv_usec) / 1000);
	}

	return 0;
}
