# How to run

### Prerequisites
At first, make sure you use linux operating system. Code is test on ubuntu 16.04.
Make sure openmpi is installed on your system.

### Make
Run following command: `cmake . && make` and you can get the executable program.

### Run
Currentely, we can run it on single machine by `mpiexec -n 3 ./dis_ffm`, which you will use one process as a server and the other two processes are used for training.
Of course, you can run the program on multiple machines.


