//
// Created by syd on 16-6-20.
//

#ifndef DIS_FFM_FFM_H
#define DIS_FFM_FFM_H

#include <vector>
#include <cmath>
#include <set>
#include <cassert>
#include "data.h"


struct MPIConf {
	int server_id;
	int group_size;
	int rank;
	//int mpi_world;
};

struct FFMParameter {
	int num_factors;
	int num_iter;
	double init_learning_rate;
	int mini_batch_size;
};

class FFMModel {
public:
	FFMParameter parameter;

	svec_t<double> w;
	svec_t<vector<double>> v;

	double sigmod(double x) {
		return 1 / (exp(-x) + 1);
	}

	double f(svec_t<double> *input) {
		double res = w[0];
		for (auto pair:*input)
			res += w[pair.first] * pair.second;

		/*
		for (int k = 0; k < parameter.num_factors; k++) {
			double first = 0;
			for (auto pair:*input)
				first += v[pair.first][k] * pair.second;

			double sec = 0;
			for (auto pair:*input)
				sec += v[pair.first][k] * pair.second * v[pair.first][k] * pair.second;
			res += first * first- sec;
		}
		 */

		//	printf("%lf\n",res);

		return sigmod(res);
	}
};


const int MPI_SEND_SIZE = 0;
const int MPI_SEND_STR_SIZE = 3;
const int MPI_SEND_FEATURE = 1;
const int MPI_SEND_LABEL = 2;


const int MPI_SEND_PUSH_W_BEGIN = 6;
const int MPI_SEND_PUSH_W = 14;
const int MPI_SEND_PUSH_V_BEGIN = 15;
const int MPI_SEND_PUSH_V = 16;


const int MPI_SEND_PULL_BEGIN = 5;
const int MPI_SEND_PULL = 7;

const int MPI_SEND_REQUEST = 14;


const int MPI_SEND_PULL_CBK_W_BEGIN = 9;
const int MPI_SEND_PULL_CBK_W = 11;
const int MPI_SEND_PULL_CBK_V_BEGIN = 12;
const int MPI_SEND_PULL_CBK_V = 13;

const int PULL_ID = 0;
const int PUSH_ID = 1;
const int CANCELL_ID = 2;

class FFMServer {
public:
	FFMModel model;

	FFMModel pull(const vector<int> &keys) {
		FFMModel result;
		for (int key:keys) {
			if (model.w.count(key)) {
				result.w[key] = model.w[key];
				if (key >= 1) {
					result.v[key - 1] = vector<double>(model.parameter.num_factors, 0);
					for (int i = 0; i < model.parameter.num_factors; i++)
						result.v[key - 1][i] = model.v[key - 1][i];
				}
			} else {
				result.w[key] = 0;
				if (key >= 1) {
					result.v[key - 1] = vector<double>(model.parameter.num_factors, 0);
					model.v[key - 1] = vector<double>(model.parameter.num_factors, 0);
				}
			}
		}

		result.parameter = model.parameter;
		return result;
	}

	void push(const svec_t<double> &w, svec_t<vector<double>> &v) {
		for (auto pair:w) {
			int key = pair.first;
			model.w[key] = pair.second;
			assert(!isnan(pair.second));
			if (key >= 1)
				for (int k = 0; k < model.parameter.num_factors; k++)
					model.v[key - 1][k] -= v[key - 1][k];
		}
	}


	void serve() {
		while (true) {
			int op;
			MPI_Status status;
			MPI_Recv(&op, 1, MPI_INT32_T, MPI_ANY_SOURCE, MPI_SEND_REQUEST, MPI_COMM_WORLD, &status);
			if (op == PULL_ID)
				serve_pull(status.MPI_SOURCE);
			else if (op == PUSH_ID)
				serve_push(status.MPI_SOURCE);
			else if (op == CANCELL_ID)
				break;
		}
	}

	void serve_push(int source) {

		MPI_Request request;
		MPI_Status status;

		char *w;
		char *v;
		int w_size;
		int v_size;

		MPI_Irecv(&w_size, 1, MPI_INT32_T, source, MPI_SEND_PUSH_W_BEGIN, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);

		w = new char[w_size];
		MPI_Irecv(w, w_size, MPI_CHAR, source, MPI_SEND_PUSH_W, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);
		assert(w[w_size-1] == 0);
		string w_str(w);
		svec_t<double> w_vec = deserialize(w);


		MPI_Irecv(&v_size, 1, MPI_INT32_T, source, MPI_SEND_PUSH_V_BEGIN, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);

		v = new char[v_size];
		MPI_Irecv(v, v_size, MPI_CHAR, source, MPI_SEND_PUSH_V, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);
		assert(v[v_size-1] == 0);
		string v_str(v);
		svec_t<vector<double>> v_vec = deserialize_v(v_str);

		push(w_vec, v_vec);

	}

	void serve_pull(int source) {
		MPI_Request request;
		MPI_Status status;
		int size;

		MPI_Irecv(&size, 1, MPI_INT32_T, source, MPI_SEND_PULL_BEGIN, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);

		vector<int> keys(size);
		MPI_Irecv(&keys[0], size, MPI_INT32_T, source, MPI_SEND_PULL, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);

		int w_size;
		int v_size;
		FFMModel pull_model = pull(keys);
		string w_str = pull_model.w.serialize();
		w_size = w_str.size() + 1;

		MPI_Isend(&w_size, 1, MPI_INT32_T, source, MPI_SEND_PULL_CBK_W_BEGIN, MPI_COMM_WORLD, &request);
		MPI_Isend(w_str.c_str(), w_size, MPI_CHAR, source, MPI_SEND_PULL_CBK_W, MPI_COMM_WORLD, &request);

		printf("Begin to serialize v.\n");
		string v_str = pull_model.v.serialize();
		v_size = v_str.size() + 1;

		MPI_Isend(&v_size, 1, MPI_INT32_T, source, MPI_SEND_PULL_CBK_V_BEGIN, MPI_COMM_WORLD, &request);


		MPI_Isend(v_str.c_str(), v_size, MPI_CHAR, source, MPI_SEND_PULL_CBK_V, MPI_COMM_WORLD, &request);

	}

};


struct MiniBatch {
	int startIndex;
	int endIndex;
};


class FFMClient {
public:
	FFMParameter parameter;
	DataSet *data;
	vector<MiniBatch> miniBatches;
	MPIConf conf;

	double calc_loss(svec_t<double> *input, double label, FFMModel &model) {
		return label - model.f(input);
	}


	FFMModel pull(const set<int> &keys) {

		int op = PULL_ID;
		MPI_Send(&op, 1, MPI_INT32_T, 0, MPI_SEND_REQUEST, MPI_COMM_WORLD);

		FFMModel result;
		int *key = new int[keys.size()];
		int *ptr = key;
		for (auto k:keys)
			*ptr++ = k;

		cout << "My rank:" << conf.rank << " Begin to pull" << endl;
		MPI_Request request;
		MPI_Status status;
		int size = keys.size();
		MPI_Isend(&size, 1, MPI_INT32_T, 0, MPI_SEND_PULL_BEGIN, MPI_COMM_WORLD, &request);
		MPI_Isend(key, size, MPI_INT32_T, 0, MPI_SEND_PULL, MPI_COMM_WORLD, &request);
		int w_size;
		int v_size;

		MPI_Irecv(&w_size, 1, MPI_INT32_T, 0, MPI_SEND_PULL_CBK_W_BEGIN, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);
		cout << "My rank:" << conf.rank << " Received pull size " << w_size << endl;
		char *w = new char[w_size + 1];
		MPI_Irecv(w, w_size, MPI_CHAR, 0, MPI_SEND_PULL_CBK_W, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);
		assert(w[w_size-1] == 0);
		cout << "My rank:" << conf.rank << " Received w " << endl;
		string w_str(w);
		result.w = deserialize(w_str);

		MPI_Irecv(&v_size, 1, MPI_INT32_T, 0, MPI_SEND_PULL_CBK_V_BEGIN, MPI_COMM_WORLD, &request);

		MPI_Wait(&request, &status);
		char *v = new char[v_size + 1];
		MPI_Irecv(v, v_size, MPI_CHAR, 0, MPI_SEND_PULL_CBK_V, MPI_COMM_WORLD, &request);
		MPI_Wait(&request, &status);
		assert(v[v_size-1]==0);
		cout << "My rank:" << conf.rank << " Received v " << endl;
		string v_str(v);
		result.v = deserialize_v(v_str);
		cout << "My rank:" << conf.rank << " Finish pull" << endl;
		return result;
	}

	void push(const svec_t<double> &w, svec_t<vector<double>> &v) {

		int op = PUSH_ID;
		MPI_Send(&op, 1, MPI_INT32_T, 0, MPI_SEND_REQUEST, MPI_COMM_WORLD);


		MPI_Request request;
		MPI_Status status;

		string w_str = w.serialize();
		string v_str = v.serialize();
		int w_size = w_str.size() + 1;
		int v_size = v_str.size() + 1;

		MPI_ISend(&w_size, 1, MPI_INT32_T, 0, MPI_SEND_PUSH_W_BEGIN, MPI_COMM_WORLD, &request);
		MPI_Isend(w_str.c_str(), w_size, MPI_CHAR, 0, MPI_SEND_PUSH_W, MPI_COMM_WORLD, &request);

		MPI_Isend(&v_size, 1, MPI_INT32_T, 0, MPI_SEND_PUSH_V_BEGIN, MPI_COMM_WORLD, &request);
		MPI_Isend(v_str.c_str(), v_size, MPI_CHAR, 0, MPI_SEND_PUSH_V, MPI_COMM_WORLD, &request);

		MPI_Wait(&request,&status);

	}

	double calc(int batchId) {
		MiniBatch batch = miniBatches[batchId];
		FFMModel model;
		set<int> push_need;
		//printf("start:%d end:%d\n",batch.startIndex,batch.endIndex);
		for (int i = batch.startIndex; i < batch.endIndex; i++) {
			svec_t<double> *input = &data->features[i];
			for (auto pair:*input)
				if (!push_need.count(pair.first + 1))
					push_need.insert(pair.first + 1);
		}
		push_need.insert(0);
		//printf("Begin to pull\n");
		model = pull(push_need);

		svec_t<double> mean_w;
		svec_t<vector<double>> mean_v;

		double total = 0;
		//printf("Begin to calc\n");
		for (int i = batch.startIndex; i < batch.endIndex; i++) {
			svec_t<double> *input = &data->features[i];
			double label = data->labels[i];
			double inner = model.w[0];
			for (auto pair:*input) {
				inner += model.w[pair.first] * pair.second;
			}
			clock_t start = clock();
			double loss = calc_loss(input, label, model);

			assert(!isnan(loss));
			total += log(1 + exp(-label * (label - loss)));
			//printf("Loss time:%lf\n",(clock()-start)/(double)CLOCKS_PER_SEC);
			//printf("i:%d Loss calculated size:%d\n",i,input->size());
			start = clock();
			for (auto pair: *input) {
				int p = pair.first;
				mean_w[p + 1] -= parameter.init_learning_rate * loss * pair.second;
				if (!mean_v.count(p))
					mean_v[p] = vector<double>(parameter.num_factors);

				for (int k = 0; k < parameter.num_factors; k++) {
					mean_v[p][k] -= parameter.init_learning_rate * loss *
					                (pair.second * inner - pair.second * pair.second * model.v[p][k]);
				}
			}
			//printf("Update time:%lf\n",(clock()-start)/(double)CLOCKS_PER_SEC);
			//printf("Update calculated\n");

		}
		for (auto pair:mean_w)
			mean_w[pair.first] /= (batch.endIndex - batch.startIndex);

		for (auto pair:mean_v)
			for (int k = 0; k < parameter.num_factors; k++)
				mean_v[pair.first][k] /= (batch.endIndex - batch.startIndex);
		//printf("Begin to push\n");
		push(mean_w, mean_v);

		return total / (batch.endIndex - batch.startIndex);
	}


	void SGD() {
		for (int i = 0; i < parameter.num_iter; i++) {

			int batchId = rand() % miniBatches.size();
			double loss = calc(batchId);
			printf("Iteration %d: loss %lf\n", i, loss);
		}
		int fini = CANCELL_ID;
		MPI_Send(&fini, 1, MPI_INT32_T, 0, MPI_SEND_REQUEST, MPI_COMM_WORLD);
	}

};


class FFMScheduler {
	FFMParameter parameter;
	DataSet *dataSet;
	MPIConf conf;
public:
	FFMScheduler(FFMParameter parameter, MPIConf conf, DataSet *dataSet) : parameter(parameter), conf(conf),
	                                                                       dataSet(dataSet) { }

	void partition() {

		for (int i = 0; i < conf.group_size; i++)
			if (i != conf.rank) {
				int total_size = dataSet->features.size();
				int size = (total_size + (conf.group_size - 2)) / (conf.group_size - 1);
				MPI_Send(&total_size, 1, MPI_INT32_T, i, MPI_SEND_SIZE, MPI_COMM_WORLD);

				stringstream ss;
				int min_size = min(size * i, total_size);
				for (int j = size * (i - 1); j < min_size; j++) {
					ss << (dataSet->features[j]).serialize();
					if (j != min_size - 1)
						ss << ',';
				}
				string result = ss.str();
				int str_size = result.size();
				MPI_Send(&str_size, 1, MPI_INT32_T, i, MPI_SEND_STR_SIZE, MPI_COMM_WORLD);
/*
				MPI_Datatype type;
				MPI_Type_contiguous(result.size(),MPI_CHAR,&type);
				MPI_Type_commit(&type);
				*/
				//cout<<result<<endl;
				MPI_Send(result.c_str(), result.size(), MPI_CHAR, i, MPI_SEND_FEATURE, MPI_COMM_WORLD);

				MPI_Send(&(dataSet->labels[(i - 1) * size]), min(size, total_size - (i - 1) * size), MPI_DOUBLE, i,
				         MPI_SEND_FEATURE, MPI_COMM_WORLD);


			}
	}

	void recv_partition() {
		int total_size;
		MPI_Recv(&total_size, 1, MPI_INT32_T, 0, MPI_SEND_SIZE, MPI_COMM_WORLD, NULL);

		int size = (total_size + (conf.group_size - 2)) / (conf.group_size - 1);
		dataSet = new DataSet;
		//dataSet->features = vector<svec_t<double>>(size);
/*
		MPI_Datatype type;
		MPI_Type_contiguous(min(size,total_size -size*(conf.rank-1)),MPI_DOUBLE,&type);
		MPI_Type_commit(&type);
		*/
		int str_size;
		MPI_Recv(&str_size, 1, MPI_INT32_T, 0, MPI_SEND_STR_SIZE, MPI_COMM_WORLD, NULL);

		char *str = new char[str_size + 1];
		MPI_Recv(str, str_size, MPI_CHAR, 0, MPI_SEND_FEATURE, MPI_COMM_WORLD, NULL);
		string data(str);
		vector<string> items = split(data, ',');
		//cout<<"Item size "<<items.size()<<endl;
		for (int i = 0; i < items.size(); i++) {
			dataSet->features.push_back(deserialize(items[i]));
		}
		dataSet->labels.resize(items.size());
		MPI_Recv(&(dataSet->labels[0]), items.size(), MPI_DOUBLE, 0, MPI_SEND_FEATURE, MPI_COMM_WORLD, NULL);
		cout << "My rank:" << conf.rank << " Finish receiving" << endl;

	}


	void server_serve() {
		FFMServer server;
		server.model.parameter = parameter;
		server.serve();
	}

	void run() {
		vector<MiniBatch> batches;

		for (int i = 0;
		     i < (dataSet->features.size() + parameter.mini_batch_size - 1) / parameter.mini_batch_size; i++) {
			MiniBatch batch;
			batch.startIndex = i * parameter.mini_batch_size;
			batch.endIndex = min((i + 1) * parameter.mini_batch_size, (int) dataSet->features.size());
			batches.push_back(batch);
		}

		FFMClient client;
		client.parameter = parameter;
		client.data = dataSet;
		client.miniBatches = batches;
		client.conf = conf;

		client.SGD();
	}

};


#endif //DIS_FFM_FFM_H
