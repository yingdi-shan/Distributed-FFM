//
// Created by syd on 16-6-21.
//

#ifndef DIS_FFM_DATA_H
#define DIS_FFM_DATA_H

#include <vector>
#include <unordered_map>
#include <sstream>

using namespace std;

vector<string> &split(const string &s, char delim, vector<string> &elems) {
	stringstream ss(s);
	string item;
	while (getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}


vector<string> split(const string &s, char delim) {
	vector<string> elems;
	split(s, delim, elems);
	return elems;
}

template <typename T>
class svec_t : public unordered_map<int, T> {
public:
	string serialize() const {
		stringstream ss;
		for (auto pair:*this) {
			ss << pair.first << ":" << pair.second << " ";
		}
		return ss.str();
	}
};

template <>
class svec_t<vector<double>> : public unordered_map<int, vector<double>> {
public:
	string serialize() const {
		printf("Entering special serialize.\n");
		stringstream ss;
		for (auto pair:*this) {
			ss << pair.first << ":";

			for(int i=0;i<pair.second.size();i++){
				ss << pair.second[i];
				if(i!=pair.second.size()-1)
					ss<<";";
			}
			ss<<" ";
		}
		ss.flush();
		printf("Finish serialize.\n");
		return ss.str();

	}
};


svec_t<double> deserialize(const string &s) {
	svec_t<double> result;
	stringstream ss(s);
	string comma;
	while(ss >> comma){
		vector<string> splited = split(comma, ':');
		int index = stoi(splited[0]);

		if(splited.size() != 2)
			cout<<comma<<endl;
		assert(splited.size()==2);
		//cout<<splited[1]<<endl;
		double value = stod(splited[1]);
		result[index] = value;
	}
	return result;
}

svec_t<vector<double>> deserialize_v(const string &s){
	svec_t<vector<double>> result;
	stringstream ss(s);
	string comma;
	while(ss >> comma){
		vector<string> splited = split(comma, ':');
		int index = stoi(splited[0]);
		vector<string> value_str = split(splited[1],';');
		vector<double> values;
		for(const string &x:value_str) {

			//if((x[0]<'0' || x[0]>'9') && x[0]!='-')
			//	cout <<"deserialize :"<< s<<endl;
			values.push_back(stod(x));
		}
		result[index] = values;
	}
	return result;
}


struct DataSet {
	vector<svec_t<double>> features;
	vector<double> labels;


	void read(const char *filename) {
		FILE *file = fopen(filename, "r");

		char line[4096];
		while (fgets(line, 4096, file)) {
			stringstream ss(line);
			double lab;
			ss >> lab;
			int index;
			double value;
			svec_t<double> feat;
			string comma;
			while (ss >> comma) {
				vector<string> splited = split(comma, ':');
				index = stoi(splited[0]);
				value = stod(splited[1]);
				feat[index] = value;
			}
			features.push_back(feat);
			labels.push_back(lab);
		}
		fclose(file);
	}
};


#endif //DIS_FFM_DATA_H
