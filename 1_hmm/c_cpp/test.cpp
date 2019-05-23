#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "hmm.h"
using namespace std;

static int load_models(const char* listname, vector<HMM*>& hmms, vector<string>& model_files) {
  FILE* fp = fopen(listname, "r");
  if (fp == nullptr) {
    perror(listname);
  }
  unsigned int count = 0;
  char filename[MAX_LINE] = "";
  while (fscanf(fp, "%s", filename) == 1) {
    hmms[count]->loadHMM(filename);
    model_files.push_back(filename);
    count++;

    if (count >= hmms.size()) {
      // return count;
      break;
    }
  }
  fclose(fp);
  return count;
}
void read_seq_data(vector<string>& seq_data, const char* seq_file_name) {
  ifstream seq_file(seq_file_name);
  if (!seq_file) {
    cerr << "Cannot open " << seq_file_name << ", aborting...\n";
    exit(-1);
  }
  string buf;
  while (getline(seq_file, buf)) {
    seq_data.push_back(buf);
  }
}

int main(int argc, const char** argv) {
  bool acc = false;
  if (argc < 4) {
    cerr << "Usage: ./test modellist.txt testing_data.txt result.txt [-a]\n";
    exit(-1);
  }
  if (argc == 5 && strcmp(argv[4], "-a") == 0) {
    cout << "Computing accuracy with testing_answer.txt.\n";
    acc = true;
  }

  /* initialize hmms and get modellist files*/
  vector<HMM*> hmms(5);
  for (int i = 0; i < 5; ++i) {
    hmms[i] = new HMM();
  }
  vector<string> model_files;
  load_models(argv[1], hmms, model_files);

  /* read testing data */
  vector<string> seq_data;
  read_seq_data(seq_data, argv[2]);

  /* compute best fit sequences */
  ofstream result_file(argv[3]);
  for (unsigned int i = 0; i < seq_data.size(); ++i) {
    double max_prob = 0.0;
    int best_model_index = -1;
    for (unsigned int j = 0; j < hmms.size(); ++j) {
      double res = hmms[j]->calculate_prob(seq_data[i]);
      if (res > max_prob) {
        max_prob = res;
        best_model_index = j;
      }
    }

    result_file << model_files[best_model_index] << " " << max_prob << endl;
  }

  /* computing accuracy */
  if (acc) {
    ofstream acc_file("acc.txt");
    ifstream res_file(argv[3]);
    ifstream answer_file("../testing_answer.txt");
    if (!answer_file || !res_file) return 0;

    string buf1, buf2;
    double numer = 0, deno = 0;
    while (getline(res_file, buf1)) {
      getline(answer_file, buf2);
      size_t p = buf1.find(' ');
      string f1 = buf1.substr(0, p);
      if (f1 == buf2) {  // sequence model match
        numer += 1;
      }
      deno += 1;
    }
    acc_file << numer / deno;
  }

  return 0;
}
