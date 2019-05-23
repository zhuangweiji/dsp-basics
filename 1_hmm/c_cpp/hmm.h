#ifndef HMM_HEADER_
#define HMM_HEADER_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <string>
#include <fstream>

using namespace std;

#ifndef MAX_STATE
#define MAX_STATE 10
#endif

#ifndef MAX_OBSERV
#define MAX_OBSERV 26
#endif

#ifndef MAX_SEQ
#define MAX_SEQ 200
#endif

#ifndef MAX_LINE
#define MAX_LINE 256
#endif

#ifndef o
#define o(t) seq[t]-'A'
#endif

#ifndef Pi
#define Pi(i) initial[i]
#endif

#ifndef A
#define A(i, j) transition[i][j]
#endif

#ifndef B
#define B(i, j) observation[j][i]
#endif

class HMM {
 private:
  char* model_name;
  int state_num;                              // number of state
  int observ_num;                             // number of observation
  double initial[MAX_STATE];                  // initial prob.
  double transition[MAX_STATE][MAX_STATE];    // transition prob.
  double observation[MAX_OBSERV][MAX_STATE];  // observation prob.

 public:
  HMM() {}
  ~HMM() {}
  int get_N() { return state_num; }
  void loadHMM(const char*);
  void dumpHMM(FILE*);
  void calculate_alpha(vector<vector<double> >&, const string&);
  void calculate_beta(vector<vector<double> >&, const string&);
  void calculate_gamma(vector<vector<double> >&, const vector<vector<double> >&, const vector<vector<double> >&);
  void calculate_epsilon(vector<vector<vector<double> > >&, const vector<vector<double> >&, const vector<vector<double> >&, const string&);
  void accumulate_params(const vector<vector<double> >&, const vector<vector<vector<double> > >&, vector<double>&, vector<vector<double> >&, vector<double>&, vector<vector<double> >&, const string&);
  void update_model(const vector<double>& pi_numer, const int sample_number, const vector<vector<double> >& epsilon_accumulation, const vector<double>& gamma_accumulation, const vector<vector<double> >& b_numer);
  double calculate_prob(const string& seq) const;
};

#endif
