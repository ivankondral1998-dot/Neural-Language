#include "neural_runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Result {double micros;std::vector<double> trace;size_t delta;size_t full;size_t blas;};
Result run(const std::vector<int>& widths,int edits,int trial,neural::Model::Mode mode){
  neural::Model m("benchmark",widths.front());
  for(size_t k=1;k<widths.size();k++){
    m.layer("L"+std::to_string(k),widths[k],k+1==widths.size()?0:1,static_cast<uint32_t>(42+k),0);
  }
  m.mode=mode;
  m.observe(std::vector<double>(static_cast<size_t>(widths.front()),0.12));
  m.infer(false); // identical warm-up and cache state
  constexpr int steps=256;
  std::vector<double> trace;
  trace.reserve(static_cast<size_t>(steps)*widths.back());
  // Full and AUTO use identical, trial-specific changes, with no wall-clock input.
  auto start=std::chrono::steady_clock::now();
  for(int t=0;t<steps;t++){
    for(int k=0;k<edits;k++){
      const int ix=(t*97 + k*29 + trial*7 +3)%widths.front();
      const double amount=((t+k)%2==0 ? 0.0005 : -0.0005);
      m.change_input(ix,amount);
    }
    const auto& output=m.infer(false);
    trace.insert(trace.end(),output.begin(),output.end());
  }
  const auto end=std::chrono::steady_clock::now();
  return {std::chrono::duration<double,std::micro>(end-start).count()/steps,
          std::move(trace),m.counters.input_delta,m.counters.full,m.counters.blas_layers};
}
int main(int argc,char** argv){
  if(!neural::has_blas){std::cerr<<"BLAS REQUIRED\n";return 2;}
  std::cout<<"network,edits,trial,auto_us,full_us,ratio,auto_delta,auto_full,full_blas_layers,max_scaled_error\n";
  const bool holdout=argc==2 && std::string(argv[1])=="--holdout";
  if(argc>2 || (argc==2 && !holdout))return 2;
  std::vector<std::vector<int>> networks=holdout ?
       std::vector<std::vector<int>>{{1536,192},{768,192,48},{256,96,24},{1024,128,64}} :
       std::vector<std::vector<int>>{{2048,256},{512,256,64},{128,128,32}};
  const std::vector<int> edits_by_case=holdout ? std::vector<int>{2,8,24,48} : std::vector<int>{1,4,16,64};
  const int trials=holdout?7:5;
  for(size_t ni=0;ni<networks.size();ni++)for(int edits:edits_by_case)for(int trial=0;trial<trials;trial++){
    // Alternate execution order to mitigate sequential CPU frequency drift.
    Result full,automatic;
    if(trial%2){full=run(networks[ni],edits,trial,neural::Model::FULL);
                automatic=run(networks[ni],edits,trial,neural::Model::AUTO);}
    else {automatic=run(networks[ni],edits,trial,neural::Model::AUTO);
          full=run(networks[ni],edits,trial,neural::Model::FULL);}
    if(!full.blas || full.trace.size()!=automatic.trace.size())return 3;
    double error=0;
    for(size_t i=0;i<full.trace.size();i++){
      const double x=automatic.trace[i],y=full.trace[i];
      if(!std::isfinite(x) || !std::isfinite(y))return 4;
      error=std::max(error,std::abs(x-y)/(1+std::max(std::abs(x),std::abs(y))));
    }
    if(error>1e-9){std::cerr<<"accuracy failure "<<error<<"\n";return 5;}
    std::cout<<ni<<','<<edits<<','<<trial<<','<<std::setprecision(12)
             <<automatic.micros<<','<<full.micros<<','<<automatic.micros/full.micros<<','
             <<automatic.delta<<','<<automatic.full<<','<<full.blas<<','<<error<<'\n';
  }
  return 0;
}
