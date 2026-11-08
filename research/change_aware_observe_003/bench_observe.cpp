#include "neural_runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Result {
  double micros=0;
  std::vector<double> output;
  size_t input_delta=0, full=0, blas=0;
};
Result run(const std::vector<int>& widths,int edits,int trial,neural::Model::Mode mode){
  neural::Model model("observe_bench",widths.front());
  for(size_t k=1;k<widths.size();k++)
    model.layer("L"+std::to_string(k),widths[k],k+1==widths.size()?0:1,
                static_cast<uint32_t>(1200+k),0);
  model.mode=mode;
  std::vector<double> values(static_cast<size_t>(widths.front()),0.12);
  model.observe(values);
  (void)model.infer(false);
  constexpr int steps=256;
  std::vector<double> trajectory;
  trajectory.reserve(steps*static_cast<size_t>(widths.back()));
  const auto start=std::chrono::steady_clock::now();
  for(int t=0;t<steps;t++){
    for(int k=0;k<edits;k++){
      const int col=(t*97+k*29+trial*7+3)%widths.front();
      values[static_cast<size_t>(col)]+=((t+k)%2==0?0.0005:-0.0005);
    }
    model.observe(values);  // old .nir instruction; no explicit observe_delta
    const auto& pred=model.infer(false);
    trajectory.insert(trajectory.end(),pred.begin(),pred.end());
  }
  const auto end=std::chrono::steady_clock::now();
  Result r;
  r.micros=std::chrono::duration<double,std::micro>(end-start).count()/steps;
  r.output=std::move(trajectory);
  r.input_delta=model.counters.input_delta;
  r.full=model.counters.full;
  r.blas=model.counters.blas_layers;
  return r;
}
int main(int argc,char**argv){
  if(!neural::has_blas){std::cerr<<"BLAS backend missing\n";return 2;}
  if(argc>2)return 2;
  const bool holdout=argc==2 && std::string(argv[1])=="--holdout";
  const bool holdout2=argc==2 && std::string(argv[1])=="--holdout2";
  const bool holdout3=argc==2 && std::string(argv[1])=="--holdout3";
  const bool holdout4=argc==2 && std::string(argv[1])=="--holdout4";
  if(argc==2&&!holdout&&!holdout2&&!holdout3&&!holdout4)return 2;
  const std::vector<std::vector<int>> networks=holdout4 ?
    std::vector<std::vector<int>>{{640,96},{1408,176},{896,144,36},{2016,192},{384,96,24}} :
    (holdout3 ? std::vector<std::vector<int>>{{1280,160},{896,112,28},{1600,160,48},{448,112,28},{1920,256,64}} :
    (holdout2 ? std::vector<std::vector<int>>{{1536,320},{768,160,40},{1792,224,32},{320,128,32}} :
    (holdout ? std::vector<std::vector<int>>{{2048,256},{1024,128,64}} :
    std::vector<std::vector<int>>{{512,128},{1024,256}})));
  const std::vector<int> edits_by_case=holdout4 ? std::vector<int>{1,6,24,72,120} :
    (holdout3 ? std::vector<int>{1,8,40,80,144} :
    (holdout2 ? std::vector<int>{1,8,48,96,160}:std::vector<int>{1,8,64,128}));
  std::cout<<"network,edits,trial,auto_us,full_us,auto_delta,auto_full,full_blas,scaled_error\n";
  for(size_t net=0;net<networks.size();net++)for(int edits:edits_by_case)for(int trial=0;trial<5;trial++){
    Result a,f;
    if(trial%2==0){a=run(networks[net],edits,trial,neural::Model::AUTO);
                   f=run(networks[net],edits,trial,neural::Model::FULL);}
    else {f=run(networks[net],edits,trial,neural::Model::FULL);
          a=run(networks[net],edits,trial,neural::Model::AUTO);}
    if(!f.blas||a.output.size()!=f.output.size())return 3;
    double worst=0;
    for(size_t i=0;i<a.output.size();i++){
      const double x=a.output[i],y=f.output[i];
      if(!std::isfinite(x)||!std::isfinite(y))return 4;
      worst=std::max(worst,std::abs(x-y)/(1+std::max(std::abs(x),std::abs(y))));
    }
    if(worst>1e-9){std::cerr<<"numerical mismatch "<<worst<<"\n";return 5;}
    std::cout<<net<<','<<edits<<','<<trial<<','<<std::setprecision(12)
             <<a.micros<<','<<f.micros<<','<<a.input_delta<<','<<a.full<<','<<f.blas
             <<','<<worst<<'\n';
  }
}
