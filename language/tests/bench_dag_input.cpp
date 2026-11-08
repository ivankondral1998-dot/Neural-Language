#include "neural_runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Result{double us; std::vector<double> outputs;size_t delta,full,blas;};
static Result measure(int width,int edits,int trial,neural::Model::Mode mode){
  neural::Model m("bench",width);
  m.layer_sources("left",256,1,123,0,{"input"});
  m.layer_sources("right",256,1,124,0,{"input"});
  m.layer_sources("merge",32,1,125,0,{"left","right"});
  m.layer_sources("output",8,0,126,0,{"merge","right"});
  m.mode=mode;
  std::vector<double> in(static_cast<size_t>(width),0.1);
  m.observe(in);m.infer(false);
  std::vector<double> trajectory;trajectory.reserve(80*8);
  const auto begin=std::chrono::steady_clock::now();
  for(int step=0;step<80;step++){
    for(int k=0;k<edits;k++){
      int idx=(step*31+k*7+trial*13)%width;
      in[static_cast<size_t>(idx)]+=(((step+k)%2)?-0.0002:0.0003);
    }
    m.observe(in);
    const auto& out=m.infer(false);
    trajectory.insert(trajectory.end(),out.begin(),out.end());
  }
  const auto end=std::chrono::steady_clock::now();
  return {std::chrono::duration<double,std::micro>(end-begin).count()/80.0,
          std::move(trajectory),m.counters.input_delta,m.counters.full,m.counters.blas_layers};
}
int main(){
  if(!neural::has_blas){std::cerr<<"BLAS NOT AVAILABLE\n";return 2;}
  std::cout<<"width,edits,trial,auto_us,full_us,delta_steps,auto_full,blas_layers,relerr\n";
  for(int width:{1024,2048})for(int edits:{1,8,64,256})for(int trial=0;trial<3;trial++){
    Result a,f;
    if(trial%2==0){a=measure(width,edits,trial,neural::Model::AUTO);f=measure(width,edits,trial,neural::Model::FULL);}
    else {f=measure(width,edits,trial,neural::Model::FULL);a=measure(width,edits,trial,neural::Model::AUTO);}
    if(!f.blas||a.outputs.size()!=f.outputs.size())throw std::runtime_error("invalid BLAS baseline");
    double worst=0;
    for(size_t i=0;i<a.outputs.size();i++){
      double dx=std::abs(a.outputs[i]-f.outputs[i])/(1+std::max(std::abs(a.outputs[i]),std::abs(f.outputs[i])));
      if(dx>1e-9||!std::isfinite(dx))throw std::runtime_error("invalid DAG numerical result");
      worst=std::max(worst,dx);
    }
    std::cout<<width<<','<<edits<<','<<trial<<','<<std::setprecision(12)<<a.us<<','<<f.us<<','
             <<a.delta<<','<<a.full<<','<<f.blas<<','<<worst<<'\n';
  }
}
