#include "neural_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// Independent floating-point forward oracle. Reads model weights; does not use
// Model::forward, cached activations or incremental propagation internals.
std::vector<double> oracle(const neural::Model& m) {
  std::vector<double> previous=m.input;
  for (const auto& l:m.layers) {
    std::vector<double> next(static_cast<size_t>(l.out));
    for(int r=0;r<l.out;r++){
      long double sum=l.b[static_cast<size_t>(r)];
      for(int c=0;c<l.in;c++){
        size_t at=static_cast<size_t>(r)*l.in+c;
        if(l.enabled[at]) sum += static_cast<long double>(l.raw(at))*previous[static_cast<size_t>(c)];
      }
      double z=static_cast<double>(sum);
      if(l.type==1)z=std::max(0.0,z);
      if(l.type==2)z=z>=0 ? 1.0/(1.0+std::exp(-z)) : std::exp(z)/(1.0+std::exp(z));
      next[static_cast<size_t>(r)]=z;
    }
    previous.swap(next);
  }
  return previous;
}

int main(){
  std::mt19937 rng(20261108);
  double worst=0.0;
  size_t comparisons=0;
  for(int q=0;q<6;q++) {
    int ni=q%2?128:16;
    neural::Model fast("input",ni),full("input",ni);
    for(int layer=0;layer<(q%3)+1;layer++){
      const int width=layer==0 ? (q%2?32:12): (layer==1?17:3);
      const int act=(layer+q)%3, storage=(layer+q)%2;
      fast.layer("L"+std::to_string(layer),width,act,static_cast<uint32_t>(37+layer+q),storage);
      full.layer("L"+std::to_string(layer),width,act,static_cast<uint32_t>(37+layer+q),storage);
    }
    fast.mode=neural::Model::DELTA; full.mode=neural::Model::FULL;
    std::vector<double> start(static_cast<size_t>(ni),0.2);
    fast.observe(start);full.observe(start);
    (void)fast.infer(false);(void)full.infer(false);
    fast.checkpoint("initial");full.checkpoint("initial");
    for(int step=0;step<480;step++){
      const int selector=static_cast<int>(rng()%13);
      if(selector<8){
        const int col=static_cast<int>(rng()%static_cast<unsigned>(ni));
        const double change=(int(rng()%19)-9)*0.0006;
        fast.change_input(col,change);full.change_input(col,change);
      } else if(selector==8){
        const int layer=static_cast<int>(rng()%fast.layers.size());
        const auto& l=fast.layers[static_cast<size_t>(layer)];
        const int r=static_cast<int>(rng()%static_cast<unsigned>(l.out));
        const int c=static_cast<int>(rng()%static_cast<unsigned>(l.in));
        const double delta=0.004;
        fast.change(l.name,r,c,delta);full.change(l.name,r,c,delta);
      } else if(selector==9){
        const int layer=static_cast<int>(rng()%fast.layers.size());
        const auto& l=fast.layers[static_cast<size_t>(layer)];
        const int r=static_cast<int>(rng()%static_cast<unsigned>(l.out));
        const int c=static_cast<int>(rng()%static_cast<unsigned>(l.in));
        const bool on=(rng()%2)==0;
        fast.gate(l.name,r,c,on);full.gate(l.name,r,c,on);
      } else if(selector==10){
        std::vector<double> v(static_cast<size_t>(ni),0.2);
        v[static_cast<size_t>(rng()%static_cast<unsigned>(ni))]+=0.03;
        fast.observe(v);full.observe(v);
      } else if(selector==11){
        fast.restore("initial");full.restore("initial");
      } else {
        std::vector<double> target(static_cast<size_t>(fast.layers.back().out),0.12);
        fast.learn(target,0.002);full.learn(target,0.002);
      }
      if(step%5==0){ // exercise multiple input updates between predictions
        continue;
      }
      const auto& actual=fast.infer(false);
      const auto& expected=full.infer(false);
      const auto independent=oracle(fast);
      if(actual.size()!=expected.size()||actual.size()!=independent.size())return 3;
      for(size_t r=0;r<actual.size();r++){
        const double scale=1+std::max(std::abs(actual[r]),std::abs(independent[r]));
        const double mismatch=std::abs(actual[r]-independent[r])/scale;
        const double mismatch_full=std::abs(actual[r]-expected[r])/scale;
        worst=std::max(worst,std::max(mismatch,mismatch_full));
        if(mismatch>1e-9 || mismatch_full>1e-9){
          std::cerr<<"MISMATCH case="<<q<<" step="<<step<<" index="<<r
                   <<" actual="<<actual[r]<<" reference="<<independent[r]<<"\n";
          return 4;
        }
        comparisons+=2;
      }
    }
    if(fast.counters.input_events==0 || fast.counters.input_delta==0)return 5;
  }
  neural::Model fail("fail",2);
  fail.layer("o",1,0,19);
  bool precondition=false,invalid_index=false,invalid_num=false;
  try{fail.change_input(0,0.1);}catch(const std::exception&){precondition=true;}
  fail.observe({0.0,0.0});
  try{fail.change_input(2,0.1);}catch(const std::exception&){invalid_index=true;}
  try{fail.change_input(0,std::numeric_limits<double>::infinity());}catch(const std::exception&){invalid_num=true;}
  if(!(precondition&&invalid_index&&invalid_num))return 6;
  std::cout<<"CHANGE_INPUT_STRESS_PASS comparisons="<<comparisons
           <<" worst_scaled_error="<<std::setprecision(15)<<worst<<"\n";
  return 0;
}
