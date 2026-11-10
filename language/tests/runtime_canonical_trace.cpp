// Differential contract: strict partial versus canonical FULL, including
// DAG fan-out, INT8 storage, mutable weights, masks, training and restoring.
#include "neural_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static neural::Model make(bool graph, bool int8) {
  neural::Model m("CanonicalTrace",32);
  const int store=int8?1:0;
  if(graph){
    m.layer_sources("left",11,1,101,store,{"input"});
    m.layer_sources("right",8,3,102,store,{"input"});
    m.layer_sources("merge",6,2,103,store,{"left","right"});
    m.layer_sources("output",3,0,104,store,{"merge","right"});
  }else{
    m.layer("left",11,1,101,store);
    m.layer("right",8,3,102,store);
    m.layer("merge",6,2,103,store);
    m.layer("output",3,0,104,store);
  }
  return m;
}
static void equal(const neural::Model& ref,const neural::Model& test,int i){
  if(ref.layers.size()!=test.layers.size())throw std::runtime_error("topology mismatch");
  for(size_t k=0;k<ref.layers.size();++k){
    const auto& lr=ref.layers[k];const auto& lt=test.layers[k];
    if(lr.w!=lt.w || lr.q!=lt.q || lr.b!=lt.b || lr.enabled!=lt.enabled)
      throw std::runtime_error("different weights on step "+std::to_string(i));
    for(size_t j=0;j<ref.a[k].size();++j){
      if(!(ref.a[k][j]==test.a[k][j]) || !(ref.z[k][j]==test.z[k][j]))
        throw std::runtime_error("different FP64 state on step "+std::to_string(i)+" layer "+std::to_string(k)+" node "+std::to_string(j));
    }
  }
}
int main() {
  try {
    int checks=0;
    for(bool graph:{false,true})for(bool int8:{false,true}){
      neural::Model reference=make(graph,int8),partial=reference;
      reference.mode=neural::Model::FULL;
      partial.mode=neural::Model::DELTA;
      if(!partial.strict_numeric)throw std::runtime_error("strict mode is not default");
      std::vector<double> input(32);
      for(int j=0;j<32;j++)input[j]=(j%5-2)*0.013;
      reference.observe(input);partial.observe(input);
      reference.infer(false);partial.infer(false);
      reference.checkpoint("start");
      partial.checkpoint("start");
      for(int step=0;step<720;step++){
        const int col=(step*17+7)%32;
        if(step%53==0){
          reference.restore("start");partial.restore("start");
          // Reference and strict both restore input and network state.
          input=reference.input;
        }else if(step%37==0){
          reference.learn({0.1,-0.2,0.3},0.0125);
          partial.learn({0.1,-0.2,0.3},0.0125);
        }else if(step%29==0){
          const bool state=((step/29)%2)==0;
          reference.gate("right",0,1,state);
          partial.gate("right",0,1,state);
        }else if(step%23==0){
          reference.change("left",1,2,0.00015);
          partial.change("left",1,2,0.00015);
        }else if(step%7==0){
          input[col]+=0.001*(step%3-1);
          input[(col+9)%32]-=0.0007;
          reference.observe(input);partial.observe(input);
        }else{
          const double amount=(step%2?1:-1)*0.000071;
          input[col]+=amount;
          reference.change_input(col,amount);
          partial.change_input(col,amount);
        }
        const auto& a=reference.infer(false);
        const auto& b=partial.infer(false);
        if(a.size()!=b.size())throw std::runtime_error("output size mismatch");
        for(size_t j=0;j<a.size();j++){
          if(!std::isfinite(a[j]) || a[j]!=b[j])
            throw std::runtime_error("different output on step "+std::to_string(step));
        }
        equal(reference,partial,step);
        ++checks;
      }
      if(partial.counters.input_delta==0)
        throw std::runtime_error("strict delta not exercised in trace");
    }
    std::cout<<"CANONICAL_TRACE_PASS cases=4 steps=720 checks="<<checks
             <<" finite_exact_full_equivalence=true\n";
  }catch(const std::exception& error){
    std::cerr<<"CANONICAL_TRACE_FAIL "<<error.what()<<"\n";
    return 2;
  }
}
