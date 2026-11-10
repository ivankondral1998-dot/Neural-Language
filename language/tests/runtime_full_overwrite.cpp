#include "neural_runtime.hpp"
#include <limits>
#include <iostream>
int main() {
  try {
    int checks=0;
    for (bool graph : {false,true}) for (int storage : {0,1}) for (int activation : {0,1,2,3}) for(bool masked:{false,true}) {
      neural::Model m("Overwrite",3);
      if(graph) {m.layer_sources("h",5,activation,11,storage,{"input"});m.layer_sources("o",2,0,13,storage,{"input","h"});}
      else {m.layer("h",5,activation,11,storage);m.layer("o",2,0,13,storage);}
      m.mode=neural::Model::FULL;
      if(masked)m.gate("h",2,1,false);
      m.observe({0.1,-0.2,0.3});m.forward();
      const auto expectedZ=m.z,expectedA=m.a;
      for(int repeat=0;repeat<4;repeat++) {
        for(auto* arrays : {&m.z,&m.a}) for(auto& values:*arrays) std::fill(values.begin(),values.end(),std::numeric_limits<double>::quiet_NaN());
        m.forward();
        if(m.z!=expectedZ||m.a!=expectedA)throw std::runtime_error("FULL left stale activations");
        checks++;
      }
    }
    neural::Model bad("Recovery",2);bad.layer("out",2,0,1);bad.layers[0].w={1.,1.,2.,0.};bad.observe({0.,1.});bad.forward();
    bad.change_input(0,600000000.);bool failed=false;
    try{bad.forward();}catch(const std::runtime_error&){failed=true;}
    if(!failed||bad.cache)throw std::runtime_error("invalid cache after partial failure");
    bad.change_input(0,-600000000.);
    if(bad.infer(false)!=std::vector<double>{1.,0.})throw std::runtime_error("recovery mismatch");
    std::cout<<"FULL_OVERWRITE_PASS cases="<<checks<<" recovery=1\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
