// Independent finite-difference gradient check of DAG backpropagation.
// Reference forward pass is implemented here without calling Model::forward.
#include "neural_runtime.hpp"
#include <cmath>
#include <iostream>
#include <vector>

static double reference_loss(const neural::Model& m, const std::vector<double>& input, double target) {
  std::vector<std::vector<double>> outputs;
  for(const auto& l:m.layers){
    std::vector<double> sources;
    for(int src:l.sources){
      const auto& values=src==-1?input:outputs[static_cast<size_t>(src)];
      sources.insert(sources.end(),values.begin(),values.end());
    }
    if(sources.size()!=static_cast<size_t>(l.in))throw std::runtime_error("oracle bad shape");
    std::vector<double> y(l.out,0.0);
    for(int r=0;r<l.out;r++){
      long double z=l.b[r];
      for(int c=0;c<l.in;c++){
        const size_t ix=static_cast<size_t>(r)*l.in+c;
        if(l.enabled[ix])z+=static_cast<long double>(l.raw(ix))*sources[c];
      }
      if(l.type==1)y[r]=std::max(0.0,static_cast<double>(z));
      else if(l.type==2)y[r]=1.0/(1.0+std::exp(-static_cast<double>(z)));
      else y[r]=static_cast<double>(z);
    }
    outputs.push_back(std::move(y));
  }
  double sum=0;
  for(double x:outputs.back())sum+=(x-target)*(x-target);
  return sum/outputs.back().size();
}

int main(){
  neural::Model base("Graph",2);
  base.layer("shared",2,0,7);
  base.layer_sources("left",2,0,9,0,{"shared"});
  base.layer_sources("right",1,0,11,0,{"shared"});
  base.layer_sources("result",1,2,13,0,{"left","right"});
  base.gate("right",0,0,false);
  const std::vector<double> x{0.4,-0.7};
  const double target=0.86,rate=0.01,epsilon=1e-6;
  auto trained=base;
  trained.observe(x);
  trained.learn({target},rate);
  int checked=0;
  double worst=0;
  for(size_t k=0;k<base.layers.size();k++){
    const auto& layer=base.layers[k];
    for(size_t j=0;j<layer.weight_count();j++){
      auto plus=base,minus=base;
      plus.layers[k].assign_weight(j,base.layers[k].raw(j)+epsilon);
      minus.layers[k].assign_weight(j,base.layers[k].raw(j)-epsilon);
      const double numeric=(reference_loss(plus,x,target)-reference_loss(minus,x,target))/(2*epsilon);
      const double update=(layer.raw(j)-trained.layers[k].raw(j))/rate;
      const double err=std::abs(numeric-update);
      worst=std::max(worst,err);
      if(err>1e-6){std::cerr<<"GRADIENT_FAIL weight layer="<<k<<" weight="<<j<<" oracle="<<numeric<<" actual="<<update<<"\n";return 1;}
      checked++;
    }
    for(size_t j=0;j<layer.b.size();j++){
      auto plus=base,minus=base;
      plus.layers[k].b[j]+=epsilon;
      minus.layers[k].b[j]-=epsilon;
      const double numeric=(reference_loss(plus,x,target)-reference_loss(minus,x,target))/(2*epsilon);
      const double update=(layer.b[j]-trained.layers[k].b[j])/rate;
      const double err=std::abs(numeric-update);
      worst=std::max(worst,err);
      if(err>1e-6){std::cerr<<"GRADIENT_FAIL bias layer="<<k<<" bias="<<j<<" oracle="<<numeric<<" actual="<<update<<"\n";return 2;}
      checked++;
    }
  }
  if(!base.graph) return 3;
  std::cout<<"GRADIENT_ORACLE_PASS "<<checked<<" parameters; max_abs_error="<<worst<<"\n";
}
