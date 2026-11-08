#include "neural_runtime.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <limits>
#include <iomanip>

static neural::Model make(const char* name, neural::Model::Mode mode){
  neural::Model m(name,64);
  m.layer_sources("branch_a",32,1,23,0,{"input"});
  m.layer_sources("branch_b",24,2,24,1,{"input"});
  m.layer_sources("merged",16,1,25,0,{"branch_a","branch_b","input"});
  m.layer_sources("final",8,0,26,0,{"merged","branch_b"});
  m.mode=mode;
  return m;
}

static std::vector<double> reference(const neural::Model& model){
  std::vector<std::vector<long double>> outputs;
  for(const auto& layer:model.layers){
    std::vector<long double> combined;
    for(int src:layer.sources){
      if(src<0)for(double x:model.input)combined.push_back(static_cast<long double>(x));
      else combined.insert(combined.end(),outputs[static_cast<size_t>(src)].begin(),
                           outputs[static_cast<size_t>(src)].end());
    }
    if(combined.size()!=static_cast<size_t>(layer.in))throw std::runtime_error("oracle shape");
    std::vector<long double> result(static_cast<size_t>(layer.out));
    for(int row=0;row<layer.out;row++){
      long double pre=layer.b[static_cast<size_t>(row)];
      for(int col=0;col<layer.in;col++){
        const size_t ix=static_cast<size_t>(row)*layer.in+col;
        pre+=static_cast<long double>(layer.effective(ix))*combined[static_cast<size_t>(col)];
      }
      // Match runtime's activation semantics after FP64 accumulation.
      double value=static_cast<double>(pre);
      result[static_cast<size_t>(row)]=neural::act(value,layer.type);
    }
    outputs.push_back(std::move(result));
  }
  std::vector<double> result;
  for(long double v:outputs.back())result.push_back(static_cast<double>(v));
  return result;
}

int main(){
  const int runs=1200;
  neural::Model fast=make("DAG",neural::Model::DELTA);
  neural::Model full=make("DAG",neural::Model::FULL);
  std::vector<double> input(64,0.1);
  fast.observe(input);full.observe(input);
  (void)fast.infer(false);(void)full.infer(false);
  std::mt19937 rng(203106);
  double worst=0;
  size_t checked=0,learned=0;
  fast.checkpoint("initial");full.checkpoint("initial");
  for(int t=0;t<runs;t++){
    const int choice=static_cast<int>(rng()%21);
    if(choice<12){
      const int col=static_cast<int>(rng()%64);
      const double delta=(static_cast<int>(rng()%101)-50)*0.00001;
      fast.change_input(col,delta);full.change_input(col,delta);
      input[static_cast<size_t>(col)]+=delta;
    }else if(choice<15){
      for(int i=0;i<3;i++){
        const int col=static_cast<int>(rng()%64);
        input[static_cast<size_t>(col)]+=(static_cast<int>(rng()%21)-10)*0.00007;
      }
      fast.observe(input);full.observe(input);
    }else if(choice<17){
      const int row=static_cast<int>(rng()%16),col=static_cast<int>(rng()%120);
      fast.change("merged",row,col,0.00004);full.change("merged",row,col,0.00004);
    }else if(choice==17){
      const int row=static_cast<int>(rng()%8),col=static_cast<int>(rng()%40);
      const bool on=(rng()%2)==0;
      fast.gate("final",row,col,on);full.gate("final",row,col,on);
    }else if(choice==18){
      std::vector<double> target(8,0.2);
      fast.learn(target,0.002);full.learn(target,0.002);learned++;
      for(size_t i=0;i<fast.layers.size();i++){
        const auto& x=fast.layers[i];const auto& y=full.layers[i];
        if(x.w!=y.w||x.q!=y.q||x.b!=y.b||x.enabled!=y.enabled)
          throw std::runtime_error("DAG training diverged after input delta");
      }
    }else if(choice==19){
      fast.restore("initial");full.restore("initial");
      input.assign(64,0.1);
    }else{
      fast.observe(input);full.observe(input);
    }
    if(t%5==0)continue; // merge independent changes before evaluation
    const auto& a=fast.infer(false),&b=full.infer(false);
    const auto oracle=reference(fast);
    if(a.size()!=b.size()||a.size()!=oracle.size())throw std::runtime_error("DAG output shape");
    for(size_t j=0;j<a.size();j++){
      const double scale=1+std::max(std::abs(b[j]),std::abs(oracle[j]));
      const double err=std::max(std::abs(a[j]-b[j]),std::abs(a[j]-oracle[j]))/scale;
      if(!std::isfinite(err)||err>1e-9){
        std::cerr<<"DAG_ERROR t="<<t<<" j="<<j<<" a="<<a[j]<<" b="<<b[j]<<" oracle="<<oracle[j]<<"\n";
        return 3;
      }
      worst=std::max(worst,err);checked+=2;
    }
  }
  if(fast.counters.input_delta==0||learned==0||checked<10000){
    std::cerr<<"graph checks not exercised: delta="<<fast.counters.input_delta<<" checked="<<checked<<" learned="<<learned<<"\n";
    return 4;
  }
  // Direct runtime graph structure rejects duplicate/forward references.
  bool rejected=false;
  try{neural::Model bad("bad",2);bad.layer_sources("node",2,0,11,0,{"missing"});}
  catch(const std::runtime_error&){rejected=true;}
  if(!rejected)return 5;
  std::cout<<"DAG_CHANGE_PASS checks="<<checked<<" worst_scaled="<<std::setprecision(16)<<worst
           <<" delta_steps="<<fast.counters.input_delta<<" train_steps="<<learned<<"\n";
}
