#include "neural_runtime.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

static neural::Model make(){
  neural::Model m("gradient",2);
  m.layer_sources("a",2,0,3,0,{"input"});
  m.layer_sources("b",2,0,4,0,{"input"});
  m.layer_sources("c",1,0,5,0,{"a","b","input"});
  for(size_t k=0;k<m.layers.size();k++){
    auto& l=m.layers[k];
    for(size_t i=0;i<l.w.size();i++)l.w[i]=0.05+0.017*static_cast<double>((i+7*k)%13);
    for(size_t i=0;i<l.b.size();i++)l.b[i]=0.005*(1+i+3*k);
  }
  m.observe({0.25,-0.17});
  return m;
}
static double loss(neural::Model model){
  const auto& v=model.infer(false);
  const double d=v[0]-0.36;
  return d*d;
}
int main(){
  neural::Model a=make();
  const neural::Model original=a;
  const double learning_rate=0.01;
  a.learn({0.36},learning_rate);
  const double eps=1e-6;
  double worst=0;
  size_t checked=0;
  for(size_t k=0;k<a.layers.size();k++){
    for(size_t i=0;i<a.layers[k].w.size();i++){
      neural::Model plus=original,minus=original;
      plus.layers[k].w[i]+=eps;
      minus.layers[k].w[i]-=eps;
      const double numeric=(loss(plus)-loss(minus))/(2*eps);
      const double analytic=(original.layers[k].w[i]-a.layers[k].w[i])/learning_rate;
      const double err=std::abs(numeric-analytic)/(1+std::abs(numeric));
      if(!std::isfinite(err)||err>1e-7){
        std::cerr<<"DAG_GRAD_FAIL k="<<k<<" i="<<i<<" numeric="<<numeric<<" analytic="<<analytic<<"\n";
        return 1;
      }
      worst=std::max(worst,err);checked++;
    }
    for(size_t i=0;i<a.layers[k].b.size();i++){
      neural::Model plus=original,minus=original;
      plus.layers[k].b[i]+=eps;
      minus.layers[k].b[i]-=eps;
      const double numeric=(loss(plus)-loss(minus))/(2*eps);
      const double analytic=(original.layers[k].b[i]-a.layers[k].b[i])/learning_rate;
      const double err=std::abs(numeric-analytic)/(1+std::abs(numeric));
      if(!std::isfinite(err)||err>1e-7)return 2;
      worst=std::max(worst,err);checked++;
    }
  }
  std::cout<<"DAG_GRADIENT_ORACLE_PASS checked="<<checked<<" worst_scaled_error="<<std::setprecision(16)<<worst<<"\n";
}
