#include "neural_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

// Independent long-double dot-product oracle (no Model::forward/infer/cache).
std::vector<double> oracle(const neural::Model& model){
  std::vector<double> prev=model.input;
  for (const auto& l: model.layers){
    std::vector<double> cur(static_cast<size_t>(l.out));
    for(int r=0;r<l.out;r++){
      long double s=l.b[static_cast<size_t>(r)];
      for(int c=0;c<l.in;c++){
        const size_t ix=static_cast<size_t>(r)*l.in+c;
        if(l.enabled[ix])s+=static_cast<long double>(l.raw(ix))*prev[static_cast<size_t>(c)];
      }
      double z=static_cast<double>(s);
      if(l.type==1)z=std::max(0.,z);
      if(l.type==2)z=z>=0 ? 1./(1.+std::exp(-z)):std::exp(z)/(1.+std::exp(z));
      cur[static_cast<size_t>(r)]=z;
    }
    prev=std::move(cur);
  }
  return prev;
}

int main(){
  std::mt19937 rng(20261108);
  std::size_t comparisons=0, coalesced=0, empty_skips=0;
  double worst=0;
  for(int network=0;network<5;network++){
    const int ni=network==4?2048:(network%2?63:16);
    neural::Model fast("fast",ni),full("full",ni);
    const int a=network==4?256:24;
    fast.layer("H",a,network%3,1200+network,network%2);
    full.layer("H",a,network%3,1200+network,network%2);
    if(network%2==0||network==4){
      fast.layer("O",network==4?16:7,network%2==0?2:0,333+network,0);
      full.layer("O",network==4?16:7,network%2==0?2:0,333+network,0);
    }
    fast.mode=neural::Model::AUTO;
    full.mode=neural::Model::FULL;
    std::vector<double> v(static_cast<size_t>(ni),0.2);
    fast.observe(v);full.observe(v);
    (void)fast.infer(false);(void)full.infer(false);
    const size_t full_count=fast.counters.full;
    for(int i=0;i<20;i++) fast.observe(v); // unchanged full observations are true no-ops
    (void)fast.infer(false);
    if(fast.counters.full!=full_count||!fast.pending_input.empty())return 10;
    // The caller can explicitly request delta mode even on a network where
    // AUTO's cost model prefers full recomputation. Coalescing is invariant.
    fast.mode=neural::Model::DELTA;
    // Alternate identical and oscillating full observations; only one distinct
    // input may change across thousands of frames without prediction.
    const auto initial=v;
    for(int i=0;i<10000;i++){
      v[3]=0.2+(i%5)*0.0001;
      fast.observe(v);full.observe(v);
    }
    if(fast.pending_input.size()!=1){
      std::cerr<<"COALESCE_FAIL size="<<fast.pending_input.size()<<"\n";return 11;
    }
    coalesced+=10000-fast.pending_input.size();
    v=initial; fast.observe(v); full.observe(v);
    const size_t before=fast.counters.full+fast.counters.delta;
    const auto& same=fast.infer(false);
    if(fast.counters.full+fast.counters.delta!=before || !fast.pending_input.empty())return 12;
    if(same!=full.infer(false))return 13;
    empty_skips++;
    fast.mode=neural::Model::AUTO;
    fast.checkpoint("initial");full.checkpoint("initial");
    for(int step=0;step<900;step++){
      const int choice=static_cast<int>(rng()%14);
      if(choice<8){
        // Full observe values, but only 1-3 input coordinates change.
        int touches=1+static_cast<int>(rng()%3);
        for(int j=0;j<touches;j++){
          const int col=static_cast<int>(rng()%static_cast<unsigned>(ni));
          const double diff=(int(rng()%21)-10)*0.00007;
          v[static_cast<size_t>(col)]+=diff;
        }
        fast.observe(v);full.observe(v);
      } else if(choice==8){
        int col=static_cast<int>(rng()%static_cast<unsigned>(ni));
        const double add=0.0004;
        fast.change_input(col,add);full.change_input(col,add);
        v[static_cast<size_t>(col)]+=add;
      } else if(choice==9){
        const auto& l=fast.layers.front();
        int row=static_cast<int>(rng()%static_cast<unsigned>(l.out));
        int col=static_cast<int>(rng()%static_cast<unsigned>(l.in));
        fast.change("H",row,col,0.0006);full.change("H",row,col,0.0006);
      } else if(choice==10){
        int row=static_cast<int>(rng()%static_cast<unsigned>(a));
        int col=static_cast<int>(rng()%static_cast<unsigned>(ni));
        bool enabled=(rng()%2)==0;
        fast.gate("H",row,col,enabled);full.gate("H",row,col,enabled);
      } else if(choice==11){
        fast.restore("initial");full.restore("initial");
        v.assign(static_cast<size_t>(ni),0.2);
      } else if(choice==12){
        // Identical observe must not change cached outputs.
        fast.observe(v);full.observe(v);
      } else {
        std::vector<double> target(static_cast<size_t>(fast.layers.back().out),0.1);
        fast.learn(target,0.0008);full.learn(target,0.0008);
      }
      if(step%4==0)continue; // allow multiple changes between infer() calls
      if(fast.pending_input.size()>static_cast<size_t>(ni))return 14;
      const auto& actual=fast.infer(false);
      const auto& reference=full.infer(false);
      const auto independent=oracle(fast);
      if(actual.size()!=reference.size()||actual.size()!=independent.size())return 15;
      for(size_t k=0;k<actual.size();k++){
        const double scale=1+std::max(std::abs(reference[k]),std::abs(independent[k]));
        const double err=std::max(std::abs(actual[k]-reference[k]),std::abs(actual[k]-independent[k]))/scale;
        worst=std::max(worst,err);
        if(err>1e-9||!std::isfinite(err)){
          std::cerr<<"REUSE_MISMATCH network="<<network<<" step="<<step<<" element="<<k
                   <<" got="<<actual[k]<<" expected="<<reference[k]<<" oracle="<<independent[k]<<"\n";
          return 16;
        }
        comparisons+=2;
      }
    }
    if(network==4 && fast.counters.input_delta==0) {
      std::cerr<<"AUTO NEVER CHOSE DELTA ON LARGE NETWORK\n";return 17;
    }
  }
  // Must not corrupt prior valid state on failed sensor values.
  neural::Model invalid("negative",2);invalid.layer("out",2,0,11);
  invalid.observe({0.2,0.3});invalid.infer(false);
  bool rejected=false;
  try{invalid.observe({0.5,std::numeric_limits<double>::quiet_NaN()});}
  catch(const std::exception&){rejected=true;}
  if(!rejected||invalid.input!=std::vector<double>({0.2,0.3}))return 18;
  // Floating-point delta inference must never silently change the model's
  // subsequent SGD trajectory. Training forces a canonical full forward.
  neural::Model stream("train",1024),canonical("train",1024);
  for(auto* m:{&stream,&canonical}){
    m->layer("H",128,1,333);
    m->layer("O",16,2,334);
    m->observe(std::vector<double>(1024,0.2));m->infer(false);
  }
  stream.mode=neural::Model::AUTO;
  canonical.mode=neural::Model::FULL;
  std::vector<double> changed(1024,0.2);changed[5]=0.201;
  stream.observe(changed);canonical.observe(changed);
  (void)stream.infer(false);(void)canonical.infer(false);
  std::vector<double> target(16,0.4);
  stream.learn(target,0.003);canonical.learn(target,0.003);
  for(size_t k=0;k<stream.layers.size();k++){
    const auto& a=stream.layers[k];const auto& b=canonical.layers[k];
    if(a.b!=b.b||a.w!=b.w||a.q!=b.q)return 19;
  }
  std::cout<<"OBSERVE_REUSE_PASS comparisons="<<comparisons<<" worst_scaled_error="
           <<std::setprecision(15)<<worst<<" coalesced_events="<<coalesced
           <<" canceled_batches="<<empty_skips<<"\n";
}
