// Neural Language CartPole-v1 pretrained-agent compatibility pilot.
// Physics matches the standard Gymnasium CartPole-v1 Euler defaults.
// This is an evaluation harness, not an implementation of an autonomous LLM.
#include "neural_runtime.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using State=std::array<double,4>;
struct CartPole {
  State s{};
  explicit CartPole(uint32_t seed){
    // Deterministic initial states for paired runs; not Gymnasium's PRNG.
    uint32_t a=seed;
    for(auto &v:s){a=1664525u*a+1013904223u;v=(double(a)/4294967296.0-0.5)*0.1;}
  }
  bool step(int action) {
    constexpr double force_mag=10., gravity=9.8, masscart=1., masspole=0.1;
    constexpr double total_mass=masscart+masspole, length=0.5;
    constexpr double polemass_length=masspole*length, tau=0.02;
    const double x=s[0],x_dot=s[1],theta=s[2],theta_dot=s[3];
    const double force=action==1?force_mag:-force_mag;
    const double costheta=std::cos(theta),sintheta=std::sin(theta);
    const double temp=(force+polemass_length*theta_dot*theta_dot*sintheta)/total_mass;
    const double thetaacc=(gravity*sintheta-costheta*temp)/(length*(4.0/3.0-masspole*costheta*costheta/total_mass));
    const double xacc=temp-polemass_length*thetaacc*costheta/total_mass;
    s={x+tau*x_dot,x_dot+tau*xacc,theta+tau*theta_dot,theta_dot+tau*thetaacc};
    return !(std::abs(s[0])>2.4||std::abs(s[2])>12.*std::acos(-1.)/180.);
  }
};

struct Stats{std::vector<int> episodes;uint64_t steps=0,disagreements=0;double max_error=0.,auto_us=0.,full_us=0.;size_t auto_delta=0,auto_full=0;};
Stats run(const std::string&model_path,int episode_count){
  neural::Model auto_model("CartPolePretrained",4),full_model("CartPolePretrained",4);
  for(auto* n:{&auto_model,&full_model}){
    n->layer("hidden1",64,3,42);n->layer("hidden2",64,3,43);n->layer("actions",2,0,44);
    n->load_model(model_path);
  }
  auto_model.mode=neural::Model::AUTO;full_model.mode=neural::Model::FULL;
  Stats stats;
  for(int episode=1;episode<=episode_count;episode++){
    CartPole env(episode);int steps=0;
    for(;steps<500;steps++){
      const std::vector<double> observation(env.s.begin(),env.s.end());
      const auto t0=std::chrono::steady_clock::now();
      auto_model.observe(observation);
      const auto a=auto_model.infer(false);
      const auto t1=std::chrono::steady_clock::now();
      full_model.observe(observation);
      const auto b=full_model.infer(false);
      const auto t2=std::chrono::steady_clock::now();
      stats.auto_us+=std::chrono::duration<double,std::micro>(t1-t0).count();
      stats.full_us+=std::chrono::duration<double,std::micro>(t2-t1).count();
      for(int k=0;k<2;k++){
        stats.max_error=std::max(stats.max_error,std::abs(a[k]-b[k]));
        if(!std::isfinite(a[k])||!std::isfinite(b[k])||std::abs(a[k]-b[k])>1e-8)
          throw std::runtime_error("AUTO/FULL prediction mismatch");
      }
      const int action_a=a[1]>a[0]?1:0,action_b=b[1]>b[0]?1:0;
      if(action_a!=action_b)stats.disagreements++;
      stats.steps++;
      if(!env.step(action_a)){++steps;break;}
    }
    stats.episodes.push_back(steps);
  }
  stats.auto_delta=auto_model.counters.input_delta;
  stats.auto_full=auto_model.counters.full;
  return stats;
}
int main(int argc,char**argv){
  try {
    if(argc<2||argc>3)throw std::runtime_error("usage: agent_cartpole MODEL.nlw [EPISODES]");
    const int n=argc==3?std::stoi(argv[2]):10;
    if(n<1||n>1000)throw std::runtime_error("episodes out of range");
    const Stats r=run(argv[1],n);
    const auto [minimum,maximum]=std::minmax_element(r.episodes.begin(),r.episodes.end());
    double total=0;for(int x:r.episodes)total+=x;
    std::cout<<std::fixed<<std::setprecision(9);
    std::cout<<"PILOT_SUCCESS episodes="<<r.episodes.size()<<" steps="<<r.steps
        <<" avg_episode="<<(total/n)<<" min_episode="<<*minimum<<" max_episode="<<*maximum
        <<" disagreements="<<r.disagreements<<" max_abs_error="<<r.max_error
        <<" auto_us_per_step="<<(r.auto_us/r.steps)<<" full_us_per_step="<<(r.full_us/r.steps)
        <<" auto_delta="<<r.auto_delta<<" auto_full="<<r.auto_full<<"\n";
    for(size_t i=0;i<r.episodes.size();i++)std::cout<<"EPISODE "<<(i+1)<<" steps="<<r.episodes[i]<<"\n";
    return r.disagreements?3:0;
  }catch(const std::exception&e){std::cerr<<"PILOT_ERROR "<<e.what()<<"\n";return 2;}
}
