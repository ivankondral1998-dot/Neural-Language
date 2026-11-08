#include "neural_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using neural::Model;
static void equal_state(Model &m, int seed, int step) {
    const auto got = m.infer(false);
    Model reference=m;
    reference.mode=Model::FULL;
    reference.cache=false;
    const auto wanted=reference.infer(false);
    for(size_t k=0;k<m.layers.size();++k){
        for(int j=0;j<m.layers[k].out;++j){
            for(int kind=0;kind<2;++kind){
                double x=kind ? m.a[k][j] : m.z[k][j];
                double y=kind ? reference.a[k][j] : reference.z[k][j];
                double tol=1e-9 * (1+std::max(std::abs(x),std::abs(y)));
                if(!std::isfinite(x)||!std::isfinite(y)||std::abs(x-y)>tol){
                    std::cerr<<"MISMATCH seed="<<seed<<" step="<<step<<" layer="<<k<<" neuron="<<j<<" kind="<<kind<<" inc="<<x<<" full="<<y<<" difference="<<(x-y)<<"\n";
                    throw std::runtime_error("differential failure");
                }
            }
        }
    }
    if(got.size()!=wanted.size())throw std::runtime_error("output shape mismatch");
}
int main(){
    const int trials=30, steps=600;
    int total_checks=0;
    for(int seed=0;seed<trials;++seed){
        std::mt19937 rng(0x51c9u+static_cast<unsigned>(seed));
        auto rnd=[&](int n)->int {return static_cast<int>(rng()%static_cast<unsigned>(n));};
        Model m("differential",2+rnd(5));
        m.mode= seed%3==0 ? Model::AUTO : Model::DELTA;
        const int nl=1+rnd(4);
        for(int k=0;k<nl;++k)
            m.layer("layer"+std::to_string(k),1+rnd(7),rnd(3),static_cast<unsigned>(100+k+seed*31),rnd(2));
        std::vector<double> input(static_cast<size_t>(m.ninput));
        for(auto& v:input)v=0.1*(rnd(13)-6);
        m.observe(input);
        equal_state(m,seed,-1);
        for(int step=0;step<steps;++step){
            const int operation=rnd(100);
            const int li=rnd(nl);
            auto &layer=m.layers[li];
            const int r=rnd(layer.out),c=rnd(layer.in);
            if(operation<53){
                int same=1+rnd(20);
                for(int j=0;j<same;++j){
                    auto &l=m.layers[li];
                    int rr=rnd(l.out),cc=rnd(l.in);
                    double delta=(rnd(2)?1:-1)*(rnd(4)==0?0.0001:0.018);
                    m.change(l.name,rr,cc,delta);
                }
            } else if(operation<66){
                m.gate(layer.name,r,c,rnd(2)==0);
            } else if(operation<74){
                auto ix=m.index(li,r,c);
                m.mutate(li,r,c,(rnd(2)?1:-1)*0.5,layer.enabled[ix]);
            } else if(operation<82){
                std::vector<double> x(static_cast<size_t>(m.ninput));
                for(auto &v:x)v=0.15*(rnd(13)-6);
                m.observe(x);
            } else if(operation<90){
                std::vector<double> t(static_cast<size_t>(m.layers.back().out));
                for(auto &v:t)v=0.05*(rnd(11)-5);
                m.learn(t,0.1);
            } else if(operation<95){
                m.checkpoint("keep"+std::to_string(step));
            } else {
                if(!m.saved.empty())m.restore(m.saved.begin()->first);
            }
            equal_state(m,seed,step);
            ++total_checks;
        }
    }
    std::cout<<"DIFFERENTIAL_STRESS_PASS seeds="<<trials<<" steps_per_seed="<<steps<<" checks="<<total_checks<<"\n";
}
