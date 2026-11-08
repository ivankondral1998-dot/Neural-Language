
#include "neural_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
int main() {
    neural::Model m("int8",2);
    m.layer("h",4,1,37,1);m.layer("out",1,0,19,1);
    if(!m.layers[0].w.empty()||!m.layers[1].w.empty())return 10;
    if(m.layers[0].q.size()!=8||m.layers[1].q.size()!=4)return 11;
    m.observe({0.7,-0.1});m.mode=neural::Model::DELTA;
    m.infer(false);
    for(int i=0;i<50;i++) {
        auto& l=m.layers[i%2]; int r=i%l.out,c=i%l.in;
        m.change(l.name,r,c,(i%3 ? .001 : .035));
        double got=m.infer(false)[0];
        m.cache=false;double full=m.infer(false)[0];
        if(std::abs(got-full)>1e-10) {std::cerr<<"Divergence "<<i<<" "<<got<<" "<<full<<"\n";return 12;}
    }
    m.checkpoint("saved");
    double before=m.infer(false)[0];
    m.change("out",0,0,.2);m.infer(false);
    m.restore("saved");
    if(std::abs(m.infer(false)[0]-before)>1e-10)return 13;
    m.learn({0.8},0.5);
    double trained=m.infer(false)[0];
    if(!std::isfinite(trained))return 14;
    m.advise(30);
    for(const auto& l:m.layers) if(!l.w.empty())return 15;
    // Saturating int8 changes must stay defined (no floating-to-int overflow).
    m.mode=neural::Model::DELTA;
    m.change("out",0,0,10000.0);
    double saturated_delta=m.infer(false)[0];
    m.cache=false;double saturated_full=m.infer(false)[0];
    if(std::abs(saturated_delta-saturated_full)>1e-10)return 17;
    std::cout<<"INT8_SHADOW_FREE_AND_DELTA_EQUAL_PASS\n";
    // A single quantization-null edit must not fake a delta in a one-layer net.
    neural::Model n("single",1);n.layer("out",1,0,17,1);
    n.observe({1.0});n.mode=neural::Model::DELTA;n.infer(false);
    n.change("out",0,0,.001);
    double inc=n.infer(false)[0];n.cache=false;double fresh=n.infer(false)[0];
    if(std::abs(inc-fresh)>1e-12)return 16;
    std::cout<<"QUANTIZATION_ROUNDING_DELTA_EQUAL_PASS\n";
}
