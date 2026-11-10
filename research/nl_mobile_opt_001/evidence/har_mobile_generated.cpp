#include "neural_runtime.hpp"
#include "mobile_plan.hpp"
#include "neural_dataset.hpp"
#include <exception>
static bool nl_mobile_try(neural::Model& m){
if(m.graph||m.layers.size()!=2||m.ninput!=561)return false;
if(m.layers[0].in!=561||m.layers[0].out!=16||m.layers[0].type!=3||m.layers[0].storage!=0||m.layers[0].disabled!=0)return false;
if(m.layers[1].in!=16||m.layers[1].out!=6||m.layers[1].type!=0||m.layers[1].storage!=0||m.layers[1].disabled!=0)return false;
if((int)m.z[0].size()!=16){m.z[0].assign(16,0);m.a[0].assign(16,0);}
neural::mobile_dense_ilp4<561,16,3>(m.layers[0].w.data(),m.layers[0].b.data(),m.input.data(),m.z[0].data(),m.a[0].data());
if((int)m.z[1].size()!=6){m.z[1].assign(6,0);m.a[1].assign(6,0);}
neural::mobile_dense_ilp4<16,6,0>(m.layers[1].w.data(),m.layers[1].b.data(),m.a[0].data(),m.z[1].data(),m.a[1].data());
return true;
}
int main() {
try {
neural::Model brain("HAR",561);
brain.layer("hidden",16,3,static_cast<uint32_t>(11ULL+0*31337ULL),0);
brain.layer("activity",6,0,static_cast<uint32_t>(11ULL+1*31337ULL),0);
brain.mobile_forward=&nl_mobile_try;
brain.load_model("har_model.nlw");
brain.mode=neural::Model::FULL;
neural::predict_data(brain,"har_test.txt",2947);
brain.report("har_report.json");
return 0;
} catch(const std::exception& e) {std::cerr<<"NEURAL ERROR: "<<e.what()<<"\n";return 1;}
}
