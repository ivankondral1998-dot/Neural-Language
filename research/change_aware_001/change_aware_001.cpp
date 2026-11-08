// Neural Language NL-COMP-001: independent experimental change-aware implementation.
// NOT the production Model::infer; measures a candidate effect-aware kernel against
// optimized full C++ and external BLAS with equal FP64 semantics and edit sequences.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cblas.h>

#ifdef SCIPY_BLAS
// scipy ships a fully optimized OpenBLAS with a symbol prefix (LP64 interface).
extern "C" void scipy_cblas_dgemv(const CBLAS_LAYOUT, const CBLAS_TRANSPOSE,
    const int, const int, const double, const double*, const int,
    const double*, const int, const double, double*, const int);
#define NL_CBLAS_DGEMV scipy_cblas_dgemv
#else
#define NL_CBLAS_DGEMV cblas_dgemv
#endif

namespace experiment {
using clock = std::chrono::steady_clock;
struct Edit { int row, col; double offset; };
struct Layer {
    int cols=0, rows=0;
    std::vector<double> w, bias, z, out;
    Layer(int c, int r):cols(c),rows(r),w(std::size_t(c)*r),bias(r),z(r),out(r){}
};
struct Net {
    std::vector<double> input;
    std::vector<Layer> layers;
    static Net create(int size, uint32_t seed) {
        const int ni=size==0?128:256, nh=size==0?256:512, no=size==0?32:128;
        Net n;
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> weights(-.12,.12), inputs(-.7,.7), biases(-.05,.05);
        n.input.resize(ni);
        for(auto& x:n.input)x=inputs(rng);
        n.layers.emplace_back(ni,nh);
        n.layers.emplace_back(nh,no);
        for(auto& l:n.layers){
            for(auto& w:l.w)w=weights(rng);
            for(auto& b:l.bias)b=biases(rng);
        }
        return n;
    }
    void edit(int layer,const Edit& e) {
        Layer& l=layers.at(static_cast<std::size_t>(layer));
        l.w[static_cast<std::size_t>(e.row)*l.cols+e.col]+=e.offset;
    }
    void forward_scalar(){
        for(std::size_t k=0;k<layers.size();++k){
            Layer& l=layers[k];const auto& x=k?layers[k-1].out:input;
            for(int r=0;r<l.rows;++r){
                double sum=l.bias[r];
                const double* weights=l.w.data()+static_cast<std::size_t>(r)*l.cols;
                for(int c=0;c<l.cols;++c)sum+=weights[c]*x[c];
                l.z[r]=sum;
                l.out[r]=(k+1<layers.size())?std::max(0.0,sum):sum;
            }
        }
    }
    void forward_blas(){
        for(std::size_t k=0;k<layers.size();++k){
            Layer& l=layers[k];const auto& x=k?layers[k-1].out:input;
            NL_CBLAS_DGEMV(CblasRowMajor,CblasNoTrans,l.rows,l.cols,1.,l.w.data(),l.cols,
                          x.data(),1,0.,l.z.data(),1);
            for(int r=0;r<l.rows;++r){
                l.z[r]+=l.bias[r];
                l.out[r]=(k+1<layers.size())?std::max(0.0,l.z[r]):l.z[r];
            }
        }
    }
    // update affected rows exactly from current weights and cached input,
    // then propagate the induced changes to downstream cached preactivations.
    // This avoids historical delta accumulation within the mutated layer.
    void forward_changed(int edited_layer, const std::vector<Edit>& edits){
        Layer& l=layers[static_cast<std::size_t>(edited_layer)];
        const auto& prev=edited_layer?layers[edited_layer-1].out:input;
        std::vector<uint8_t> dirty(l.rows,0);
        for(const auto& e:edits)dirty[e.row]=1;
        std::vector<std::pair<int,double>> changed;
        changed.reserve(edits.size());
        for(int r=0;r<l.rows;++r)if(dirty[r]){
            const double prior=l.out[r];
            double sum=l.bias[r];
            const double* w=l.w.data()+static_cast<std::size_t>(r)*l.cols;
            for(int c=0;c<l.cols;++c)sum+=w[c]*prev[c];
            l.z[r]=sum;
            l.out[r]=(edited_layer+1<int(layers.size()))?std::max(0.0,sum):sum;
            const double diff=l.out[r]-prior;
            if(diff!=0.)changed.emplace_back(r,diff);
        }
        for(std::size_t k=static_cast<std::size_t>(edited_layer)+1;k<layers.size();++k){
            if(changed.empty())break;
            Layer& next=layers[k];
            std::vector<std::pair<int,double>> new_changes;
            new_changes.reserve(next.rows);
            for(int r=0;r<next.rows;++r){
                double diff=0.;
                const double* weights=next.w.data()+static_cast<std::size_t>(r)*next.cols;
                for(const auto& e:changed)diff+=weights[e.first]*e.second;
                const double old=next.out[r];
                next.z[r]+=diff;
                next.out[r]=(k+1<layers.size())?std::max(0.0,next.z[r]):next.z[r];
                if(next.out[r]!=old)new_changes.emplace_back(r,next.out[r]-old);
            }
            changed.swap(new_changes);
        }
    }
    double result_checksum()const {
        double total=0.;const auto& x=layers.back().out;
        for(std::size_t i=0;i<x.size();++i)total+=x[i]*(1.+double(i)*.001);
        return total;
    }
};

std::vector<std::vector<Edit>> make_events(int size,int layer,int batch,int steps,uint32_t seed){
    const int rows=layer==0?(size==0?256:512):(size==0?32:128);
    const int cols=layer==0?(size==0?128:256):(size==0?256:512);
    std::mt19937 rng(seed ^ (uint32_t(size)*4099u + uint32_t(layer)*127u + uint32_t(batch)*313u));
    std::uniform_int_distribution<int> rr(0,rows-1),cc(0,cols-1);
    std::uniform_real_distribution<double> dd(-.003,.003);
    std::vector<std::vector<Edit>> events;
    events.reserve(steps);
    for(int i=0;i<steps;++i){
        std::vector<Edit> xs;xs.reserve(batch);
        for(int j=0;j<batch;++j)xs.push_back({rr(rng),cc(rng),dd(rng)});
        events.push_back(std::move(xs));
    }
    return events;
}
void update(Net& net,int which,int layer,const std::vector<Edit>& xs,int event){
    for(const auto& e:xs)net.edit(layer,e);
    if(which==0)net.forward_scalar();
    else if(which==1)net.forward_blas();
    else if(which==2){
        // Periodic drift guard is part of the measured algorithm.
        if((event+1)%128==0)net.forward_scalar();
        else net.forward_changed(layer,xs);
    }else throw std::runtime_error("unknown backend");
}
struct Validation { double max_relative=0; double max_abs=0; unsigned comparisons=0; };
Validation validate(int size,int layer,int steps,const std::vector<std::vector<Edit>>& events){
    std::array<Net,3> nets={Net::create(size,20261108u),Net::create(size,20261108u),Net::create(size,20261108u)};
    for(auto& n:nets)n.forward_scalar();
    Validation v;
    for(int step=0;step<steps;++step){
        for(int k=0;k<3;++k)update(nets[k],k,layer,events[step],step);
        for(std::size_t i=0;i<nets[1].layers.back().out.size();++i){
            double expected=nets[1].layers.back().out[i];
            for(int k:{0,2}){
                const double actual=nets[k].layers.back().out[i];
                if(!std::isfinite(actual)||!std::isfinite(expected))throw std::runtime_error("nonfinite prediction");
                const double diff=std::abs(actual-expected);
                const double rel=diff/(1.+std::abs(expected));
                v.max_abs=std::max(v.max_abs,diff);v.max_relative=std::max(v.max_relative,rel);
                ++v.comparisons;
                if(rel>1.e-9)throw std::runtime_error("prediction mismatch >1e-9 on trace at step "+std::to_string(step));
            }
        }
    }
    return v;
}
struct Sample {double aggregate_us=0,p95_us=0;double checksum=0;};
Sample sample(int size,int layer,int which,int steps,int warmup,
              const std::vector<std::vector<Edit>>& events){
    Net net=Net::create(size,20261108u);
    net.forward_scalar();
    for(int j=0;j<warmup;++j)update(net,which,layer,events[j],j);
    auto start=clock::now();
    double checksum=0;
    for(int j=warmup;j<warmup+steps;++j){
        update(net,which,layer,events[j],j);
        checksum+=net.result_checksum(); // ALL backends observe result equally.
    }
    auto end=clock::now();
    const double aggregate=std::chrono::duration<double,std::micro>(end-start).count()/steps;
    // Separate profiling pass so individual clock reads do not distort aggregate latency.
    std::vector<double> ticks;
    ticks.reserve(steps);
    net=Net::create(size,20261108u);net.forward_scalar();
    for(int j=0;j<warmup;++j)update(net,which,layer,events[j],j);
    for(int j=warmup;j<warmup+steps;++j){
        auto a=clock::now();
        update(net,which,layer,events[j],j);
        auto b=clock::now();
        ticks.push_back(std::chrono::duration<double,std::micro>(b-a).count());
    }
    std::sort(ticks.begin(),ticks.end());
    const auto index=static_cast<std::size_t>(std::ceil(.95*ticks.size()))-1;
    return {aggregate,ticks[index],checksum};
}
} // namespace experiment

#ifndef NEURAL_EXPERIMENT_NO_MAIN
int main(int argc,char** argv){
    using namespace experiment;
    try{
        if(argc!=7)throw std::runtime_error("usage: change_aware_001 SIZE(0|1) LAYER(0|1) BATCH REPS STEPS WARMUP");
        const int size=std::stoi(argv[1]),layer=std::stoi(argv[2]),batch=std::stoi(argv[3]);
        const int reps=std::stoi(argv[4]),steps=std::stoi(argv[5]),warmup=std::stoi(argv[6]);
        if((size!=0&&size!=1)||(layer!=0&&layer!=1)||batch<1||batch>128||reps<1||reps>20||steps<1||steps>10000||warmup<0||warmup>1000)
            throw std::runtime_error("invalid experiment bounds");
        const auto events=make_events(size,layer,batch,steps+warmup,20261108u);
        const auto checked=validate(size,layer,steps+warmup,events);
        std::cout<<std::setprecision(12)<<"CHECK\tmax_relative="<<checked.max_relative<<"\tmax_abs="<<checked.max_abs<<"\tcomparisons="<<checked.comparisons<<"\n";
        constexpr const char* labels[]={"scalar_full","blas_full","change_aware"};
        for(int trial=0;trial<reps;++trial){
            for(int x=0;x<3;++x){
                const int k=(x+trial)%3; // Rotate measurement order.
                const Sample s=sample(size,layer,k,steps,warmup,events);
                std::cout<<"SAMPLE\tsize="<<size<<"\tlayer="<<layer<<"\tedits="<<batch
                    <<"\ttrial="<<trial<<"\tmethod="<<labels[k]
                    <<"\tmean_us="<<s.aggregate_us<<"\tp95_us="<<s.p95_us
                    <<"\tchecksum="<<s.checksum<<"\n";
            }
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<"EXPERIMENT ERROR: "<<e.what()<<"\n";return 2;}
}

#endif // NEURAL_EXPERIMENT_NO_MAIN
