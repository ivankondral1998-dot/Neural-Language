// NL-COMP-001B: benchmark ACTUAL Neural Language 0.3 Model against OpenBLAS.
// Preserve production header unchanged (blob ee71cdab32b248814aa23f26318130bd155c824f).
#define NEURAL_EXPERIMENT_NO_MAIN
#include "change_aware_001.cpp"
#include "language/runtime/neural_runtime.hpp"

namespace runtime_probe {
using experiment::Net;
using experiment::Edit;
using experiment::clock;
constexpr const char* method_names[4]={"runtime_FULL","runtime_AUTO","runtime_DELTA","external_BLAS"};

neural::Model make(int size){
    const int ni=size==0?128:256;
    const int nh=size==0?256:512;
    const int no=size==0?32:128;
    const Net base=Net::create(size,20261108u);
    neural::Model m("neural_runtime_original",ni);
    m.layer("hidden",nh,1,11,0);
    m.layer("output",no,0,12,0);
    for(std::size_t k=0;k<base.layers.size();++k){
        m.layers[k].w=base.layers[k].w;
        m.layers[k].b=base.layers[k].bias;
    }
    m.observe(base.input);
    m.mode=neural::Model::FULL;
    m.infer(false);
    if(!m.counters.blas_layers)throw std::runtime_error("production Model::forward did not use BLAS");
    return m;
}
std::vector<double> update_neural(neural::Model& m,int method,int layer,const std::vector<Edit>& edits){
    static constexpr const char* names[]={"hidden","output"};
    m.mode=method==0?neural::Model::FULL:(method==1?neural::Model::AUTO:neural::Model::DELTA);
    for(const auto& e:edits)m.change(names[layer],e.row,e.col,e.offset);
    return m.infer(false);
}
std::vector<double> update_experiment(Net& n,int layer,const std::vector<Edit>& edits){
    for(const auto& e:edits)n.edit(layer,e);
    n.forward_blas();
    return n.layers.back().out;
}
struct Verify{double maxrel=0,maxabs=0;uint64_t comparisons=0;};
Verify verify(int size,int layer,const std::vector<std::vector<Edit>>& events){
    std::array<neural::Model,3> nn={make(size),make(size),make(size)};
    Net external=Net::create(size,20261108u);
    external.forward_blas();
    Verify v;
    for(std::size_t step=0;step<events.size();++step){
        auto gold=update_experiment(external,layer,events[step]);
        for(int k=0;k<3;++k){
            auto actual=update_neural(nn[k],k,layer,events[step]);
            if(actual.size()!=gold.size())throw std::runtime_error("runtime wrong output shape");
            for(std::size_t i=0;i<gold.size();++i){
                const double diff=std::abs(actual[i]-gold[i]);
                const double rel=diff/(1.+std::abs(gold[i]));
                if(!std::isfinite(diff)||rel>1.e-9)throw std::runtime_error("runtime vs OpenBLAS mismatch step="+std::to_string(step)+" method="+method_names[k]);
                v.maxrel=std::max(v.maxrel,rel);v.maxabs=std::max(v.maxabs,diff);++v.comparisons;
            }
        }
    }
    return v;
}
struct Measure{double mean_us=0,p95_us=0,checksum=0;std::size_t blas_layers=0,full=0,delta=0;};
Measure measure(int size,int layer,int which,int steps,int warmup,const std::vector<std::vector<Edit>>& events){
    neural::Model model=make(size);
    Net ext=Net::create(size,20261108u);
    ext.forward_blas();
    auto advance=[&](int ix){
        if(which<3)return update_neural(model,which,layer,events[ix]);
        return update_experiment(ext,layer,events[ix]);
    };
    for(int j=0;j<warmup;++j)advance(j);
    double checksum=0;
    const auto begin=clock::now();
    for(int j=warmup;j<warmup+steps;++j){
        auto out=advance(j);
        for(std::size_t i=0;i<out.size();++i)checksum+=out[i]*(1.0+.001*double(i));
    }
    const auto end=clock::now();
    const double average=std::chrono::duration<double,std::micro>(end-begin).count()/steps;
    // Independent per-step latency loop; do not penalize aggregate with timer reads.
    model=make(size);ext=Net::create(size,20261108u);ext.forward_blas();
    for(int j=0;j<warmup;++j)advance(j);
    std::vector<double> times;times.reserve(steps);
    for(int j=warmup;j<warmup+steps;++j){
        const auto a=clock::now();(void)advance(j);const auto b=clock::now();
        times.push_back(std::chrono::duration<double,std::micro>(b-a).count());
    }
    std::sort(times.begin(),times.end());
    return {average,times[static_cast<std::size_t>(std::ceil(.95*times.size()))-1],checksum,
            model.counters.blas_layers,model.counters.full,model.counters.delta};
}
} // runtime_probe

int main(int argc,char** argv){
    try{
        if(argc!=7)throw std::runtime_error("usage: runtime_compare SIZE LAYER BATCH REPS STEPS WARMUP");
        const int size=std::stoi(argv[1]),layer=std::stoi(argv[2]),batch=std::stoi(argv[3]);
        const int reps=std::stoi(argv[4]),steps=std::stoi(argv[5]),warmup=std::stoi(argv[6]);
        if((size!=0&&size!=1)||(layer!=0&&layer!=1)||batch<1||batch>128||reps<1||reps>20||steps<1||steps>10000||warmup<0||warmup>1000)
            throw std::runtime_error("invalid experiment bounds");
        const auto events=experiment::make_events(size,layer,batch,steps+warmup,20261108u);
        const auto checked=runtime_probe::verify(size,layer,events);
        std::cout<<std::setprecision(12)<<"CHECK\tmax_relative="<<checked.maxrel<<"\tmax_abs="<<checked.maxabs<<"\tcomparisons="<<checked.comparisons<<"\n";
        for(int trial=0;trial<reps;++trial){
            for(int j=0;j<4;++j){
                int method=(j+trial)%4;
                const auto s=runtime_probe::measure(size,layer,method,steps,warmup,events);
                std::cout<<"SAMPLE\tsize="<<size<<"\tlayer="<<layer<<"\tedits="<<batch
                    <<"\ttrial="<<trial<<"\tmethod="<<runtime_probe::method_names[method]
                    <<"\tmean_us="<<s.mean_us<<"\tp95_us="<<s.p95_us
                    <<"\tchecksum="<<s.checksum<<"\tblas_layers="<<s.blas_layers
                    <<"\tfull="<<s.full<<"\tdelta="<<s.delta<<"\n";
            }
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<"RUNTIME_COMPARE ERROR: "<<e.what()<<"\n";return 2;}
}
