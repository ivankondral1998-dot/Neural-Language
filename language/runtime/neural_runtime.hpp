#pragma once
// Neural Language 0.2 adaptive native runtime, C++17. No Python interpreter in execution path.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Optional BLAS kernels. Default build remains self-contained C++17.
#if defined(NEURAL_RUNTIME_SCIPY_BLAS)
extern "C" void scipy_cblas_dgemv64_(int,int,int64_t,int64_t,double,const double*,int64_t,const double*,int64_t,double,double*,int64_t);
#elif defined(NEURAL_RUNTIME_CBLAS)
#include <cblas.h>
#endif
namespace neural {
inline constexpr bool has_blas =
#if defined(NEURAL_RUNTIME_SCIPY_BLAS) || defined(NEURAL_RUNTIME_CBLAS)
 true;
#else
 false;
#endif
inline void blas_gemv(int rows,int cols,const double* w,const double* x,double* y) {
#if defined(NEURAL_RUNTIME_SCIPY_BLAS)
 scipy_cblas_dgemv64_(101,111,rows,cols,1.0,w,cols,x,1,0.0,y,1);
#elif defined(NEURAL_RUNTIME_CBLAS)
 cblas_dgemv(CblasRowMajor,CblasNoTrans,rows,cols,1.0,w,cols,x,1,0.0,y,1);
#else
 (void)rows;(void)cols;(void)w;(void)x;(void)y;
 throw std::logic_error("BLAS backend not linked");
#endif
}
inline void require(bool p, const char* m) { if(!p) throw std::runtime_error(m); }
inline void require(bool p, const std::string& m) { if(!p) throw std::runtime_error(m); }
// Validate numeric inputs in ALL builds, including -DNDEBUG release binaries.
// Construct the error string only when a value actually violates the contract.
inline void finite(double v, const char* where) {
    if(!std::isfinite(v) || std::abs(v)>=1.0e9)
        throw std::runtime_error(std::string("invalid numeric value: ")+where);
}
inline std::string escape(const std::string& s) { std::string out; for (unsigned char c:s) {if(c=='"')out+="\\\"";else if(c=='\\')out+="\\\\";else if(c=='\n')out+="\\n";else if(c<32) {std::ostringstream o;o<<"\\u"<<std::hex<<std::setfill('0')<<std::setw(4)<<int(c);out+=o.str();}else out+=char(c);}return out;}
inline double act(double z,int type) { if(type==1) return std::max(0.0,z); if(type==2) return z>=0?1.0/(1.0+std::exp(-z)):std::exp(z)/(1.0+std::exp(z)); return z; }
inline double derivative(double z,double a,int type) {if(type==1)return z>0?1.0:0.0;if(type==2)return a*(1-a);return 1.0;}
struct Layer {
  std::string name; int in=0,out=0,type=0,storage=0; std::vector<double> w,b;std::vector<int8_t> q; double scale=1.0/127.0;std::vector<uint8_t> enabled;size_t disabled=0;
  // Exactly one owning weight buffer: either q (one byte/weight) or w (eight bytes/weight).
  size_t weight_count() const { return storage ? q.size() : w.size(); }
  double raw(size_t i) const { return storage ? double(q[i])*scale : w[i]; }
  double effective(size_t i)const { return enabled[i] ? raw(i) : 0.0; }
  void assign_weight(size_t i,double value) {
    if(storage) {
      // Fixed scale/clip range [-1, 1] for this prototype.
      // Clamp as floating point BEFORE conversion: avoid out-of-range int casts.
      const double scaled=value/scale;
      const int code=scaled<=-127.0 ? -127 : (scaled>=127.0 ? 127 : int(std::lround(scaled)));
      q[i]=static_cast<int8_t>(code);
    } else w[i]=value;
  }
};
struct Counters {size_t full=0,delta=0,train=0,events=0,auto_full=0,auto_delta=0,blas_layers=0;};
struct Snapshot {std::vector<Layer> layers;uint64_t version;std::vector<double> input;bool has_input;};
class Model {
public:
  std::string name; int ninput;std::vector<Layer> layers; std::vector<double> input;
  std::vector<std::vector<double>> z,a; bool has_input=false,cache=false;
  uint64_t version=0,cache_version=0;Counters counters;
  std::map<std::string,Snapshot> saved;
  struct Pending {int row, col;double difference;};
  std::vector<Pending> pending;
  int pending_layer=-1;
  size_t delta_streak=0;
  enum Mode {AUTO=0,FULL=1,DELTA=2};Mode mode=AUTO;
  std::vector<std::string> events;
  explicit Model(std::string n,int ni):name(std::move(n)),ninput(ni) {require(ni>0&&ni<=2048,"input dimension out of range");}
  void layer(std::string nm,int out,int type,uint32_t seed,int storage=0) {
    require(out>0&&out<=4096,"layer dimension out of range");require(layers.size()<8,"too many layers");
    require(storage==0||storage==1,"storage must be float or int8");
    int in=layers.empty()?ninput:layers.back().out;
    size_t total=static_cast<size_t>(in)*out;require(total<=4000000,"weight budget exceeded");
    Layer l;l.name=std::move(nm);l.in=in;l.out=out;l.type=type;l.storage=storage;
    if(l.storage)l.q.resize(total);else l.w.resize(total);l.enabled.assign(total,1);l.b.assign(out,0);
    std::mt19937 rng(seed);std::uniform_real_distribution<double> dis(-0.35,0.35);
    for(size_t i=0;i<total;i++)l.assign_weight(i,dis(rng));
    layers.push_back(std::move(l));
    z.resize(layers.size());a.resize(layers.size());
  }
  void observe(std::vector<double> x){require((int)x.size()==ninput,"input shape mismatch");for(double v:x)finite(v,"input");if(!has_input||x!=input) {cache=false;pending.clear();pending_layer=-1;delta_streak=0;} input=std::move(x);has_input=true;}
  // All full passes use the same FP64 weights, activation semantics and mask.
  // A BLAS matrix-vector multiply is used only when every edge is enabled.
  void forward() {
    require(has_input,"observe an input before inference");
    for(size_t k=0;k<layers.size();k++) {
      const auto& l=layers[k];const auto& prev=k?a[k-1]:input;
      if((int)z[k].size()!=l.out){z[k].assign(l.out,0);a[k].assign(l.out,0);} else {std::fill(z[k].begin(),z[k].end(),0.0);std::fill(a[k].begin(),a[k].end(),0.0);}
      const bool blas = has_blas && l.storage==0 && l.disabled==0 && l.w.size()>=256;
      if(blas) {
        blas_gemv(l.out,l.in,l.w.data(),prev.data(),z[k].data());
        counters.blas_layers++;
        for(int r=0;r<l.out;r++) {
          z[k][r]+=l.b[r];finite(z[k][r],"forward activation");a[k][r]=act(z[k][r],l.type);
        }
      } else {
        for(int r=0;r<l.out;r++){
          double total=l.b[r];size_t base=static_cast<size_t>(r)*l.in;
          for(int c=0;c<l.in;c++) total+=l.effective(base+c)*prev[c];
          finite(total,"forward activation");z[k][r]=total;a[k][r]=act(total,l.type);
        }
      }
    }
    cache=true;cache_version=version;pending.clear();pending_layer=-1;delta_streak=0;counters.full++;
  }
  size_t weight_count()const {size_t n=0;for(auto& l:layers)n+=l.weight_count();return n;}
  size_t full_cost()const {return weight_count();}
  // A batch affects one source layer. Its neuron activation deltas propagate once,
  // regardless of how many different weights in that layer were edited.
  double predicted_delta_cost(int l)const {
    double cost=6.0*pending.size()+2.0*layers[l].out;
    size_t active=std::min(pending.size(),static_cast<size_t>(layers[l].out));
    for(size_t k=l+1;k<layers.size();k++){
      cost+=double(active)*layers[k].out;
      active=static_cast<size_t>(layers[k].out);
    }
    return cost;
  }
  bool choose_delta()const {
    if(mode==FULL||!cache||pending.empty()||pending_layer<0)return false;
    if(delta_streak>=128)return false; // Bound long-term floating point drift.
    if(mode==DELTA)return true;
    if(full_cost()<1024)return false;
    // Conservative, explicit heuristic. Not a trained cost model; held-out
    // benchmarking is required before claiming cross-platform optimality.
    const double full=double(full_cost())*(has_blas?0.35:0.85)+256.0*layers.size();
    return predicted_delta_cost(pending_layer) < full*0.85;
  }
  void flush_pending() {
    if(pending.empty())return;
    if(!choose_delta()){if(mode==AUTO)counters.auto_full++;forward();return;}
    int l=pending_layer;
    auto& L=layers[l];
    auto const& src=l==0?input:a[l-1];
    std::vector<double> change(L.out,0);
    std::vector<int> active;
    for(const auto& p:pending)change[p.row]+=p.difference*src[p.col];
    for(int r=0;r<L.out;r++){
      if(change[r]==0)continue;
      double old=a[l][r];z[l][r]+=change[r];finite(z[l][r],"batch delta activation");
      a[l][r]=act(z[l][r],L.type);change[r]=a[l][r]-old;
      if(change[r]!=0)active.push_back(r);
    }
    for(size_t k=l+1;k<layers.size();k++){
      const Layer& next=layers[k];std::vector<double> newchanges(next.out,0);
      std::vector<int> next_active;
      if(active.empty())break;
      for(int r=0;r<next.out;r++){
        double dz=0;size_t base=static_cast<size_t>(r)*next.in;
        for(int c:active)dz+=next.effective(base+static_cast<size_t>(c))*change[c];
        double old=a[k][r];z[k][r]+=dz;finite(z[k][r],"batch propagated activation");
        a[k][r]=act(z[k][r],next.type);newchanges[r]=a[k][r]-old;
        if(newchanges[r]!=0)next_active.push_back(r);
      }
      change=std::move(newchanges);active=std::move(next_active);
    }
    cache=true;cache_version=version;pending.clear();pending_layer=-1;
    delta_streak++;counters.delta++;if(mode==AUTO)counters.auto_delta++;
  }
  const std::vector<double>& infer(bool announce=true){
    if(!cache)forward();
    else if(!pending.empty())flush_pending();
    else if(cache_version!=version)forward();
    require(!layers.empty(),"empty network");
    if(announce){std::cout<<"PREDICT version="<<version<<" values=";for(size_t i=0;i<a.back().size();i++)std::cout<<(i?",":"")<<std::setprecision(10)<<a.back()[i];std::cout<<"\n";}return a.back();
  }
  int layer_index(const std::string& n)const {for(size_t k=0;k<layers.size();k++)if(layers[k].name==n)return int(k);throw std::runtime_error("unknown layer: "+n);}
  size_t index(int l,int r,int c)const {require(l>=0&&(size_t)l<layers.size(),"layer index invalid");const auto& x=layers[l];require(r>=0&&r<x.out&&c>=0&&c<x.in,"weight index out of range");return static_cast<size_t>(r)*x.in+c;}
  // Weight changes are staged; flush only when a prediction/learning step needs
  // activations. Different-layer batches fall back to a safe full recompute.
  void mutate(int l,int row,int col,double new_weight,int new_gate) {
    size_t ix=index(l,row,col);Layer& L=layers[l];finite(new_weight,"weight");
    require(new_gate==0||new_gate==1,"gate invalid");
    const double old=L.effective(ix);const int old_gate=L.enabled[ix];
    // If the edit crosses layers, a single-layer batch cannot capture all
    // dependencies; invalidate and recover with a mathematically full pass.
    if(!pending.empty() && pending_layer!=l){cache=false;pending.clear();pending_layer=-1;delta_streak=0;}
    if(old_gate!=new_gate) {
      if(new_gate) {require(L.disabled>0,"inconsistent disabled count");L.disabled--;}
      else L.disabled++;
    }
    L.assign_weight(ix,new_weight);
    L.enabled[ix]=static_cast<uint8_t>(new_gate);
    // Delta must use the rounded, effective weight, not the unquantized request.
    const double updated=L.effective(ix);
    version++;counters.events++;
    if(old==updated){if(cache&&pending.empty())cache_version=version;return;}
    if(mode==FULL || !cache) {cache=false;pending.clear();pending_layer=-1;return;}
    if(pending.empty())pending_layer=l;
    pending.push_back({row,col,updated-old});
    if(pending.size()>4096) {cache=false;pending.clear();pending_layer=-1;delta_streak=0;}
  }
  void change(const std::string& layer_name,int r,int c,double d) {
    finite(d,"delta");int l=layer_index(layer_name);auto ix=index(l,r,c);mutate(l,r,c,layers[l].raw(ix)+d,layers[l].enabled[ix]);
  }
  void gate(const std::string& layer_name,int r,int c,bool on) {
    int l=layer_index(layer_name);auto ix=index(l,r,c);mutate(l,r,c,layers[l].raw(ix),on?1:0);
  }
  void learn(const std::vector<double>& target,double rate){
    require(!layers.empty(),"empty network");require(has_input,"observe before learn");require((int)target.size()==layers.back().out,"target shape mismatch");
    require(rate>0&&rate<=1,"learning rate out of range");for(double v:target)finite(v,"target");
    infer(false);
    std::vector<std::vector<double>> grad(layers.size());
    size_t last=layers.size()-1;
    grad[last].resize(layers[last].out);
    for(int i=0;i<layers[last].out;i++)grad[last][i]=2.0*(a[last][i]-target[i])/layers[last].out*derivative(z[last][i],a[last][i],layers[last].type);
    for(int k=int(last)-1;k>=0;k--){
      grad[k].assign(layers[k].out,0);
      const Layer& next=layers[k+1];
      for(int j=0;j<layers[k].out;j++){
        double v=0;for(int r=0;r<next.out;r++)v+=next.effective(static_cast<size_t>(r)*next.in+j)*grad[k+1][r];
        grad[k][j]=v*derivative(z[k][j],a[k][j],layers[k].type);
      }
    }
    // All gradients calculated using old weights; update after backpropagation.
    for(size_t k=0;k<layers.size();k++){
      Layer& l=layers[k];const auto& prev=k?a[k-1]:input;
      for(int r=0;r<l.out;r++){
        for(int c=0;c<l.in;c++){
          size_t ix=static_cast<size_t>(r)*l.in+c;
          if(l.enabled[ix]){double w=l.raw(ix)-rate*grad[k][r]*prev[c];finite(w,"updated weight");l.assign_weight(ix,w);}
        }
        double b=l.b[r]-rate*grad[k][r];finite(b,"updated bias");l.b[r]=b;
      }
    }
    version++;cache=false;pending.clear();pending_layer=-1;delta_streak=0;counters.train++;
  }
  void checkpoint(const std::string& label){require(saved.count(label)==0,"checkpoint already exists; use a new name");require(saved.size()<64,"checkpoint limit");saved[label]=Snapshot{layers,version,input,has_input};std::cout<<"CHECKPOINT "<<label<<" version="<<version<<"\n";}
  void restore(const std::string& label){auto it=saved.find(label);require(it!=saved.end(),"unknown checkpoint: "+label);layers=it->second.layers;input=it->second.input;has_input=it->second.has_input;version++;cache=false;pending.clear();pending_layer=-1;delta_streak=0;std::cout<<"RESTORE "<<label<<" version="<<version<<"\n";}
  void expect(const std::vector<double>& expected,double tol) {
    require(std::isfinite(tol) && tol>=0.0,"expect tolerance must be finite and nonnegative");
    for(double value:expected)require(std::isfinite(value),"expect target must be finite");
    const auto& got=infer(false);
    require(got.size()==expected.size(),"expect shape mismatch");
    for(size_t i=0;i<got.size();i++) {
      const double diff=std::abs(got[i]-expected[i]);
      // Negated <= ensures NaN can never produce a spurious PASS.
      if(!std::isfinite(got[i]) || !(diff<=tol))
        throw std::runtime_error("expect failed at output "+std::to_string(i));
    }
    std::cout<<"EXPECT PASS tolerance="<<tol<<"\n";
  }
  // Holdout evaluation is read-only: no input, weight, cache, counter or mode
  // changes in the user's model. The author must supply genuinely new inputs.
  void validate(const std::vector<double>& values,
                const std::vector<double>& target,double tolerance) {
    require(static_cast<int>(values.size())==ninput,"validation input shape mismatch");
    require(!layers.empty() && static_cast<int>(target.size())==layers.back().out,
            "validation target shape mismatch");
    require(std::isfinite(tolerance) && tolerance>=0,"validation tolerance invalid");
    for(double v:values)finite(v,"validation input");
    for(double v:target)finite(v,"validation target");
    Model trial=isolated_copy();
    trial.observe(values);
    const auto& result=trial.infer(false);
    double sum=0.0, maximum=0.0;
    for(size_t i=0;i<result.size();i++) {
      require(std::isfinite(result[i]),"validation result non-finite");
      const double delta=std::abs(result[i]-target[i]);
      require(std::isfinite(delta),"validation difference non-finite");
      maximum=std::max(maximum,delta);
      sum+=delta*delta;
      require(std::isfinite(sum),"validation loss non-finite");
    }
    if(maximum>tolerance)
      throw std::runtime_error("validation failed: max_abs="+std::to_string(maximum)+
                               " tolerance="+std::to_string(tolerance));
    std::ostringstream ss;
    ss<<std::setprecision(10)<<"VALIDATE PASS max_abs="<<maximum
      <<" mse="<<(sum/result.size())<<" tolerance="<<tolerance
      <<" original_unchanged=true";
    std::cout<<ss.str()<<"\n";note(ss.str());
  }
  // Opt-in native helper invoked from .nir, not a general optimizer or LLM.
  // Profile a deterministic *single-layer-change* workload on isolated copies.
  // Compare FULL to the existing guarded AUTO policy, never to an obsolete
  // per-change baseline. Accept AUTO only with repeated measurable advantage.
  // The model's weights, topology and input remain unmodified.
  void optimize(int rounds){
    require(has_input,"observe before optimize");
    require(rounds>=20 && rounds<=2000,"optimize rounds invalid");
    infer(false);
    size_t layer_id=0, largest=0;
    for(size_t i=0;i<layers.size();i++)
      if(layers[i].weight_count()>largest){largest=layers[i].weight_count();layer_id=i;}
    struct Result{double ms;std::vector<double> trajectory;};
    auto measure=[&](Mode selected,int trial)->Result {
      Model m=isolated_copy();m.mode=selected;m.counters=Counters{};
      Result result;result.trajectory.reserve(static_cast<size_t>(rounds)*layers.back().out);
      auto started=std::chrono::steady_clock::now();
      const int rows=m.layers[layer_id].out,cols=m.layers[layer_id].in;
      for(int j=0;j<rounds;j++) {
        const int row=(j*17+trial*3+1)%rows,col=(j*23+trial*7+1)%cols;
        const double delta=(j%2 ? -1.0 : 1.0)*(m.layers[layer_id].storage ? 2*m.layers[layer_id].scale : 0.0005);
        m.change(m.layers[layer_id].name,row,col,delta);
        const auto& pred=m.infer(false);
        for(double v:pred){finite(v,"optimization trajectory");result.trajectory.push_back(v);}
      }
      result.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
      return result;
    };
    std::vector<double> full_times,auto_times;
    size_t checked=0;
    for(int trial=0;trial<3;trial++){
      // Alternate order to reduce systematic clock/cache bias.
      Result f=trial%2 ? measure(FULL,trial) : Result{};
      Result a=measure(AUTO,trial);
      if(trial%2==0)f=measure(FULL,trial);
      require(f.trajectory.size()==a.trajectory.size(),"optimize shape mismatch");
      for(size_t j=0;j<f.trajectory.size();j++) {
        const double bound=1e-9*(1+std::max(std::abs(f.trajectory[j]),std::abs(a.trajectory[j])));
        require(std::abs(f.trajectory[j]-a.trajectory[j])<=bound,
                "optimize full/auto mismatch");
        checked++;
      }
      full_times.push_back(f.ms);auto_times.push_back(a.ms);
    }
    auto median=[](std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];};
    const double full=median(full_times),automatic=median(auto_times);
    bool improved=automatic<full*0.85;
    // Avoid accepting improvement driven by one noisy outlier.
    for(size_t i=0;i<full_times.size();i++)
      if(!(auto_times[i]<full_times[i]*0.95))improved=false;
    mode=improved?AUTO:FULL;
    std::ostringstream ss;
    ss<<std::setprecision(10)<<"OPTIMIZE selected="<<(improved?"AUTO":"FULL")
      <<" full_median_ms="<<full<<" auto_median_ms="<<automatic
      <<" checked="<<checked<<" equivalence=PASS"
      <<" workload=single_layer_edits"<<" weights_unchanged=true"
      <<" note=local_profile_not_general_speed_claim";
    std::cout<<ss.str()<<"\n";note(ss.str());
  }
  // Portable textual checkpoint of learned weights and edge masks.
  // No code execution or general deserialization. Shape and numeric values
  // are validated before updating the live model. This is NOT a security
  // boundary against untrusted/concurrently modified filesystem objects.
  static void safe_model_filename(const std::string& name){
    require(!name.empty() && name!="." && name.size()<=80 &&
            name.find('/')==std::string::npos && name.find('\\')==std::string::npos &&
            name.find("..") == std::string::npos,"invalid local model filename");
    for(unsigned char c:name)require(c>=32 && c!=127,"model filename contains controls");
  }
  void save_model(const std::string& requested)const {
    safe_model_filename(requested);
    std::string path=requested;
    auto occupied=[](const std::string& name){
      std::error_code ec;
      auto status=std::filesystem::symlink_status(name,ec);
      if(ec && ec!=std::errc::no_such_file_or_directory)
        throw std::runtime_error("cannot check model output");
      return std::filesystem::exists(status);
    };
    if(occupied(path)) {
      auto dot=requested.rfind('.');
      const auto stem=dot==std::string::npos?requested:requested.substr(0,dot);
      const auto ext=dot==std::string::npos?std::string{}:requested.substr(dot);
      bool found=false;
      for(int i=1;i<10000;i++){
        path=stem+"_"+std::to_string(i)+ext;
        if(!occupied(path)){found=true;break;}
      }
      require(found,"no free model filename");
    }
    std::ofstream out(path,std::ios::out);
    require(bool(out),"cannot write model");
    out<<std::setprecision(17)<<"NLMODEL2\n"<<ninput<<' '<<layers.size()<<"\n"<<int(mode)<<"\n";
    for(const auto& l:layers){
      out<<l.name<<' '<<l.in<<' '<<l.out<<' '<<l.type<<' '<<l.storage<<"\n";
      for(double v:l.b){finite(v,"saved bias");out<<v<<' ';}out<<"\n";
      for(size_t k=0;k<l.weight_count();k++){
        const double w=l.raw(k);finite(w,"saved weight");out<<w<<' ';
      }
      out<<"\n";
      for(auto g:l.enabled){out<<int(g)<<' ';}
      out<<"\n";
    }
    out<<"END\n";
    out.flush();require(bool(out),"failed to write model");
    std::cout<<"MODEL_SAVED "<<path<<" version="<<version<<"\n";
  }
  void load_model(const std::string& path){
    safe_model_filename(path);
    std::error_code ec;
    auto stat=std::filesystem::symlink_status(path,ec);
    require(!ec && std::filesystem::is_regular_file(stat),"model input must be a regular local file");
    // Never allocate based on an untrusted count: the network shape is already
    // specified by .nir and has its own resource bounds.
    std::ifstream in(path);
    require(bool(in),"cannot open model");
    std::string marker;int input_size=0;size_t layer_count=0;
    require(bool(in>>marker>>input_size>>layer_count) && marker=="NLMODEL2" &&
            input_size==ninput && layer_count==layers.size(),"model header/shape mismatch");
    int stored_mode=0;require(bool(in>>stored_mode) && stored_mode>=0 && stored_mode<=2,"invalid saved optimizer mode");
    auto staged=layers;
    for(size_t k=0;k<staged.size();k++) {
      auto& l=staged[k];std::string name;int ni=0,no=0,ty=0,st=0;
      require(bool(in>>name>>ni>>no>>ty>>st) &&
              name==l.name && ni==l.in && no==l.out && ty==l.type && st==l.storage,
              "model layer mismatch");
      for(auto& b:l.b){
        require(bool(in>>b),"truncated model bias");finite(b,"loaded bias");
      }
      for(size_t j=0;j<l.weight_count();j++){
        double w=0;require(bool(in>>w),"truncated model weight");finite(w,"loaded weight");
        l.assign_weight(j,w);
        // Check quantized weights for exact representability after a roundtrip.
        if(l.storage)require(std::abs(l.raw(j)-w)<=1e-14,"inexact quantized model weight");
      }
      l.disabled=0;
      for(auto& enabled:l.enabled){
        int gate=0;require(bool(in>>gate) && (gate==0||gate==1),"invalid model gate");
        enabled=static_cast<uint8_t>(gate);
        if(!enabled)l.disabled++;
      }
    }
    std::string end;require(bool(in>>end) && end=="END","invalid model terminator");
    in>>std::ws;require(in.peek()==std::char_traits<char>::eof(),"trailing model data");
    layers=std::move(staged);mode=static_cast<Mode>(stored_mode);version++;cache=false;pending.clear();pending_layer=-1;delta_streak=0;
    std::cout<<"MODEL_LOADED "<<path<<" version="<<version<<"\n";
  }
  void inspect()const {std::cout<<"INSPECT model="<<name<<" layers="<<layers.size()<<" weights="<<weight_count()<<" version="<<version<<" cache="<<(cache?(pending.empty()?"valid":"pending"):"invalid")<<" pending="<<pending.size()<<"\n";}
  void note(const std::string& m){events.push_back(m);}
  // Testing helpers use a state-only copy. Snapshots and event history must not
  // be duplicated: they can be substantially larger than the active network.
  Model isolated_copy()const {
    Model t(name,ninput);
    t.layers=layers;t.input=input;t.z=z;t.a=a;
    t.has_input=has_input;t.cache=cache;t.version=version;t.cache_version=cache_version;
    t.pending=pending;t.pending_layer=pending_layer;t.delta_streak=delta_streak;
    t.mode=mode;
    return t;
  }
  // Differential self-check invoked by `.nir` VERIFY. Work happens only on
  // independent copies; not a benchmark, not evidence of general correctness.
  // Compare each layer's preactivation and activation, not merely final output.
  void verify(int rounds){
    require(has_input,"observe before verify");
    require(!layers.empty(),"verify requires layers");
    require(rounds>=1 && rounds<=2000,"verify rounds outside 1..2000");
    Model incremental=isolated_copy(),reference=isolated_copy();
    incremental.mode=DELTA;reference.mode=FULL;
    incremental.infer(false);reference.infer(false);
    std::mt19937 rng(0x91e10u); // Reproducible independent of original model's RNG.
    auto random_index=[&](int upper)->int {
      return static_cast<int>(rng()%static_cast<uint32_t>(upper));
    };
    double largest=0;
    size_t values_compared=0;
    auto compare=[&](int step){
      const auto& first=incremental.infer(false);
      const auto& other=reference.infer(false);
      require(first.size()==other.size(),"verify output shape mismatch");
      for(size_t k=0;k<layers.size();k++){
        const Layer& l=incremental.layers[k];
        const Layer& r=reference.layers[k];
        require(l.weight_count()==r.weight_count(),"verify weight shape mismatch");
        for(size_t j=0;j<l.weight_count();j++){
          require(l.enabled[j]==r.enabled[j],"verify edge mismatch");
          require(l.raw(j)==r.raw(j),"verify weight mismatch");
        }
        for(int j=0;j<l.out;j++){
          for(int type=0;type<2;type++){
            const double x=type?incremental.a[k][j]:incremental.z[k][j];
            const double y=type?reference.a[k][j]:reference.z[k][j];
            require(std::isfinite(x)&&std::isfinite(y),"verify non-finite state");
            double error=std::abs(x-y);
            largest=std::max(largest,error);
            const double bound=1e-9*(1.0+std::max(std::abs(x),std::abs(y)));
            if(error>bound)throw std::runtime_error("verify numerical mismatch at step="+std::to_string(step)+" layer="+std::to_string(k)+" neuron="+std::to_string(j));
            values_compared++;
          }
        }
      }
    };
    compare(-1);
    for(int step=0;step<rounds;step++){
      const int kind=step%17;
      const int li=random_index(static_cast<int>(layers.size()));
      const Layer& l=incremental.layers[li];
      const int row=random_index(l.out),col=random_index(l.in);
      if(kind==7){
        std::vector<double> values(input.size());
        for(auto& v:values)v=0.1*(random_index(15)-7);
        incremental.observe(values);reference.observe(values);
      } else if(kind==11){
        std::vector<double> target(static_cast<size_t>(layers.back().out));
        for(auto& v:target)v=0.1*(random_index(11)-5);
        incremental.learn(target,0.1);reference.learn(target,0.1);
      } else if(kind==5 || kind==13){
        const bool enabled=kind==13;
        incremental.gate(l.name,row,col,enabled);
        reference.gate(l.name,row,col,enabled);
      } else {
        // Every fifth edit is below one INT8 quantization step, including the
        // regression case in which a rounded weight must NOT alter the cache.
        const double magnitude=l.storage ? l.scale*(step%5==0?0.1:1.5) : (step%5==0?0.00001:0.009);
        const double delta=step%2?magnitude:-magnitude;
        incremental.change(l.name,row,col,delta);
        reference.change(l.name,row,col,delta);
        if(kind==4 && layers.size()>1){
          const int second=(li+1)%static_cast<int>(layers.size());
          incremental.change(layers[second].name,0,0,0.02);
          reference.change(layers[second].name,0,0,0.02);
        }
      }
      compare(step);
    }
    require(incremental.counters.delta>0 || rounds<2,
            "verify did not exercise the incremental path");
    std::ostringstream out;
    out<<std::setprecision(10)<<"VERIFY PASS rounds="<<rounds
       <<" values_compared="<<values_compared
       <<" delta_paths="<<incremental.counters.delta
       <<" ref_full="<<reference.counters.full
       <<" max_abs_error="<<largest
       <<" original_unchanged=true";
    std::cout<<out.str()<<"\n";
    note(out.str());
  }
  // Native, non-LLM development assistant: measure two kernels, check equivalence, never edit originals.
  void advise(int rounds){
    require(has_input,"observe before advise");require(rounds>=1&&rounds<=100000,"advice rounds invalid");
    infer(false);
    auto run=[&](Mode md, bool& ok)->std::pair<double,double> {
      Model t=isolated_copy();t.mode=md;t.counters=Counters{};
      volatile double checksum=0;
      auto start=std::chrono::steady_clock::now();
      // Stay on one layer. Alternating layers forces a full recompute, so a
      // round-robin loop cannot measure incremental at all.
      size_t which=0; size_t best=0;
      for(size_t i=0;i<t.layers.size();i++) if(t.layers[i].weight_count()>best){best=t.layers[i].weight_count();which=i;}
      for(int j=0;j<rounds;j++){
        int row=(j*7+1)%t.layers[which].out,col=(j*11+1)%t.layers[which].in;
        // Quantized edits must cross at least one representable step.
        double d=((j%2)?-1:1)*(t.layers[which].storage ? 2*t.layers[which].scale : 0.0001);
        t.change(t.layers[which].name,row,col,d);
        checksum+=t.infer(false)[0];
      }
      if(md==DELTA && t.counters.delta==0 && rounds>=2)
        ok=false;
      auto end=std::chrono::steady_clock::now();
      // Independently recompute from the same final weights, not from potentially stale cache.
      auto cached=t.infer(false);t.cache=false;auto fresh=t.infer(false);
      for(size_t i=0;i<cached.size();i++)if(std::abs(cached[i]-fresh[i])>1e-8)ok=false;
      require(std::isfinite(checksum),"benchmark checksum invalid");
      return {std::chrono::duration<double,std::milli>(end-start).count(),checksum};
    };
    bool ok=true;
    const auto full_result=run(FULL,ok),delta_result=run(DELTA,ok);
    const double full=full_result.first,delta=delta_result.first;
    // Compare the full trajectory checksum too, not only final predictions.
    if(std::abs(full_result.second-delta_result.second)>1e-8*rounds)ok=false;
    require(ok,"advice numerical equivalence FAILED");
    // Claim only against a plain full pass, never against an older self.
    // advise never changes mode.
    const std::string recommendation=delta<full*0.85?"REVIEW_INCREMENTAL":"NO_CHANGE";
    std::ostringstream ss;ss<<std::setprecision(10)<<"ADVICE full_ms="<<full<<" incremental_ms="<<delta
      <<" ratio="<<full/delta<<" equal=PASS recommendation="<<recommendation
      <<" note=against_plain_full_not_old_self";
    std::cout<<ss.str()<<"\n";note(ss.str());
  }
  void report(const std::string& filename)const {
    require(!filename.empty()&&filename!="."&&filename.find('/')==std::string::npos&&filename.find('\\')==std::string::npos&&filename.find("..") ==std::string::npos&&filename.size()<=80,"report filename must be local basename");
    for(unsigned char c:filename)require(c>=32&&c!=127,"report filename contains control character");
    std::string destination=filename;
    // Preserve previous reports. A local numbered filename is chosen on collision.
    // exists(path) follows symlinks and returns false for a dangling link.
    // Treat existing links (including dangling ones) as occupied so a report
    // cannot accidentally create a target outside the working directory.
    // NOTE: this is not atomic: concurrent writers still need an OS-level
    // exclusive no-follow create in a deployment with untrusted directories.
    auto occupied=[](const std::string& path){
      std::error_code error;
      const auto status=std::filesystem::symlink_status(path,error);
      if(error && error!=std::errc::no_such_file_or_directory)
        throw std::runtime_error("cannot inspect report destination");
      return std::filesystem::exists(status);
    };
    if(occupied(destination)) {
      auto dot=filename.rfind('.');const std::string stem=dot==std::string::npos?filename:filename.substr(0,dot);
      const std::string extension=dot==std::string::npos?"":filename.substr(dot);
      bool found=false;
      for(int i=1;i<10000;i++) {
        destination=stem+"_"+std::to_string(i)+extension;
        if(!occupied(destination)){found=true;break;}
      }
      require(found,"no free report filename");
    }
    std::ofstream f(destination,std::ios::out);require(bool(f),"cannot create report");
    f<<"{\n  \"language\":\"Neural Language 0.2\",\n  \"model\":\""<<escape(name)<<"\",\n  \"version\":"<<version<<",\n  \"weights\":"<<weight_count()<<",\n";
    f<<"  \"mode\":\""<<(mode==AUTO?"AUTO":(mode==FULL?"FULL":"DELTA"))<<"\",\n";
    f<<"  \"counts\":{\"full\":"<<counters.full<<",\"delta\":"<<counters.delta<<",\"train\":"<<counters.train<<",\"changes\":"<<counters.events<<",\"auto_full\":"<<counters.auto_full<<",\"auto_delta\":"<<counters.auto_delta<<",\"blas_layers\":"<<counters.blas_layers<<"},\n";
    f<<"  \"events\":[";for(size_t i=0;i<events.size();i++)f<<(i?",":"")<<"\""<<escape(events[i])<<"\"";f<<"]\n}\n";
    require(bool(f),"report write failed");std::cout<<"REPORT "<<destination<<"\n";
  }
};
} // namespace neural
