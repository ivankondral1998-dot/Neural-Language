#pragma once
// Neural Language 0.1 native runtime, C++17. No Python interpreter in execution path.
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

namespace neural {
inline void require(bool p, const std::string& m) { if (!p) throw std::runtime_error(m); }
inline void finite(double v, const std::string& where) { require(std::isfinite(v) && std::abs(v)<1.0e9, "invalid numeric value: " + where); }
inline std::string escape(const std::string& s) { std::string out; for (unsigned char c:s) {if(c=='"')out+="\\\"";else if(c=='\\')out+="\\\\";else if(c=='\n')out+="\\n";else if(c<32) {std::ostringstream o;o<<"\\u"<<std::hex<<std::setfill('0')<<std::setw(4)<<int(c);out+=o.str();}else out+=char(c);}return out;}
inline double act(double z,int type) { if(type==1) return std::max(0.0,z); if(type==2) return z>=0?1.0/(1.0+std::exp(-z)):std::exp(z)/(1.0+std::exp(z)); return z; }
inline double derivative(double z,double a,int type) {if(type==1)return z>0?1.0:0.0;if(type==2)return a*(1-a);return 1.0;}
struct Layer {
  std::string name; int in=0,out=0,type=0; std::vector<double> w,b;std::vector<uint8_t> enabled;
  double effective(size_t i)const{return enabled[i]?w[i]:0.0;}
};
struct Counters {size_t full=0,delta=0,train=0,events=0,auto_full=0,auto_delta=0;};
struct Snapshot {std::vector<Layer> layers;uint64_t version;std::vector<double> input;bool has_input;};
class Model {
public:
  std::string name; int ninput;std::vector<Layer> layers; std::vector<double> input;
  std::vector<std::vector<double>> z,a; bool has_input=false,cache=false;
  uint64_t version=0,cache_version=0;Counters counters;
  std::map<std::string,Snapshot> saved;
  enum Mode {AUTO=0,FULL=1,DELTA=2};Mode mode=AUTO;
  std::vector<std::string> events;
  explicit Model(std::string n,int ni):name(std::move(n)),ninput(ni) {require(ni>0&&ni<=2048,"input dimension out of range");}
  void layer(std::string nm,int out,int type,uint32_t seed) {
    require(out>0&&out<=4096,"layer dimension out of range");require(layers.size()<8,"too many layers");
    int in=layers.empty()?ninput:layers.back().out;
    size_t total=static_cast<size_t>(in)*out;require(total<=4000000,"weight budget exceeded");
    Layer l;l.name=std::move(nm);l.in=in;l.out=out;l.type=type;
    l.w.resize(total);l.enabled.assign(total,1);l.b.assign(out,0);
    std::mt19937 rng(seed);std::uniform_real_distribution<double> dis(-0.35,0.35);
    for(double& w:l.w)w=dis(rng);
    layers.push_back(std::move(l));
    z.resize(layers.size());a.resize(layers.size());
  }
  size_t weight_count()const {size_t n=0;for(auto& l:layers)n+=l.w.size();return n;}
  void observe(std::vector<double> x){require((int)x.size()==ninput,"input shape mismatch");for(double v:x)finite(v,"input");if(!has_input||x!=input) cache=false; input=std::move(x);has_input=true;}
  void forward() {
    require(has_input,"observe an input before inference");
    for(size_t k=0;k<layers.size();k++) {
      const auto& l=layers[k];const auto& prev=k? a[k-1]:input;
      z[k].assign(l.out,0);a[k].assign(l.out,0);
      for(int r=0;r<l.out;r++){
        double total=l.b[r];size_t base=static_cast<size_t>(r)*l.in;
        for(int c=0;c<l.in;c++) total+=l.effective(base+c)*prev[c];
        finite(total,"forward activation");z[k][r]=total;a[k][r]=act(total,l.type);
      }
    }
    cache=true;cache_version=version;counters.full++;
  }
  const std::vector<double>& infer(bool announce=true){if(!cache||cache_version!=version)forward();require(!layers.empty(),"empty network");if(announce){std::cout<<"PREDICT version="<<version<<" values=";for(size_t i=0;i<a.back().size();i++)std::cout<<(i?",":"")<<std::setprecision(10)<<a.back()[i];std::cout<<"\n";}return a.back();}
  int layer_index(const std::string& n)const {for(size_t k=0;k<layers.size();k++)if(layers[k].name==n)return int(k);throw std::runtime_error("unknown layer: "+n);}
  size_t index(int l,int r,int c)const {require(l>=0&&(size_t)l<layers.size(),"layer index invalid");const auto& x=layers[l];require(r>=0&&r<x.out&&c>=0&&c<x.in,"weight index out of range");return static_cast<size_t>(r)*x.in+c;}
  size_t full_cost()const {return weight_count();}
  size_t delta_cost(int l)const {size_t cost=1;for(size_t k=l+1;k<layers.size();k++)cost+=layers[k].w.size();return cost;}
  bool can_delta(int l)const {
    if(!cache||cache_version!=version)return false;
    if(mode==FULL)return false;
    if(mode==DELTA)return true;
    // Conservative static estimate; deliberately no promise of optimality.
    return full_cost()>128 && delta_cost(l)*3<full_cost();
  }
  // Exactly one effective weight changes. Cached preactivations carry the correct ReLU sign.
  void mutate(int l,int row,int col,double new_weight,int new_gate) {
    size_t ix=index(l,row,col);Layer& L=layers[l];finite(new_weight,"weight");
    require(new_gate==0||new_gate==1,"gate invalid");
    const double old=L.effective(ix);
    const bool incremental=can_delta(l);
    L.w[ix]=new_weight;L.enabled[ix]=(uint8_t)new_gate;
    const double diff=L.effective(ix)-old;
    version++;counters.events++;
    if(incremental) {
      double source=l==0?input[col]:a[l-1][col];
      const double oldact=a[l][row];z[l][row]+=diff*source;finite(z[l][row],"delta activation");a[l][row]=act(z[l][row],L.type);
      // Propagate a changed activation through all dependent neurons.
      int changed=row;double actdiff=a[l][row]-oldact;
      for(size_t k=l+1;k<layers.size();k++){
        auto& next=layers[k];std::vector<double> changes(next.out,0);
        for(int r=0;r<next.out;r++){
          double dz=next.effective(static_cast<size_t>(r)*next.in+changed)*actdiff;
          z[k][r]+=dz;finite(z[k][r],"propagated activation");
          double newa=act(z[k][r],next.type);changes[r]=newa-a[k][r];a[k][r]=newa;
        }
        // Future downstream layers require all changed sources, not just one.
        for(size_t future=k+1;future<layers.size();future++){
          auto& dest=layers[future];std::vector<double> newchanges(dest.out,0);
          for(int r=0;r<dest.out;r++){
            double dz=0;for(int c=0;c<dest.in;c++)dz+=dest.effective(static_cast<size_t>(r)*dest.in+c)*changes[c];
            z[future][r]+=dz;finite(z[future][r],"propagated activation");
            double newa=act(z[future][r],dest.type);newchanges[r]=newa-a[future][r];a[future][r]=newa;
          }
          changes=std::move(newchanges);
        }
        break;
      }
      cache_version=version;counters.delta++;if(mode==AUTO)counters.auto_delta++;
    }else{cache=false;if(mode==AUTO)counters.auto_full++;}
  }
  void change(const std::string& layer_name,int r,int c,double d) {
    finite(d,"delta");int l=layer_index(layer_name);auto ix=index(l,r,c);mutate(l,r,c,layers[l].w[ix]+d,layers[l].enabled[ix]);
  }
  void gate(const std::string& layer_name,int r,int c,bool on) {
    int l=layer_index(layer_name);auto ix=index(l,r,c);mutate(l,r,c,layers[l].w[ix],on?1:0);
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
          if(l.enabled[ix]){double w=l.w[ix]-rate*grad[k][r]*prev[c];finite(w,"updated weight");l.w[ix]=w;}
        }
        double b=l.b[r]-rate*grad[k][r];finite(b,"updated bias");l.b[r]=b;
      }
    }
    version++;cache=false;counters.train++;
  }
  void checkpoint(const std::string& label){require(saved.count(label)==0,"checkpoint already exists; use a new name");require(saved.size()<64,"checkpoint limit");saved[label]=Snapshot{layers,version,input,has_input};std::cout<<"CHECKPOINT "<<label<<" version="<<version<<"\n";}
  void restore(const std::string& label){auto it=saved.find(label);require(it!=saved.end(),"unknown checkpoint: "+label);layers=it->second.layers;input=it->second.input;has_input=it->second.has_input;version++;cache=false;std::cout<<"RESTORE "<<label<<" version="<<version<<"\n";}
  void expect(const std::vector<double>& expected,double tol) {
    auto got=infer(false);require(got.size()==expected.size(),"expect shape mismatch");
    for(size_t i=0;i<got.size();i++) if(std::abs(got[i]-expected[i])>tol)throw std::runtime_error("expect failed at output "+std::to_string(i));
    std::cout<<"EXPECT PASS tolerance="<<tol<<"\n";
  }
  void inspect()const {std::cout<<"INSPECT model="<<name<<" layers="<<layers.size()<<" weights="<<weight_count()<<" version="<<version<<" cache="<<(cache?"valid":"invalid")<<"\n";}
  void note(const std::string& m){events.push_back(m);}
  // Native, non-LLM development assistant: measure two kernels, check equivalence, never edit originals.
  void advise(int rounds){
    require(has_input,"observe before advise");require(rounds>=1&&rounds<=100000,"advice rounds invalid");
    infer(false);
    auto run=[&](Mode md, bool& ok)->double {
      Model t=*this;t.mode=md;t.saved.clear();t.events.clear();t.counters=Counters{};
      volatile double checksum=0;
      auto start=std::chrono::steady_clock::now();
      for(int j=0;j<rounds;j++){
        size_t which=static_cast<size_t>(j)%t.layers.size();
        int row=(j*7+1)%t.layers[which].out,col=(j*11+1)%t.layers[which].in;
        double d=((j%2)?-1:1)*0.0001;
        t.change(t.layers[which].name,row,col,d);
        checksum+=t.infer(false)[0];
      }
      auto end=std::chrono::steady_clock::now();
      // Independently recompute from the same final weights, not from potentially stale cache.
      auto cached=t.infer(false);t.cache=false;auto fresh=t.infer(false);
      for(size_t i=0;i<cached.size();i++)if(std::abs(cached[i]-fresh[i])>1e-8)ok=false;
      require(std::isfinite(checksum),"benchmark checksum invalid");
      return std::chrono::duration<double,std::milli>(end-start).count();
    };
    bool ok=true;double full=run(FULL,ok),delta=run(DELTA,ok);
    require(ok,"advice numerical equivalence FAILED");
    // A single short benchmark isn't statistically sufficient to claim any speedup.
    const std::string recommendation=delta<full*0.85?"REVIEW_INCREMENTAL":"NO_CHANGE";
    std::ostringstream ss;ss<<std::setprecision(10)<<"ADVICE full_ms="<<full<<" incremental_ms="<<delta
      <<" ratio="<<full/delta<<" equal=PASS recommendation="<<recommendation;
    std::cout<<ss.str()<<"\n";note(ss.str());
  }
  void report(const std::string& filename)const {
    require(!filename.empty()&&filename.find('/')==std::string::npos&&filename.find('\\')==std::string::npos&&filename.find("..") ==std::string::npos&&filename.size()<=80,"report filename must be local basename");
    std::string destination=filename;
    // Preserve previous reports. A local numbered filename is chosen on collision.
    if(std::filesystem::exists(destination)) {
      auto dot=filename.rfind('.');const std::string stem=dot==std::string::npos?filename:filename.substr(0,dot);
      const std::string extension=dot==std::string::npos?"":filename.substr(dot);
      bool found=false;
      for(int i=1;i<10000;i++) {
        destination=stem+"_"+std::to_string(i)+extension;
        if(!std::filesystem::exists(destination)){found=true;break;}
      }
      require(found,"no free report filename");
    }
    std::ofstream f(destination,std::ios::out);require(bool(f),"cannot create report");
    f<<"{\n  \"language\":\"Neural Language 0.1\",\n  \"model\":\""<<escape(name)<<"\",\n  \"version\":"<<version<<",\n  \"weights\":"<<weight_count()<<",\n";
    f<<"  \"counts\":{\"full\":"<<counters.full<<",\"delta\":"<<counters.delta<<",\"train\":"<<counters.train<<",\"changes\":"<<counters.events<<"},\n";
    f<<"  \"events\":[";for(size_t i=0;i<events.size();i++)f<<(i?",":"")<<"\""<<escape(events[i])<<"\"";f<<"]\n}\n";
    require(bool(f),"report write failed");std::cout<<"REPORT "<<destination<<"\n";
  }
};
} // namespace neural
