// First paired mobile performance experiment for the same pretrained .nir policy.
// Neural Language C++17 runtime versus independent minimal optimized C++17 forward.
// Tested model topology: 4 -> 64 tanh -> 64 tanh -> 2 linear, FP64, dense.
// The comparator intentionally does not use neural_runtime.hpp to parse or infer.
// Build both in ONE executable with the same compiler/flags. This is NOT
// TFLite / ONNX / BLAS and does not establish general performance superiority.
#include "neural_runtime.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using State = std::array<double, 4>;
using Scores = std::array<double, 2>;
using Clock = std::chrono::steady_clock;
void ensure(bool ok, const char* msg) {
  if (!ok) throw std::runtime_error(msg);
}
struct PlainLayer {
  int ni = 0, no = 0, activation = 0;
  std::vector<double> bias, weights;
};
struct PlainCpp {
  std::array<PlainLayer, 3> layers;
  explicit PlainCpp(const std::string& path) {
    std::ifstream f(path);
    ensure(bool(f), "cannot open baseline checkpoint");
    std::string marker; int input = 0, count = 0, mode = 0;
    ensure(bool(f >> marker >> input >> count >> mode), "bad model header");
    ensure(marker == "NLMODEL2" && input == 4 && count == 3 && mode >= 0 && mode <= 2,
           "unexpected model format or topology");
    constexpr int dims[4] = {4, 64, 64, 2};
    constexpr int types[3] = {3, 3, 0};
    constexpr const char* names[3] = {"hidden1", "hidden2", "actions"};
    for (int k = 0; k < 3; ++k) {
      auto& layer = layers[static_cast<size_t>(k)];
      std::string name; int storage = -1;
      ensure(bool(f >> name >> layer.ni >> layer.no >> layer.activation >> storage),
             "truncated baseline layer header");
      ensure(name == names[k] && layer.ni == dims[k] && layer.no == dims[k+1] &&
             layer.activation == types[k] && storage == 0,
             "baseline incompatible layer");
      layer.bias.resize(static_cast<size_t>(layer.no));
      layer.weights.resize(static_cast<size_t>(layer.ni) * layer.no);
      for (double& v : layer.bias) ensure(bool(f >> v) && std::isfinite(v),
                                          "bad baseline bias");
      for (double& v : layer.weights) ensure(bool(f >> v) && std::isfinite(v),
                                             "bad baseline weight");
      for (size_t i=0;i<layer.weights.size();++i) {
        int gate = -1;
        ensure(bool(f >> gate) && gate == 1, "baseline expects fully enabled dense model");
      }
    }
    std::string end;
    ensure(bool(f >> end) && end == "END", "model missing END");
    f >> std::ws;
    ensure(f.peek() == std::char_traits<char>::eof(), "unexpected trailing model bytes");
  }
  Scores predict(const State& observation) const {
    std::array<double,64> h1{}, h2{};
    const auto& l0 = layers[0];
    for(int r=0;r<64;++r) {
      double z=l0.bias[static_cast<size_t>(r)];
      for(int c=0;c<4;++c)z+=l0.weights[static_cast<size_t>(r)*4+c]*observation[c];
      h1[r]=std::tanh(z);
    }
    const auto& l1=layers[1];
    for(int r=0;r<64;++r) {
      double z=l1.bias[static_cast<size_t>(r)];
      for(int c=0;c<64;++c)z+=l1.weights[static_cast<size_t>(r)*64+c]*h1[c];
      h2[r]=std::tanh(z);
    }
    const auto& l2=layers[2];
    Scores result{};
    for(int r=0;r<2;++r) {
      double z=l2.bias[static_cast<size_t>(r)];
      for(int c=0;c<64;++c)z+=l2.weights[static_cast<size_t>(r)*64+c]*h2[c];
      result[r]=z;
    }
    return result;
  }
};
neural::Model load_neural(const std::string& model) {
  neural::Model m("CartPolePretrained",4);
  m.layer("hidden1",64,3,42);
  m.layer("hidden2",64,3,43);
  m.layer("actions",2,0,44);
  m.load_model(model);
  m.mode=neural::Model::FULL;
  return m;
}
std::vector<State> states(int n) {
  std::vector<State> out;
  out.reserve(static_cast<size_t>(n));
  std::uint32_t seed=0xC0FFEE11u;
  for(int i=0;i<n;++i) {
    State s{};
    for(auto& x:s) {
      seed=1664525u*seed+1013904223u;
      x=(static_cast<double>(seed)/4294967296.0-0.5)*0.1;
    }
    out.push_back(s);
  }
  return out;
}
struct Golden {
  State x; Scores y;
};
constexpr Golden goldens[] = {
  {{0.,0.,0.,0.}, {0.12913951301483756,-0.12815577824153035}},
  {{0.02,-0.01,0.04,-0.03}, {-0.3736812805283707,0.3745496992378324}},
  {{-0.01,0.04,-0.04,0.01}, {0.7849797970119223,-0.7840781729061173}},
  {{0.05,0.05,0.05,0.05}, {-2.051765586405924,2.053296764932673}},
  {{-0.05,-0.05,-0.05,-0.05}, {2.346353593857841,-2.3461097929302865}}
};
double error_bound(double x, double y) {
  return 1e-9*(1.0+std::max(std::abs(x),std::abs(y)));
}
void check_pair(const Scores& a,const Scores& b,double& maxerr) {
  for(int j=0;j<2;++j) {
    ensure(std::isfinite(a[j]) && std::isfinite(b[j]),"non-finite inference");
    const double d=std::abs(a[j]-b[j]);
    maxerr=std::max(maxerr,d);
    ensure(d<=error_bound(a[j],b[j]),"reference and neural mismatch");
  }
}
double median(std::vector<double> v) {
  ensure(!v.empty(),"no timings");
  std::sort(v.begin(),v.end());return v[v.size()/2];
}
double elapsed_us(Clock::time_point start) {
  return std::chrono::duration<double,std::micro>(Clock::now()-start).count();
}
double run_neural(neural::Model& model,const std::vector<State>& data,double& sink) {
  const auto start=Clock::now();
  double checksum=0.;
  for(const auto& s:data) {
    model.observe(std::vector<double>(s.begin(),s.end()));
    const auto& y=model.infer(false);
    checksum+=y[0]+y[1]*0.125;
  }
  const double us=elapsed_us(start);
  ensure(std::isfinite(checksum),"invalid Neural checksum");
  sink+=checksum;
  return us/static_cast<double>(data.size());
}
double run_cpp(const PlainCpp& model,const std::vector<State>& data,double& sink) {
  const auto start=Clock::now();
  double checksum=0.;
  for(const auto& s:data) {
    const auto y=model.predict(s);
    checksum+=y[0]+y[1]*0.125;
  }
  const double us=elapsed_us(start);
  ensure(std::isfinite(checksum),"invalid baseline checksum");
  sink+=checksum;
  return us/static_cast<double>(data.size());
}
int number(const char* raw,int low,int high) {
  const std::string s(raw);
  ensure(!s.empty() && s.find_first_not_of("0123456789")==std::string::npos,
         "argument must be unsigned decimal");
  const long n=std::stol(s);
  ensure(n>=low && n<=high,"argument out of allowed range");
  return static_cast<int>(n);
}
} // namespace

int main(int argc,char**argv) {
  try {
    ensure(argc==2 || argc==4,"usage: bench_cartpole_compare MODEL.nlw [SAMPLES TRIALS]");
    const int n=argc==4?number(argv[2],256,50000):4096;
    const int trials=argc==4?number(argv[3],3,31):7;
    const PlainCpp cpp(argv[1]);
    auto neural=load_neural(argv[1]);
    const auto data=states(n);
    double maxerr=0.;
    for(const auto& test:goldens) {
      const auto expected=cpp.predict(test.x);
      neural.observe(std::vector<double>(test.x.begin(),test.x.end()));
      const auto& actual=neural.infer(false);
      check_pair(expected,{actual[0],actual[1]},maxerr);
      check_pair(expected,test.y,maxerr);
    }
    for(int i=0;i<std::min(512,n);++i) {
      const auto expected=cpp.predict(data[static_cast<size_t>(i)]);
      neural.observe(std::vector<double>(data[static_cast<size_t>(i)].begin(),
                                         data[static_cast<size_t>(i)].end()));
      const auto& actual=neural.infer(false);
      check_pair(expected,{actual[0],actual[1]},maxerr);
    }
    std::cout<<std::setprecision(12)<<"REFERENCE_EQUIVALENCE_PASS vectors="
             <<(std::min(512,n)+5)<<" max_abs_error="<<maxerr<<'\n';
    // Repeated states, equal sample stream, no loading/printing in timed region.
    // The two alternatives run in alternating order to limit order bias.
    std::vector<double> nt,ct;nt.reserve(trials);ct.reserve(trials);
    double neural_sink=0.,cpp_sink=0.;
    const auto warmup=states(256);
    run_cpp(cpp,warmup,cpp_sink);
    run_neural(neural,warmup,neural_sink);
    for(int t=0;t<trials;++t) {
      double a=0.,b=0.;
      if(t%2==0) {
        a=run_neural(neural,data,neural_sink);
        b=run_cpp(cpp,data,cpp_sink);
      } else {
        b=run_cpp(cpp,data,cpp_sink);
        a=run_neural(neural,data,neural_sink);
      }
      nt.push_back(a);ct.push_back(b);
      std::cout<<"TRIAL index="<<t+1<<" first="<<(t%2==0?"neural":"plain_cpp")
               <<" neural_us_per_infer="<<a
               <<" plain_cpp_us_per_infer="<<b<<'\n';
    }
    // Both engines have processed the same inputs and the same number of
    // iterations; per-engine checksums must also match to FP tolerance.
    const double checksum_error=std::abs(neural_sink-cpp_sink);
    ensure(checksum_error<=error_bound(neural_sink,cpp_sink),"timed checksum mismatch");
    const double nmed=median(nt),cmed=median(ct);
    std::cout<<"BENCH_PASS samples="<<n<<" trials="<<trials
             <<" neural_median_us_per_infer="<<nmed
             <<" plain_cpp_median_us_per_infer="<<cmed
             <<" neural_speedup_vs_plain_cpp="<<cmed/nmed
             <<" max_abs_error="<<maxerr
             <<" checksum_abs_error="<<checksum_error
             <<" engine_neural=FULL engine_baseline=independent_cpp_fp64"
             <<" note=mobile_on_device_needed_no_tflite_comparison\n";
    return 0;
  }catch(const std::exception& e) {
    std::cerr<<"BENCH_ERROR "<<e.what()<<'\n';
    return 2;
  }
}
