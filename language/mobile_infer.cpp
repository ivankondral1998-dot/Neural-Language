// Neural Language: lightweight native inference for trusted NLMODEL2 checkpoints.
// C++17; no Python, BLAS, training state, networking, or third-party dependencies.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace nlmobile {
inline void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
inline bool valid(double x) { return std::isfinite(x) && std::abs(x) < 1e9; }

struct Layer {
    int in=0, out=0, activation=0, storage=0;
    std::vector<double> weights, bias;
    std::vector<int8_t> compact;
    std::vector<uint8_t> gates;
};
struct Model {
    int input_size=0;
    std::vector<Layer> layers;
    std::size_t weights=0, payload_bytes=0, activation_bytes=0;

    static Model load(const std::string& path, std::size_t max_bytes=256u*1024u*1024u) {
        // Files are expected to be locally generated and trusted; this is not a sandbox.
        std::ifstream in(path);
        check(bool(in), "cannot open checkpoint");
        std::string magic;
        int inputs=0, count=0, mode=0;
        check(bool(in>>magic>>inputs>>count>>mode), "truncated checkpoint header");
        check(magic=="NLMODEL2", "requires sequential NLMODEL2 checkpoint");
        check(inputs>=1 && inputs<=2048 && count>=1 && count<=8 && mode>=0 && mode<=2,
              "invalid checkpoint dimensions or mode");
        Model model;
        model.input_size=inputs;
        model.layers.reserve(static_cast<std::size_t>(count));
        std::unordered_set<std::string> seen;
        int previous=inputs;
        std::size_t max_pair=static_cast<std::size_t>(inputs), current=static_cast<std::size_t>(inputs);
        for(int k=0;k<count;++k) {
            std::string name;
            Layer layer;
            check(bool(in>>name>>layer.in>>layer.out>>layer.activation>>layer.storage), "truncated layer header");
            check(!name.empty() && seen.insert(name).second, "duplicate or empty layer name");
            check(layer.in==previous && layer.out>=1 && layer.out<=4096, "layer shape mismatch");
            check(layer.activation>=0 && layer.activation<=2 && (layer.storage==0 || layer.storage==1),
                  "invalid activation/storage kind");
            std::size_t n=static_cast<std::size_t>(layer.in)*static_cast<std::size_t>(layer.out);
            check(n<=4000000u && model.weights+n<=4000000u, "weight limit exceeded");
            std::size_t bytes=n*(layer.storage?1u:8u)+n+8u*static_cast<std::size_t>(layer.out);
            check(model.payload_bytes+bytes<=max_bytes, "tensor byte limit exceeded");
            model.weights+=n;
            model.payload_bytes+=bytes;
            layer.bias.resize(static_cast<std::size_t>(layer.out));
            for(double& b:layer.bias) check(bool(in>>b) && valid(b), "invalid bias");
            if(layer.storage)layer.compact.resize(n); else layer.weights.resize(n);
            for(std::size_t j=0;j<n;++j){
                double w=0;
                check(bool(in>>w) && valid(w), "invalid weight");
                if(layer.storage){
                    double scaled=w*127.0;
                    check(scaled>=-127.000001 && scaled<=127.000001, "INT8 range error");
                    int q=static_cast<int>(std::lround(std::max(-127.0,std::min(127.0,scaled))));
                    check(std::abs(double(q)*(1.0/127.0)-w)<=1e-14, "nonrepresentable INT8 weight");
                    layer.compact[j]=static_cast<int8_t>(q);
                } else layer.weights[j]=w;
            }
            layer.gates.resize(n);
            for(uint8_t& gate:layer.gates){
                int x=-1;
                check(bool(in>>x) && (x==0 || x==1), "invalid connection mask");
                gate=static_cast<uint8_t>(x);
            }
            max_pair=std::max(max_pair,current+static_cast<std::size_t>(layer.out));
            current=static_cast<std::size_t>(layer.out);
            previous=layer.out;
            model.layers.push_back(std::move(layer));
        }
        std::string end;
        check(bool(in>>end) && end=="END", "missing END marker");
        in>>std::ws;
        check(in.peek()==std::char_traits<char>::eof(), "trailing checkpoint data");
        model.activation_bytes=8u*max_pair;
        check(model.payload_bytes+model.activation_bytes<=max_bytes, "activation byte limit exceeded");
        return model;
    }

    std::vector<double> predict(const std::vector<double>& input) const {
        check(input.size()==static_cast<std::size_t>(input_size), "input shape mismatch");
        for(double x:input)check(valid(x), "invalid input number");
        std::vector<double> current=input, next;
        for(const Layer& layer:layers){
            next.resize(static_cast<std::size_t>(layer.out));
            for(int r=0;r<layer.out;++r){
                double z=layer.bias[static_cast<std::size_t>(r)];
                std::size_t base=static_cast<std::size_t>(r)*static_cast<std::size_t>(layer.in);
                for(int c=0;c<layer.in;++c){
                    std::size_t index=base+static_cast<std::size_t>(c);
                    if(layer.gates[index]){
                        double w=layer.storage ? double(layer.compact[index])*(1.0/127.0) : layer.weights[index];
                        z+=w*current[static_cast<std::size_t>(c)];
                    }
                }
                check(valid(z), "invalid activation");
                if(layer.activation==1) z=std::max(0.0,z);
                if(layer.activation==2) z=z>=0 ? 1.0/(1.0+std::exp(-z)) : std::exp(z)/(1.0+std::exp(z));
                next[static_cast<std::size_t>(r)]=z;
            }
            current.swap(next);
            next.clear();
        }
        return current;
    }
};

std::vector<double> parse_csv(const std::string& source){
    check(!source.empty(), "missing input numbers");
    std::vector<double> values;
    std::size_t start=0;
    while(start<source.size()){
        auto stop=source.find(',',start);
        std::string token=source.substr(start,stop==std::string::npos?std::string::npos:stop-start);
        check(!token.empty(), "empty input number");
        std::size_t consumed=0;
        double value=0;
        try { value=std::stod(token,&consumed); }
        catch(const std::exception&){throw std::runtime_error("invalid input literal");}
        check(consumed==token.size() && valid(value), "invalid input literal");
        values.push_back(value);
        check(values.size()<=2048u, "too many input values");
        if(stop==std::string::npos) break;
        start=stop+1;
        check(start<source.size(), "trailing comma");
    }
    return values;
}
} // namespace nlmobile

// Stable C ABI for Android NDK/JNI bridges and other native embedding layers.
// The caller owns the returned handle and must call nlmobile_close.
namespace {
thread_local std::string nlmobile_error;
}
extern "C" {
void* nlmobile_open(const char* path, std::size_t byte_limit) noexcept {
    try {
        nlmobile_error.clear();
        nlmobile::check(path != nullptr, "model path is null");
        return new nlmobile::Model(nlmobile::Model::load(path, byte_limit));
    } catch (const std::exception& e) {
        nlmobile_error = e.what();
        return nullptr;
    }
}
std::size_t nlmobile_input_count(const void* handle) noexcept {
    return handle ? static_cast<const nlmobile::Model*>(handle)->input_size : 0;
}
std::size_t nlmobile_output_count(const void* handle) noexcept {
    const auto* model = static_cast<const nlmobile::Model*>(handle);
    return model && !model->layers.empty() ? static_cast<std::size_t>(model->layers.back().out) : 0;
}
std::size_t nlmobile_tensor_payload_bytes(const void* handle) noexcept {
    const auto* model = static_cast<const nlmobile::Model*>(handle);
    return model ? model->payload_bytes + model->activation_bytes : 0;
}
int nlmobile_predict(void* handle, const double* input, std::size_t input_count,
                     double* output, std::size_t output_capacity) noexcept {
    try {
        nlmobile_error.clear();
        nlmobile::check(handle != nullptr && input != nullptr && output != nullptr, "null inference argument");
        auto* model = static_cast<nlmobile::Model*>(handle);
        nlmobile::check(input_count == static_cast<std::size_t>(model->input_size), "input shape mismatch");
        const std::vector<double> values(input, input + input_count);
        const auto prediction = model->predict(values);
        nlmobile::check(output_capacity >= prediction.size(), "output buffer too small");
        std::copy(prediction.begin(), prediction.end(), output);
        return static_cast<int>(prediction.size());
    } catch (const std::exception& e) {
        nlmobile_error = e.what();
        return -1;
    }
}
const char* nlmobile_last_error() noexcept { return nlmobile_error.c_str(); }
void nlmobile_close(void* handle) noexcept { delete static_cast<nlmobile::Model*>(handle); }
}

#ifndef NL_MOBILE_LIBRARY_ONLY
int main(int argc,char** argv){
    try {
        std::string path, data;
        std::size_t budget=256u*1024u*1024u;
        bool stats=false;
        for(int i=1;i<argc;++i){
            std::string option=argv[i];
            if(option=="--model" && i+1<argc) path=argv[++i];
            else if(option=="--input" && i+1<argc) data=argv[++i];
            else if(option=="--stats")stats=true;
            else if(option=="--max-tensor-bytes" && i+1<argc){
                std::string arg=argv[++i];
                nlmobile::check(!arg.empty() && arg.find_first_not_of("0123456789")==std::string::npos,
                                "invalid byte budget");
                std::size_t pos=0;
                try { budget=static_cast<std::size_t>(std::stoull(arg,&pos)); }
                catch(const std::exception&){throw std::runtime_error("invalid byte budget");}
                nlmobile::check(pos==arg.size() && budget>0, "invalid byte budget");
            } else throw std::runtime_error("usage: mobile_infer --model trained.nlw --input 0.1,0.2 [--stats] [--max-tensor-bytes N]");
        }
        nlmobile::check(!path.empty() && !data.empty(), "--model and --input required");
        auto model=nlmobile::Model::load(path,budget);
        auto prediction=model.predict(nlmobile::parse_csv(data));
        std::cout<<std::setprecision(17)<<"PREDICT values=";
        for(std::size_t i=0;i<prediction.size();++i)std::cout<<(i?",":"")<<prediction[i];
        std::cout<<"\n";
        if(stats)std::cout<<"PROFILE weights="<<model.weights
                <<" tensor_payload_bytes="<<model.payload_bytes
                <<" activation_payload_bytes="<<model.activation_bytes
                <<" estimated_peak_payload_bytes="<<model.payload_bytes+model.activation_bytes
                <<" note=payload_not_process_RSS\n";
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"NEURAL MOBILE ERROR: "<<e.what()<<"\n";
        return 2;
    }
}
#endif
