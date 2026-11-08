// NLMOBIN1: explicitly little-endian, CRC32-protected compact inference checkpoint.
// Included inside nlmobile namespace after Model is defined. No external dependencies.
#pragma once
namespace binary_detail {
inline uint32_t u32(const unsigned char* p) {
    return uint32_t(p[0]) | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24;
}
inline uint64_t u64(const unsigned char* p) {
    return uint64_t(u32(p)) | uint64_t(u32(p+4))<<32;
}
inline double number(const unsigned char* p) {
    static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,
                  "NLMOBIN1 requires IEEE-754 binary64");
    const uint64_t bits=u64(p);
    double v=0;
    std::memcpy(&v,&bits,8);
    check(valid(v),"binary checkpoint invalid number");
    return v;
}
struct Reader {
    std::ifstream stream;
    uint64_t remaining;
    uint32_t crc=0xffffffffu;
    explicit Reader(const std::string& path, uint64_t bytes) : stream(path,std::ios::binary),remaining(bytes) {
        check(bool(stream),"cannot open binary checkpoint");
        stream.seekg(40,std::ios::beg);
        check(bool(stream),"binary checkpoint seek failed");
    }
    void read(unsigned char* out,std::size_t count) {
        check(uint64_t(count)<=remaining,"binary checkpoint truncated payload");
        check(count<=static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()),
              "binary read exceeds stream limit");
        stream.read(reinterpret_cast<char*>(out),static_cast<std::streamsize>(count));
        check(bool(stream),"binary checkpoint truncated payload");
        remaining-=count;
        for(std::size_t i=0;i<count;++i){
            crc^=out[i];
            for(int j=0;j<8;++j)crc=(crc>>1)^((crc&1u)?0xedb88320u:0u);
        }
    }
    uint32_t uint32() {unsigned char b[4];read(b,4);return u32(b);}
    double fp64() {unsigned char b[8];read(b,8);return number(b);}
    unsigned char byte() {unsigned char b=0;read(&b,1);return b;}
};
} // namespace binary_detail

inline Model load_binary_checkpoint(const std::string& path,std::size_t max_bytes){
    std::ifstream stream(path,std::ios::binary|std::ios::ate);
    check(bool(stream),"cannot open binary checkpoint");
    const auto length=stream.tellg();
    check(length>=std::streamoff(40),"binary checkpoint header truncated");
    const uint64_t file_size=static_cast<uint64_t>(length);
    stream.seekg(0,std::ios::beg);
    std::array<unsigned char,40> header{};
    stream.read(reinterpret_cast<char*>(header.data()),40);
    check(bool(stream),"binary checkpoint header truncated");
    check(std::memcmp(header.data(),"NLMOBIN1",8)==0,"binary checkpoint magic invalid");
    const uint32_t version=binary_detail::u32(header.data()+8);
    const uint32_t inputs=binary_detail::u32(header.data()+12);
    const uint32_t layers=binary_detail::u32(header.data()+16);
    const uint32_t nweights=binary_detail::u32(header.data()+20);
    const uint64_t payload=binary_detail::u64(header.data()+24);
    const uint32_t expected_crc=binary_detail::u32(header.data()+32);
    const uint32_t reserved=binary_detail::u32(header.data()+36);
    check(version==1 && reserved==0,"unsupported binary checkpoint version");
    check(inputs>=1 && inputs<=2048 && layers>=1 && layers<=8 && nweights>=1 && nweights<=4000000u,
          "binary checkpoint dimensions invalid");
    check(payload==file_size-40,"binary checkpoint length or trailing bytes invalid");
    // An in-memory tensor representation is always at least as large as the serialized payload.
    check(payload<=static_cast<uint64_t>(max_bytes),"binary checkpoint exceeds byte budget");
    Model model;
    model.input_size=static_cast<int>(inputs);
    model.layers.reserve(layers);
    binary_detail::Reader rd(path,payload);
    int previous=static_cast<int>(inputs);
    std::size_t current=inputs,max_pair=inputs;
    for(uint32_t k=0;k<layers;++k){
        Layer layer;
        const uint32_t in=rd.uint32(),out=rd.uint32(),act=rd.uint32(),storage=rd.uint32();
        check(in==static_cast<uint32_t>(previous) && out>=1 && out<=4096 && act<=2 && storage<=1,
              "binary checkpoint layer mismatch");
        layer.in=static_cast<int>(in);
        layer.out=static_cast<int>(out);
        layer.activation=static_cast<int>(act);
        layer.storage=static_cast<int>(storage);
        const std::size_t n=static_cast<std::size_t>(in)*out;
        check(n<=4000000u && model.weights+n<=4000000u,"binary checkpoint weight limit exceeded");
        const std::size_t bytes=n*(storage?1u:8u)+n+8u*out;
        check(model.payload_bytes<=max_bytes && bytes<=max_bytes-model.payload_bytes,
              "binary checkpoint tensor budget exceeded");
        model.weights+=n;
        model.payload_bytes+=bytes;
        layer.bias.resize(out);
        for(double& v:layer.bias)v=rd.fp64();
        if(storage){
            layer.compact.resize(n);
            for(auto& w:layer.compact){
                const unsigned char b=rd.byte();
                const int signed_code=b>=128?int(b)-256:int(b);
                check(signed_code>=-127,"binary checkpoint INT8 code invalid");
                w=static_cast<int8_t>(signed_code);
            }
        } else {
            layer.weights.resize(n);
            for(double& v:layer.weights)v=rd.fp64();
        }
        layer.gates.resize(n);
        const std::size_t packed=(n+7u)/8u;
        for(std::size_t j=0;j<packed;++j){
            const unsigned char group=rd.byte();
            for(unsigned bit=0;bit<8;++bit){
                const std::size_t ix=j*8u+bit;
                if(ix<n)layer.gates[ix]=static_cast<uint8_t>((group>>bit)&1u);
                else check(((group>>bit)&1u)==0,"binary checkpoint mask padding invalid");
            }
        }
        max_pair=std::max(max_pair,current+static_cast<std::size_t>(out));
        current=out;
        previous=static_cast<int>(out);
        model.layers.push_back(std::move(layer));
    }
    check(rd.remaining==0,"binary checkpoint unexpected payload");
    check((rd.crc^0xffffffffu)==expected_crc,"binary checkpoint CRC32 mismatch");
    check(model.weights==nweights,"binary checkpoint total weights mismatch");
    model.activation_bytes=8u*max_pair;
    check(model.payload_bytes<=max_bytes && model.activation_bytes<=max_bytes-model.payload_bytes,
          "binary checkpoint activation budget exceeded");
    return model;
}
