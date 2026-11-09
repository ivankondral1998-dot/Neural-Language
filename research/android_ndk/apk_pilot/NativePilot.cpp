// Research-only Android JNI harness for ORIGINAL compiler-emitted .nir C++.
#include <jni.h>
#include <unistd.h>
#include <fcntl.h>
#include <mutex>
#include <sstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <cerrno>
#include <cstring>

#define main neural_pilot_cartpole_nir
#include "generated/pretrained_cartpole.cpp"
#undef main
#define main neural_pilot_train_nir
#include "generated/integrated.cpp"
#undef main
#define main neural_pilot_inference_nir
#include "generated/inference.cpp"
#undef main
#define main neural_pilot_agent_runner
#include "../../agent_pilot/agent_cartpole.cpp"
#undef main

namespace {
std::mutex runner_mutex;
std::string invoke(const std::string& directory,int kind,int episodes) {
    std::lock_guard<std::mutex> lock(runner_mutex);
    int original_cwd=open(".",O_RDONLY|O_DIRECTORY);
    if(original_cwd<0)throw std::runtime_error("Cannot save working directory");
    std::ostringstream result, errors;
    auto *old_stdout=std::cout.rdbuf(result.rdbuf());
    auto *old_stderr=std::cerr.rdbuf(errors.rdbuf());
    int status=-1;
    try {
        if(chdir(directory.c_str())!=0)throw std::runtime_error("Cannot access private run directory");
        if(kind==1){
            if(episodes!=10&&episodes!=100)throw std::runtime_error("Invalid episode count");
            std::string n=std::to_string(episodes);
            char arg0[]="neural_agent",arg1[]="cartpole_pretrained.nlw";
            char* argv[]={arg0,arg1,&n[0]};
            status=neural_pilot_agent_runner(3,argv);
        } else if(kind==2) {
            int train=neural_pilot_train_nir();
            status=train==0?neural_pilot_inference_nir():train;
        } else if(kind==3) {
            status=neural_pilot_cartpole_nir();
        } else throw std::runtime_error("Unknown test kind");
    } catch(const std::exception& e) {
        errors<<"JNI ERROR "<<e.what()<<"\n";
        status=-1;
    }
    std::cout.flush();
    std::cerr.flush();
    std::cout.rdbuf(old_stdout);
    std::cerr.rdbuf(old_stderr);
    const int restore=fchdir(original_cwd);
    close(original_cwd);
    if(restore!=0)throw std::runtime_error("Failed to restore working directory");
    std::string combined=result.str();
    if(!errors.str().empty())combined+="\nERROR:\n"+errors.str();
    if(kind==1 && status==0 && combined.find("PILOT_SUCCESS episodes="+std::to_string(episodes))==std::string::npos)
        status=-1;
    if(kind==2 && status==0 &&
       (combined.find("MODEL_SAVED trained.nlw")==std::string::npos ||
        combined.find("MODEL_LOADED trained.nlw")==std::string::npos ||
        combined.find("PREDICT version=1 values=0.3")==std::string::npos))
        status=-1;
    return std::string(status==0?"PASS\n":"FAIL\n")+combined;
}
}
extern "C" JNIEXPORT jstring JNICALL
Java_dev_neurallanguage_pilot_MainActivity_nativeRun(JNIEnv* env,jclass,jstring jdir,jint kind,jint episodes){
    if(!jdir)return env->NewStringUTF("FAIL: null directory");
    const char* utf=env->GetStringUTFChars(jdir,nullptr);
    if(!utf)return nullptr;
    std::string directory(utf);
    env->ReleaseStringUTFChars(jdir,utf);
    try{
        const auto output=invoke(directory,kind,episodes);
        return env->NewStringUTF(output.c_str());
    }catch(const std::exception& e){
        std::string s=std::string("FAIL: ")+e.what();
        return env->NewStringUTF(s.c_str());
    }
}