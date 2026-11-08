// Java/Android bridge for Neural Language inference. No training or network access.
#include <jni.h>
#include <cstdint>
#include <string>
#include <vector>
#include <limits>
#include "../mobile_infer.h"

namespace {
void raise(JNIEnv* env,const char* type,const char* message) {
    jclass cls=env->FindClass(type);
    if(cls){env->ThrowNew(cls,message);env->DeleteLocalRef(cls);}
}
void* pointer(jlong value){return reinterpret_cast<void*>(static_cast<std::uintptr_t>(value));}
}

extern "C" {
JNIEXPORT jlong JNICALL Java_com_neurallanguage_mobile_NeuralMobile_nativeOpen
    (JNIEnv* env,jclass,jstring path,jlong limit) {
    if(!path || limit<=0 || static_cast<uint64_t>(limit)>std::numeric_limits<std::size_t>::max()) {
        raise(env,"java/lang/IllegalArgumentException","checkpoint path or limit invalid");return 0;
    }
    const char* value=env->GetStringUTFChars(path,nullptr);
    if(!value)return 0; // pending OutOfMemoryError
    void* handle=nlmobile_open(value,static_cast<std::size_t>(limit));
    env->ReleaseStringUTFChars(path,value);
    if(!handle){raise(env,"java/lang/IllegalArgumentException",nlmobile_last_error());return 0;}
    return static_cast<jlong>(reinterpret_cast<std::uintptr_t>(handle));
}
JNIEXPORT jint JNICALL Java_com_neurallanguage_mobile_NeuralMobile_nativeInputSize
    (JNIEnv*,jclass,jlong handle){return static_cast<jint>(nlmobile_input_count(pointer(handle)));}
JNIEXPORT jint JNICALL Java_com_neurallanguage_mobile_NeuralMobile_nativeOutputSize
    (JNIEnv*,jclass,jlong handle){return static_cast<jint>(nlmobile_output_count(pointer(handle)));}
JNIEXPORT jdoubleArray JNICALL Java_com_neurallanguage_mobile_NeuralMobile_nativePredict
    (JNIEnv* env,jclass,jlong handle,jdoubleArray input) {
    const std::size_t n=nlmobile_input_count(pointer(handle));
    const std::size_t output_size=nlmobile_output_count(pointer(handle));
    if(!input || !n || n!=static_cast<std::size_t>(env->GetArrayLength(input))) {
        raise(env,"java/lang/IllegalArgumentException","input shape or model handle invalid");return nullptr;
    }
    if(!output_size || output_size>4096){
        raise(env,"java/lang/IllegalStateException","model output shape invalid");return nullptr;
    }
    std::vector<double> in(n),out(output_size);
    env->GetDoubleArrayRegion(input,0,static_cast<jsize>(n),in.data());
    if(env->ExceptionCheck())return nullptr;
    int rc=nlmobile_predict(pointer(handle),in.data(),in.size(),out.data(),out.size());
    if(rc<0){raise(env,"java/lang/IllegalArgumentException",nlmobile_last_error());return nullptr;}
    jdoubleArray result=env->NewDoubleArray(rc);
    if(!result)return nullptr;
    env->SetDoubleArrayRegion(result,0,rc,out.data());
    return result;
}
JNIEXPORT void JNICALL Java_com_neurallanguage_mobile_NeuralMobile_nativeClose
    (JNIEnv*,jclass,jlong handle){nlmobile_close(pointer(handle));}
}
