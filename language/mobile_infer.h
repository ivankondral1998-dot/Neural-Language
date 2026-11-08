/* Neural Language native inference ABI (NLMODEL2, sequential dense). */
#ifndef NEURAL_LANGUAGE_MOBILE_INFER_H
#define NEURAL_LANGUAGE_MOBILE_INFER_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Returns NULL on failure. Caller must release handle with nlmobile_close(). */
void* nlmobile_open(const char* checkpoint_path, size_t byte_budget);
size_t nlmobile_input_count(const void* handle);
size_t nlmobile_output_count(const void* handle);
/* Numeric payload estimate, not process RSS. */
size_t nlmobile_tensor_payload_bytes(const void* handle);
/* Returns output element count, or -1 on error. */
int nlmobile_predict(void* handle, const double* input, size_t n_input,
                     double* output, size_t output_capacity);
/* Thread-local error text; valid until next call in same thread. */
const char* nlmobile_last_error(void);
void nlmobile_close(void* handle);
#ifdef __cplusplus
}
#endif
#endif
