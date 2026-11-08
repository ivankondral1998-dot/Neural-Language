#include "../mobile_infer.h"
int main(int argc, char** argv) {
    if (argc != 2) return 1;
    void *model = nlmobile_open(argv[1], 4096);
    if (!model) return 2;
    const double input[2] = {0.25,0.75};
    double output[1] = {0};
    int rc = nlmobile_predict(model, input, 2, output, 1);
    nlmobile_close(model);
    return (rc == 1 && output[0] > 0.29999 && output[0] < 0.30001) ? 0 : 3;
}
