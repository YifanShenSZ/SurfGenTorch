#include <iostream>
#include <torch/torch.h>

namespace basic {
    std::tuple<std::string, bool, std::string> initialize(int argc, const char** argv);
} // namespace basic

int main(int argc, const char** argv) {
    std::string job;
    bool diabatic;
    std::string opt;
    std::tie(job, diabatic, opt) = basic::initialize(argc, argv);

    std::cout << std::endl;
    if (job == "min") {

    }
    else if (job == "sad") {

    }
    else if (job == "mex") {

    }
}