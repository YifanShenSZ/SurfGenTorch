#include <iostream>
#include <torch/torch.h>

namespace basic {
    std::string initialize(int argc, const char** argv);
} // namespace basic

int main(int argc, const char** argv) {
    using namespace basic;
    std::string job = initialize(argc, argv);

    std::cout << std::endl;
    if (job == "min") {

    }
    else if (job == "sad") {

    }
    else if (job == "mex") {

    }
}