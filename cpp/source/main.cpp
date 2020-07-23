#include <iostream>
#include <omp.h>
#include <torch/torch.h>
#include <FortranLibrary.hpp>
#include "../Cpp-Library_v1.0.0/argparse.hpp"
#include "../Cpp-Library_v1.0.0/general.hpp"
#include "../include/net.hpp"

argparse::ArgumentParser parse_args(const int & argc, const char ** & argv);
std::tuple<int, at::Tensor> cartdim_origin(const std::string & format, const std::string & origin_file, const int & indim);
std::vector<std::string> verify_data_set(const std::vector<std::string> & original_data_set);

int main(int argc, const char** argv) {
    // welcome
    std::cout << "SurfGenTorch: surface generation package based on libtorch\n";
    std::cout << "Yifan Shen 2020\n\n";
    general::ShowTime(); std::cout << '\n';
    // global initialization
    srand((unsigned)time(NULL));

    // command line input
    argparse::ArgumentParser args = parse_args(argc, argv);

    std::string job = args.retrieve<std::string>("job");
    std::cout << "Job type: " + job << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::cout << "File format: " + format + "\n";

    std::string IntCoordDef = args.retrieve<std::string>("IntCoordDef");
    int intdim = FL::GeometryTransformation::DefineInternalCoordinate(format, IntCoordDef);
    std::cout << "Internal coordinate space dimension = " << intdim << '\n';

    int cartdim; at::Tensor origin;
    std::tie(cartdim, origin) = cartdim_origin(format, args.retrieve<std::string>("origin"), intdim);

if (job == "pretrain") {
    std::vector<size_t> symmetry;
    if (args.gotArgument("symmetry")) {
        symmetry = args.retrieve<std::vector<size_t>>("symmetry");
        std::cout << "Number of irreducible representations = " << symmetry.size() << '\n';
        if (std::accumulate(symmetry.begin(), symmetry.end(), 0) != intdim)
        throw std::invalid_argument("Every dimension must be assigned to an irreducible");
    } else {
        symmetry.resize(1);
        symmetry[0] = intdim;
        std::cout << "No symmetry\n";
    }

    std::vector<std::string> data_set = verify_data_set(args.retrieve<std::vector<std::string>>("data_set"));

    std::string data_type = "double";
    if (args.gotArgument("data_type")) data_type = args.retrieve<std::string>("data_type");
    std::cout << "Data type in use: " << data_type << '\n';

    DimRed::pretrain(origin, intdim, symmetry, data_set, data_type);
}

    std::cout << '\n';
    general::ShowTime();
    std::cout << "Mission success\n";

    return 0;
}

argparse::ArgumentParser parse_args(const int & argc, const char ** & argv) {
    general::EchoCommand(argc, argv); std::cout << '\n';
    argparse::ArgumentParser parser("Surface generation package based on libtorch");
    // required argument
    parser.add_argument("-j", "--job", 1, false, "job type: pretrain, train");
    parser.add_argument("-f", "--format", 1, false, "file format: Columbus7 or default");
    parser.add_argument("-i", "--IntCoordDef", 1, false, "internal coordinate definition file");
    parser.add_argument("-o", "--origin", 1, false, "internal coordinate space origin file");

    parser.add_argument("-c", "--check_point", 1, true, "check point to continue with");

    parser.add_argument("-r", "--restart", 0, true, "simply restart previous training");

    // pretrain
    parser.add_argument("-s", "--symmetry", '+', true, "symmetry (internal dimension per irreducible representation)");
    parser.add_argument("-d", "--data_set", '+', true, "data set list file or directory");
    parser.add_argument("-t", "--data_type", 1, true, "data type: float, double, default = double");
    
    // train : pretrain
    parser.add_argument("-z", "--zero_point", 1, true, "zero of potential energy, default = 0");
    parser.add_argument("-n", "--NStates", 1, true, "number of electronic states");
    parser.add_argument("-v", "--valid_states", '+', true, "states 1 to valid_states[i] are valid in data_set[i]");

    parser.parse_args(argc, argv);
    return parser;
}