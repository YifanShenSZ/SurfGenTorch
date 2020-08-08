#include <iostream>
#include <torch/torch.h>
#include <FortranLibrary.hpp>
#include "../Cpp-Library_v1.0.0/argparse.hpp"
#include "../Cpp-Library_v1.0.0/utility.hpp"
#include "../include/SSAIC.hpp"
#include "../include/pretrain.hpp"

argparse::ArgumentParser parse_args(const int & argc, const char ** & argv);
std::vector<std::string> verify_data_set(const std::vector<std::string> & original_data_set);

int main(int argc, const char** argv) {
    // welcome
    std::cout << "SurfGenTorch: surface generation package based on libtorch\n";
    std::cout << "Yifan Shen 2020\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::ShowTime();
    std::cout << '\n';
    srand((unsigned)time(NULL));
    // retrieve command line arguments and initialize
    std::string job = args.retrieve<std::string>("job");
    std::cout << "Job type: " + job << '\n';
    std::string format = args.retrieve<std::string>("format");
    std::cout << "File format: " + format + "\n";
    std::string IntCoordDef = args.retrieve<std::string>("IntCoordDef");
    std::string origin = args.retrieve<std::string>("origin");
    std::string scale_symmetry = args.retrieve<std::string>("scale_symmetry");
    SSAIC::define_SSAIC(format, IntCoordDef, origin, scale_symmetry);
    std::vector<std::string> data_set = verify_data_set(args.retrieve<std::vector<std::string>>("data_set"));
    std::vector<std::string> checkpoint;
    if (args.gotArgument("checkpoint")) checkpoint = args.retrieve<std::vector<std::string>>("checkpoint");
    std::string optimizer = "TR";
    if (args.gotArgument("optimizer")) optimizer = args.retrieve<std::string>("optimizer");
    size_t epoch = 1000;
    if (args.gotArgument("epoch")) epoch = args.retrieve<size_t>("epoch");

    std::cout << '\n';
    if (job == "pretrain") {
        // retrieve command line arguments
        size_t irred = args.retrieve<size_t>("irreducible");
        size_t max_depth = 0;
        if (args.gotArgument("max_depth")) max_depth = args.retrieve<size_t>("max_depth");
        size_t chk_depth = max_depth;
        if (args.gotArgument("chk_depth")) chk_depth = args.retrieve<size_t>("chk_depth");
        size_t freeze = chk_depth;
        if (args.gotArgument("freeze")) freeze = args.retrieve<size_t>("freeze");
        // run
        DimRed::pretrain(irred, max_depth, data_set,
            checkpoint, chk_depth, freeze,
            optimizer, epoch);
    }

    std::cout << '\n';
    CL::utility::ShowTime();
    std::cout << "Mission success\n";
    return 0;
}

argparse::ArgumentParser parse_args(const int & argc, const char ** & argv) {
    CL::utility::EchoCommand(argc, argv); std::cout << '\n';
    argparse::ArgumentParser parser("Surface generation package based on libtorch");
    
    // required arguments
    parser.add_argument("--job", 1, false, "pretrain, train");
    parser.add_argument("--format", 1, false, "Columbus7, default");
    parser.add_argument("--IntCoordDef", 1, false, "internal coordinate definition file");
    parser.add_argument("--origin", 1, false, "internal coordinate origin file");
    parser.add_argument("--scale_symmetry", 1, false, "scale and symmetry definition file");
    parser.add_argument("--data_set", '+', false, "data set list file or directory");
    
    // optional arguments
    parser.add_argument("-c","--checkpoint", '+', true, "checkpoint to continue from");
    parser.add_argument("-o","--optimizer", 1, true, "Adam, CG, TR (default = TR)");
    parser.add_argument("-e","--epoch", 1, true, "default = 1000");
    
    // pretrain only
    parser.add_argument("-i","--irreducible", 1, true, "the irreducible to pretrain");
    parser.add_argument("--max_depth", 1, true, "max depth of the pretraining network");
    parser.add_argument("--chk_depth", 1, true, "max depth of the checkpoint network");
    parser.add_argument("-f","--freeze", 1, true, "freeze leading layers");
    
    // train
    /// parser.add_argument("-z", "--zero_point", 1, true, "zero of potential energy, default = 0");
    /// parser.add_argument("-n", "--NStates", 1, true, "number of electronic states");
    /// parser.add_argument("-v", "--valid_states", '+', true, "states 1 to valid_states[i] are valid in data_set[i]");

    parser.parse_args(argc, argv);
    return parser;
}