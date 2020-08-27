#include <iostream>
#include <torch/torch.h>

#include <FortranLibrary.hpp>
#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
#include "Hd.hpp"

argparse::ArgumentParser parse_args(const int & argc, const char ** & argv);

std::vector<std::string> verify_data_set(const std::vector<std::string> & original_data_set);

namespace pretrain {
    void pretrain(const size_t & irred, const size_t & max_depth, const size_t & freeze,
    const std::vector<std::string> & data_set,
    const std::vector<std::string> & chk, const size_t & chk_depth,
    const std::string & opt = "TR", const size_t & epoch = 1000,
    const size_t & batch_size = 32, const double & learning_rate = 0.001);
} // namespace pretrain

namespace train {
    void train(const std::string & Hd_in, const size_t & max_depth, const size_t & freeze,
    const std::vector<std::string> & data_set, const double & zero_point, const double & weight,
    const std::vector<std::string> & chk, const size_t & chk_depth, const std::vector<double> & guess_diag,
    const std::string & opt = "TR", const size_t & epoch = 1000,
    const size_t & batch_size = 32, const double & learning_rate = 0.001);
} // namespace train

int main(int argc, const char** argv) {
    // Welcome
    std::cout << "SurfGenTorch: surface generation package based on libtorch\n";
    std::cout << "Yifan Shen 2020\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::ShowTime();
    std::cout << '\n';
    srand((unsigned)time(NULL));
    // Retrieve required command line arguments and initialize
    std::string job = args.retrieve<std::string>("job");
    std::cout << "Job type: " + job << '\n';
    std::string SSAIC_in = args.retrieve<std::string>("SSAIC_in");
    SSAIC::define_SSAIC(SSAIC_in);
    std::vector<std::string> data_set = verify_data_set(args.retrieve<std::vector<std::string>>("data_set"));
    // Retrieve optional command line arguments for network
    size_t max_depth = 0; // 0 means unlimited
    if (args.gotArgument("max_depth")) max_depth = args.retrieve<size_t>("max_depth");
    std::vector<std::string> checkpoint;
    if (args.gotArgument("checkpoint")) checkpoint = args.retrieve<std::vector<std::string>>("checkpoint");
    size_t chk_depth = max_depth;
    if (args.gotArgument("chk_depth")) chk_depth = args.retrieve<size_t>("chk_depth");
    size_t freeze = (chk_depth < max_depth || max_depth == 0) ? chk_depth : 0;
    if (args.gotArgument("freeze")) freeze = args.retrieve<size_t>("freeze");
    // for optimization
    std::string optimizer = "TR";
    if (args.gotArgument("optimizer")) optimizer = args.retrieve<std::string>("optimizer");
    size_t epoch = 1000;
    if (args.gotArgument("epoch")) epoch = args.retrieve<size_t>("epoch");
    size_t batch_size = 32;
    if (args.gotArgument("batch_size")) batch_size = args.retrieve<size_t>("batch_size");
    double learning_rate = 0.001;
    if (args.gotArgument("learning_rate")) learning_rate = args.retrieve<double>("learning_rate");

    std::cout << '\n';
    if (job == "pretrain") {
        // Retrieve command line arguments
        assert(("Irreducible is required for training", args.gotArgument("irreducible")));
        size_t irred = args.retrieve<size_t>("irreducible") - 1;
        assert(("irreducible out of range", irred < SSAIC::NIrred));
        // Run
        pretrain::pretrain(irred, max_depth, freeze,
            data_set,
            checkpoint, chk_depth,
            optimizer, epoch, batch_size, learning_rate);
    }
    else if (job == "train") {
        // Retrieve command line arguments and initialize
        assert(("DimRed.in is required for training", args.gotArgument("DimRed_in")));
        std::string DimRed_in = args.retrieve<std::string>("DimRed_in");
        DimRed::define_DimRed(DimRed_in);
        assert(("Hd.in is required for training", args.gotArgument("Hd_in")));
        std::string Hd_in = args.retrieve<std::string>("Hd_in");
        double zero_point = 0.0;
        if (args.gotArgument("zero_point")) zero_point = args.retrieve<double>("zero_point");
        double weight = 1.0;
        if (args.gotArgument("weight")) weight = args.retrieve<double>("weight");
        std::vector<double> guess_diag;
        if (args.gotArgument("guess_diag")) guess_diag = args.retrieve<std::vector<double>>("guess_diag");
        // Run
        train::train(Hd_in, max_depth, freeze,
            data_set, zero_point, weight,
            checkpoint, chk_depth, guess_diag,
            optimizer, epoch, batch_size, learning_rate);
    }

    std::cout << '\n';
    CL::utility::ShowTime();
    std::cout << "Mission success\n";
    return 0;
}

argparse::ArgumentParser parse_args(const int & argc, const char ** & argv) {
    CL::utility::EchoCommand(argc, argv); std::cout << '\n';
    argparse::ArgumentParser parser("Surface generation package based on libtorch");

    // Required arguments
    parser.add_argument("--job", 1, false, "pretrain, train");
    parser.add_argument("--SSAIC_in", 1, false, "an input file to define scaled and symmetry adapted internal coordinate");
    parser.add_argument("--data_set", '+', false, "data set list file or directory");

    // Optional arguments for network
    parser.add_argument("--max_depth", 1, true, "max depth of the training network, default = unlimited");
    parser.add_argument("-c","--checkpoint", '+', true, "checkpoint to continue from");
    parser.add_argument("--chk_depth", 1, true, "max depth of the trained network, default = max_depth");
    parser.add_argument("-f","--freeze", 1, true, "freeze leading pretraining layers, default = chk_depth < max_depth ? chk_depth : 0");
    // for optimization
    parser.add_argument("-o","--optimizer", 1, true, "Adam, SGD, CG, TR (default = TR)");
    parser.add_argument("-e","--epoch", 1, true, "default = 1000");
    parser.add_argument("-b","--batch_size", 1, true, "batch size for Adam & SGD (default = 32)");
    parser.add_argument("-l","--learning_rate", 1, true, "learning rate for Adam & SGD (default = 0.001)");

    // pretrain only
    parser.add_argument("-i","--irreducible", 1, true, "the irreducible to pretrain");    

    // train only
    parser.add_argument("--DimRed_in", 1, true, "an input file to define dimensionality reduction network");
    parser.add_argument("--Hd_in", 1, true, "an input file to define diabatic Hamiltonian (Hd)");
    parser.add_argument("-z","--zero_point", 1, true, "zero of potential energy, default = 0");
    parser.add_argument("-w","--weight", 1, true, "Ethresh in weight adjustment, default = 1");
    parser.add_argument("-g","--guess_diag", '+', true, "initial guess of Hd diagonal, default = 0");

    parser.parse_args(argc, argv);
    return parser;
}