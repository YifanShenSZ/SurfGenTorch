/*
SurfGenTorch: surface generation package based on libtorch

Yifan Shen 2020
*/

#include <iostream>
#include <torch/torch.h>

#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
#include "observable_net.hpp"
#include "Hd.hpp"

argparse::ArgumentParser parse_args(const int & argc, const char ** & argv) {
    CL::utility::EchoCommand(argc, argv); std::cout << '\n';
    argparse::ArgumentParser parser("Surface generation package based on libtorch");

    // Required arguments
    parser.add_argument("--job", 1, false, "DimRed, Hd");
    parser.add_argument("--SSAIC_in", 1, false, "an input file to define scaled and symmetry adapted internal coordinate");
    parser.add_argument("--DimRed_in", 1, false, "an input file to define dimensionality reduction network");
    parser.add_argument("--data_set", '+', false, "data set list file or directory");

    // Optional arguments for network
    parser.add_argument("-c","--checkpoint", '+', true, "checkpoint to continue from");
    parser.add_argument("-f","--freeze", 1, true, "freeze how many leading training layers (default = 0)");
    // for optimization
    parser.add_argument("-o","--optimizer", 1, true, "Adam, SGD, SD, CG, TR (default = TR)");
    parser.add_argument("-e","--epoch", 1, true, "default = 1000");
    parser.add_argument("-b","--batch_size", 1, true, "batch size for Adam & SGD (default = 32)");
    parser.add_argument("-l","--learning_rate", 1, true, "learning rate for Adam & SGD (default = 0.001)");
    parser.add_argument("-g","--GPU", 0, true, "use GPU for Adam & SGD");

    // for training dimensionality reduction
    parser.add_argument("-i","--irreducible", 1, true, "(job == DimRed) the irreducible to train dimensionality reduction");

    // for training further quantities based on an established dimensionality reduction
    parser.add_argument("--input_layer_in",    1, true, "(job != DimRed) an input file to define the symmetry adapted polynomials");
    parser.add_argument("-t","--train_DimRed", 0, true, "(job != DimRed) simultaneously train the dimensionality reduction network");

    // for training diabatic Hamiltonian
    parser.add_argument("--Hd_in",        1, true, "(job == Hd) an input file to define diabatic Hamiltonian (Hd)");
    parser.add_argument("--zero_point",   1, true, "(job == Hd) zero of potential energy, default = 0");
    parser.add_argument("--weight",       1, true, "(job == Hd) Ethresh in weight adjustment, default = 1");
    parser.add_argument("--guess_diag", '+', true, "(job == Hd) initial guess of Hd diagonal, default = pytorch initialization");

    parser.parse_args(argc, argv);
    return parser;
}

// Check if user inputs are directories (end with /)
// otherwise consider as lists, then read the lists for directories
std::vector<std::string> verify_data_set(const std::vector<std::string> & original_data_set) {
    std::vector<std::string> data_set;
    for (std::string item : original_data_set) {
        if (item[item.size()-1] == '/') data_set.push_back(item);
        else {
            std::string prefix = CL::utility::GetPrefix(item);
            std::string directory;
            std::ifstream ifs; ifs.open(item);
                ifs >> directory;
                while (ifs.good()) {
                    if (directory[directory.size()-1] != '/') directory = directory + "/";
                    directory = prefix + directory;
                    data_set.push_back(directory);
                    ifs >> directory;
                }
            ifs.close();
        }
    }
    // output to job log
    std::cout << "The training set will be read from: \n    ";
    size_t line_length = 4;
    for (size_t i = 0; i < data_set.size()-1; i++) {
        line_length += data_set[i].size() + 2;
        if (line_length > 75) {
            std::cout << '\n' << "    ";
            line_length = 4;
        }
        std::cout << data_set[i] << ", ";
    }
    line_length += data_set[data_set.size()-1].size() + 2;
    if (line_length > 75) std::cout << '\n' << "    ";
    std::cout << data_set[data_set.size()-1] << '\n';
    return data_set;
}

namespace train_DimRed {
void train(const size_t & irred, const size_t & freeze, const std::vector<std::string> & chk,
const std::vector<std::string> & data_set,
const std::string & opt = "TR", const size_t & epoch = 1000,
const size_t & batch_size = 32, const double & learning_rate = 0.001, const bool & GPU = true);
} // namespace train_DimRed

namespace train_Hd {
void train(const std::vector<double> & guess_diag, const size_t & freeze, const std::vector<std::string> & chk,
const std::vector<std::string> & data_set, const bool & train_DimRed,
const double & zero_point, const double & weight,
const std::string & opt = "TR", const size_t & epoch = 1000,
const size_t & batch_size = 32, const double & learning_rate = 0.001, const bool & GPU = true);
} // namespace train_Hd

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
    std::string DimRed_in = args.retrieve<std::string>("DimRed_in");
    std::vector<std::string> data_set = verify_data_set(args.retrieve<std::vector<std::string>>("data_set"));
    // Retrieve optional command line arguments for network
    std::vector<std::string> checkpoint;
    if (args.gotArgument("checkpoint")) checkpoint = args.retrieve<std::vector<std::string>>("checkpoint");
    size_t freeze = 0;
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
    if (job == "DimRed") {
        DimRed::define_DimRed_train(DimRed_in);
        // Retrieve command line arguments and initialize
        assert(("Irreducible is required for training dimensionality reduction", args.gotArgument("irreducible")));
        size_t irred = args.retrieve<size_t>("irreducible") - 1;
        assert(("irreducible out of range", irred < SSAIC::NIrred));
        // Run
        train_DimRed::train(irred, freeze, checkpoint, data_set,
            optimizer, epoch, batch_size, learning_rate, args.gotArgument("GPU"));
    }
    else {
        if (args.gotArgument("train_DimRed")) {
            std::cout << "The dimensionality reduction network will be trained simultaneously\n";
            DimRed::define_DimRed_train(DimRed_in);
        }
        else DimRed::define_DimRed(DimRed_in);
        // Retrieve command line arguments and initialize
        assert(("input_layer.in is required for training further quantities based on an established dimensionality reduction", args.gotArgument("input_layer_in")));
        std::string input_layer_in = args.retrieve<std::string>("input_layer_in");
        ON::define_PNR(input_layer_in);
        if (job == "Hd") {
            // Retrieve command line arguments and initialize
            assert(("Hd.in is required for training Hd", args.gotArgument("Hd_in")));
            std::string Hd_in = args.retrieve<std::string>("Hd_in");
            Hd::define_Hd_train(Hd_in);
            double zero_point = 0.0;
            if (args.gotArgument("zero_point")) zero_point = args.retrieve<double>("zero_point");
            double weight = 1.0;
            if (args.gotArgument("weight")) weight = args.retrieve<double>("weight");
            std::vector<double> guess_diag;
            if (args.gotArgument("guess_diag")) guess_diag = args.retrieve<std::vector<double>>("guess_diag");
            // Run
            train_Hd::train(guess_diag, freeze, checkpoint,
                data_set, args.gotArgument("train_DimRed"),
                zero_point, weight,
                optimizer, epoch, batch_size, learning_rate, args.gotArgument("GPU"));
        }
    }

    std::cout << '\n';
    CL::utility::ShowTime();
    std::cout << "Mission success\n";
    return 0;
}