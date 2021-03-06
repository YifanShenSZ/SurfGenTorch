/*
Data view: view ab initio data details used in SurfGenTorch

Yifan Shen 2020
*/

#include <iostream>
#include <torch/torch.h>

#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
#include "observable_net.hpp"
#include "AbInitio.hpp"

argparse::ArgumentParser parse_args(const int & argc, const char ** & argv) {
    CL::utility::EchoCommand(argc, argv); std::cout << '\n';
    argparse::ArgumentParser parser("Surface generation package based on libtorch");

    // Required arguments
    parser.add_argument("--job", 1, false, "DimRed, Hd");
    parser.add_argument("--SSAIC_in", 1, false, "an input file to define scaled and symmetry adapted internal coordinate");
    parser.add_argument("--data_set", '+', false, "data set list file or directory");

    // for training dimensionality reduction
    parser.add_argument("-i","--irreducible", 1, true, "the irreducible to train dimensionality reduction");

    // for training further quantities based on an established dimensionality reduction
    parser.add_argument("--DimRed_in", 1, true, "an input file to define dimensionality reduction network");
    parser.add_argument("--input_layer_in", 1, true, "an input file to define the symmetry adapted polynomials");

    // for training diabatic Hamiltonian
    parser.add_argument("-z","--zero_point", 1, true, "zero of potential energy, default = 0");
    parser.add_argument("-w","--weight", 1, true, "Ethresh in weight adjustment, default = 1");

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

int main(int argc, const char** argv) {
    // Welcome
    std::cout << "Data view: view ab initio data details used in SurfGenTorch\n";
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

    std::cout << '\n';
    if (job == "DimRed") {
        // Retrieve command line arguments
        assert(("Irreducible is required for training", args.gotArgument("irreducible")));
        size_t irred = args.retrieve<size_t>("irreducible") - 1;
        assert(("irreducible out of range", irred < SSAIC::NIrred));
        // Run
        auto * GeomSet = AbInitio::read_GeomSet(data_set);
        std::cout << "Number of geometries = " << GeomSet->size_int() << '\n';
        std::cout << "The scaled and symmetry adapted geometries of irreducible " << irred << " are:\n";
        size_t count = 0;
        for (auto & data : GeomSet->example()) {
            std::cout << "Data " << count << ":\n"
                      << data->SAIgeom[irred] << '\n';
            count++;
        }
    }
    else {
        // Initialize DimRed and ON
        assert(("DimRed.in is required for training further quantities based on an established dimensionality reduction", args.gotArgument("DimRed_in")));
        std::string DimRed_in = args.retrieve<std::string>("DimRed_in");
        DimRed::define_DimRed(DimRed_in);
        assert(("input_layer.in is required for training further quantities based on an established dimensionality reduction", args.gotArgument("input_layer_in")));
        std::string input_layer_in = args.retrieve<std::string>("input_layer_in");
        ON::define_PNR(input_layer_in);
        if (job == "Hd") {
            // Retrieve command line arguments and initialize
            double zero_point = 0.0;
            if (args.gotArgument("zero_point")) zero_point = args.retrieve<double>("zero_point");
            double weight = 1.0;
            if (args.gotArgument("weight")) weight = args.retrieve<double>("weight");
            // Run
            // Read data set
            AbInitio::DataSet<AbInitio::RegHam> * RegSet;
            AbInitio::DataSet<AbInitio::DegHam> * DegSet;
            std::tie(RegSet, DegSet) = AbInitio::read_HamSet(data_set, zero_point, weight);
            std::cout << "Number of regular data = " << RegSet->size_int() << '\n';
            size_t count = 0;
            for (auto & data : RegSet->example()) {
                std::cout << "Data " << count << ":\n";
                std::cout << "Input layer:\n";
                for (size_t i = 0; i < data->input_layer.size(); i++) {
                    std::cout << "Irreducible " << i + 1 << ":\n"
                              << data->input_layer[i] << '\n';
                }
                std::cout << "Eigenvalues of Jacobian . Jacobian^T:\n";
                for (size_t i = 0; i < data->JT.size(); i++) {
                    auto & irred = data->JT[i];
                    at::Tensor JJT = irred.transpose(0, 1).mm(irred);
                    at::Tensor eigval, eigvec;
                    std::tie(eigval, eigvec) = JJT.symeig();
                    std::cout << "Irreducible " << i + 1 << ":\n"
                              << eigval << '\n';
                }
                std::cout << "Energy:\n" << data->energy << '\n';
                count++;
            }
            std::cout << "Number of degenerate data = " << DegSet->size_int() << '\n';
            count = 0;
            for (auto & data : DegSet->example()) {
                std::cout << "Data " << count << ":\n";
                std::cout << "Input layer:\n";
                for (size_t i = 0; i < data->input_layer.size(); i++) {
                    std::cout << "Irreducible " << i + 1 << ":\n"
                              << data->input_layer[i] << '\n';
                }
                std::cout << "Eigenvalues of Jacobian . Jacobian^T:\n";
                for (size_t i = 0; i < data->JT.size(); i++) {
                    auto & irred = data->JT[i];
                    at::Tensor JJT = irred.transpose(0, 1).mm(irred);
                    at::Tensor eigval, eigvec;
                    std::tie(eigval, eigvec) = JJT.symeig();
                    std::cout << "Irreducible " << i + 1 << ":\n"
                              << eigval << '\n';
                }
                std::cout << "Composite representation Hamiltonian:\n" << data->H << '\n';
                count++;
            }
        }
    }

    std::cout << '\n';
    CL::utility::ShowTime();
    std::cout << "Mission success\n";
    return 0;
}