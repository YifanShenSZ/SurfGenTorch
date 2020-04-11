#include <iostream>
#include <torch/torch.h>
#include <FortranLibrary.hpp>
#include "../Cpp-Library_v1.0.0/argparse.hpp"
#include "../Cpp-Library_v1.0.0/general.hpp"
#include "../Cpp-Library_v1.0.0/chemistry.hpp"
#include "../include/AbInitio.hpp"

// Command line input
argparse::ArgumentParser parse_args(const int & argc, const char ** & argv) {
    argparse::ArgumentParser parser("Surface generation package based on libtorch");
    // file format: Columbus7 or default
    parser.add_argument("-f", "--format", 1, false);
    // internal coordinate definition file
    parser.add_argument("-i", "--internal_coordinate_definition", 1, false);
    // job type: fit, continue or analyze
    parser.add_argument("-j", "--job", 1, false);
    // simply restart previous fitting
    parser.add_argument("-r", "--restart");
    // ---------- Parameters to start a new fit ----------
        // number of electronic states, required for a new fit
        parser.add_argument("-n", "--NStates", 1);
        // internal coordinate space origin, required for a new fit
        parser.add_argument("-o", "--origin", 1);
        // zero of potential energy, default = 0
        parser.add_argument("-z", "--zero_point", 1);
        // training set directories, required for a new fit
        parser.add_argument("-t", "--training_set", '+');
        // states 1 to valid_states[i] are valid in training_set[i]
        parser.add_argument("-v", "--valid_states", '+');
        // check point to continue with
        parser.add_argument("-c", "--check_point", 1);
    // ----------------------- End -----------------------
    parser.parse_args(argc, argv);
    return parser;
}

// Program initializer
void initialize(const int & NState) {
    srand((unsigned)time(NULL));
    FL::Chemistry::InitializePhaseFixing(NState);
}

int main(int argc, const char** argv) {
    std::cout << "SurfGenTorch: surface generation package based on libtorch\n";
    std::cout << "Yifan Shen 2020\n\n";
    general::ShowTime();
    argparse::ArgumentParser args = parse_args(argc, argv);

    std::string format = args.retrieve<std::string>("format");
    std::cout << "File format = " + format + "\n";

    std::string IntCoordDef = args.retrieve<std::string>("internal_coordinate_definition");
    int intdim = FL::GeometryTransformation::DefineInternalCoordinate(format, IntCoordDef);
    std::cout << "Internal coordinate space dimension = " << intdim << '\n';
    
    std::string job = args.retrieve<std::string>("job");
    std::cout << "Job type: " + job << '\n';
    if (job == "fit") {
        if (args.gotArgument("restart")) {
            // TO BE IMPLEMENTED
        } else {
            int NStates = args.retrieve<int>("NStates");
            
            int cartdim;
            auto top = at::TensorOptions().dtype(torch::kFloat64);
            at::Tensor origin = at::zeros(intdim, top);
            if (format == "Columbus7") {
                chemistry::xyz_mass<double> molorigin(args.retrieve<std::string>("origin"), true);
                cartdim = 3 * molorigin.NAtoms();
                FL::GeometryTransformation::InternalCoordinate(molorigin.geom().data(), origin.data_ptr<double>(), cartdim, intdim);
            } else {
                chemistry::xyz<double> molorigin(args.retrieve<std::string>("origin"), true);
                cartdim = 3 * molorigin.NAtoms();
                FL::GeometryTransformation::InternalCoordinate(molorigin.geom().data(), origin.data_ptr<double>(), cartdim, intdim);
            }
            
            double zero_point = 0.0;
            if (args.gotArgument("zero_point")) zero_point = args.retrieve<double>("zero_point");
            
            std::vector<std::string> training_set = args.retrieve<std::vector<std::string>>("training_set");

            std::vector<size_t> valid_states(training_set.size(), NStates);
            if (args.gotArgument("valid_states")) {
                valid_states = args.retrieve<std::vector<size_t>>("valid_states");
                if (valid_states.size() != training_set.size()) throw std::invalid_argument("Must specify 1 valid state per training set directory");
            }

            AbInitio::DataSet<AbInitio::RegularData<float>> * RegularDataSet;
            AbInitio::DataSet<AbInitio::DegenerateData<float>> * DegenerateDataSet;
            AbInitio::InitializeTrainingSet(format, training_set, valid_states,
            origin, zero_point, intdim, RegularDataSet, DegenerateDataSet);
        }
    }
}