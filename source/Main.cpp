#include <iostream>
#include <omp.h>
#include <torch/torch.h>
#include <FortranLibrary.hpp>
#include "../Cpp-Library_v1.0.0/argparse.hpp"
#include "../Cpp-Library_v1.0.0/general.hpp"
#include "../Cpp-Library_v1.0.0/chemistry.hpp"
#include "../include/AbInitio.hpp"
#include "../include/DimRed.hpp"

// Command line input
argparse::ArgumentParser parse_args(const int & argc, const char ** & argv) {
    argparse::ArgumentParser parser("Surface generation package based on libtorch");
    // file format: Columbus7 or default
    parser.add_argument("-f", "--format", 1, false);
    // internal coordinate definition file
    parser.add_argument("-i", "--internal_coordinate_definition", 1, false);
    // job type: pretrain, train, continue or analyze
    parser.add_argument("-j", "--job", 1, false);
    // simply restart previous training
    parser.add_argument("-r", "--restart");
    // ---------- Parameters to start a new fit ----------
        // number of electronic states, required for a new fit
        parser.add_argument("-n", "--NStates", 1);
        // internal coordinate space origin, required for a new fit
        parser.add_argument("-o", "--origin", 1);
        // zero of potential energy, default = 0
        parser.add_argument("-z", "--zero_point", 1);
        // data set directories, required for a new fit
        parser.add_argument("-d", "--data_set", '+');
        // states 1 to valid_states[i] are valid in data_set[i]
        parser.add_argument("-v", "--valid_states", '+');
        // fit coefficient to continue with
        parser.add_argument("-c", "--fit_coefficient", 1);
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

    srand((unsigned)time(NULL));

    argparse::ArgumentParser args = parse_args(argc, argv);

    std::string format = args.retrieve<std::string>("format");
    std::cout << "File format = " + format + "\n";

    std::string IntCoordDef = args.retrieve<std::string>("internal_coordinate_definition");
    int intdim = FL::GeometryTransformation::DefineInternalCoordinate(format, IntCoordDef);
    std::cout << "Internal coordinate space dimension = " << intdim << '\n';
    
    std::string job = args.retrieve<std::string>("job");
    std::cout << "Job type: " + job << '\n';

if (args.gotArgument("restart")) {
    // TO BE IMPLEMENTED
    return 1;
}

if (job == "pretrain") {
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

    std::vector<std::string> data_set = args.retrieve<std::vector<std::string>>("data_set");

    AbInitio::DataSet<AbInitio::geom<double>> * GeomSet;
    AbInitio::read_GeomSet(data_set, origin, intdim, GeomSet);
    std::cout << "Number of geometries = " << GeomSet->size_int() << '\n';

    size_t batch_size = omp_get_num_threads();
    std::cout << "batch size = " << batch_size << '\n';
    auto geom_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    * GeomSet, batch_size);

    int count = 0;
    for (auto & batch : * geom_loader) {
        std::cout << batch[0]->intgeom() << '\n';
        count++;
    }
    std::cout << count;

} else if (job == "train") {
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
    
    std::vector<std::string> data_set = args.retrieve<std::vector<std::string>>("data_set");
    
    std::vector<size_t> valid_states(data_set.size(), NStates);
    if (args.gotArgument("valid_states")) {
        valid_states = args.retrieve<std::vector<size_t>>("valid_states");
        if (valid_states.size() != data_set.size()) throw std::invalid_argument("Must specify 1 valid state per data set directory");
    }

    AbInitio::DataSet<AbInitio::RegularData<double>> * RegularDataSet;
    AbInitio::DataSet<AbInitio::DegenerateData<double>> * DegenerateDataSet;
    AbInitio::read_DataSet(data_set, valid_states, origin, zero_point, intdim, RegularDataSet, DegenerateDataSet);
    std::cout << "Number of regular data = " << RegularDataSet->size_int() << '\n';
    std::cout << "Number of degenerate data = " << DegenerateDataSet->size_int() << '\n';
    
    size_t batch_size = omp_get_num_threads();
    std::cout << "batch size = " << batch_size << '\n';
    auto RegularData_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    * RegularDataSet, batch_size);
    auto DegenerateData_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    * DegenerateDataSet, batch_size);
    //for (auto & batch : * RegularData_loader) {
    //    std::cout << batch[0]->energy() << '\n';
    //}
}

return 0;
}