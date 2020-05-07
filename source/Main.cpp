#include <iostream>
#include <omp.h>
#include <torch/torch.h>
#include <FortranLibrary.hpp>
#include "../Cpp-Library_v1.0.0/argparse.hpp"
#include "../Cpp-Library_v1.0.0/general.hpp"
#include "../include/AbInitio.hpp"
#include "../include/DimRed.hpp"
#include "../include/utility.hpp"

argparse::ArgumentParser parse_args(const int & argc, const char ** & argv) {
    argparse::ArgumentParser parser("Surface generation package based on libtorch");
    
    // required argument
    parser.add_argument("-f", "--format", 1, false, "file format: Columbus7 or default");
    parser.add_argument("-i", "--IntCoordDef", 1, false, "internal coordinate definition file");
    parser.add_argument("-j", "--job", 1, false, "job type: pretrain, train");
    parser.add_argument("-o", "--origin", 1, false, "internal coordinate space origin");

    parser.add_argument("-c", "--check_point", 1, true, "check point to continue with");

    parser.add_argument("-r", "--restart", 0, true, "simply restart previous training");

    // pretrain
    parser.add_argument("-s", "--symmetry", '+', true, "symmetry (internal dimension / irreducible representation)");
    parser.add_argument("-d", "--data_set", '+', true, "data set directories");
    parser.add_argument("-t", "--data_type", 1, true, "data type: float, double, default = float");
    
    // train
    parser.add_argument("-z", "--zero_point", 1, true, "zero of potential energy, default = 0");
    parser.add_argument("-n", "--NStates", 1, true, "number of electronic states");
    parser.add_argument("-v", "--valid_states", '+', true, "states 1 to valid_states[i] are valid in data_set[i]");

    parser.parse_args(argc, argv);
    return parser;
}

int main(int argc, const char** argv) {
    // welcome
    std::cout << "SurfGenTorch: surface generation package based on libtorch\n";
    std::cout << "Yifan Shen 2020\n\n";
    general::ShowTime();
    std::cout << '\n';
    // global initialization
    srand((unsigned)time(NULL));

    // command line input
    std::cout << "Echo of user command line input:\n";
    std::cout << argv[0];
    for (size_t i = 1; i < argc; i++) std::cout << ' ' << argv[i];
    std::cout << "\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);

    std::string format = args.retrieve<std::string>("format");
    std::cout << "File format: " + format + "\n";

    std::string IntCoordDef = args.retrieve<std::string>("IntCoordDef");
    int intdim = FL::GeometryTransformation::DefineInternalCoordinate(format, IntCoordDef);
    std::cout << "Internal coordinate space dimension = " << intdim << '\n';
    
    std::string job = args.retrieve<std::string>("job");
    std::cout << "Job type: " + job << '\n';

    int cartdim;
    at::Tensor origin;
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
    size_t line_length = 36;
    std::cout << "The training set will be read from: ";
    for (size_t i = 0; i < data_set.size()-1; i++) {
        if (line_length > 80) {
            std::cout << '\n' << "    ";
            line_length = 4;
        }
        std::cout << data_set[i] << ", ";
        line_length += data_set[i].size() + 2;
    }
    std::cout << data_set[data_set.size()-1] << '\n';

    std::string data_type = "float";
    if (args.gotArgument("data_type")) data_type = args.retrieve<std::string>("data_type");
    std::cout << "Data type in use: " << data_type << '\n';

    if (data_type == "double") {
        AbInitio::DataSet<AbInitio::geom<double>> * GeomSet;
        AbInitio::read_GeomSet(data_set, origin, intdim, GeomSet);
        std::cout << "Number of geometries = " << GeomSet->size_int() << '\n';

        size_t batch_size = 10 * omp_get_max_threads();
        batch_size = batch_size < GeomSet->size_int() ? batch_size : GeomSet->size_int();
        std::cout << "batch size = " << batch_size << '\n';
        auto geom_loader = torch::data::make_data_loader(* GeomSet, batch_size);

        std::shared_ptr<DimRed::Net> reduction_net = std::make_shared<DimRed::Net>(symmetry);
        reduction_net->to(torch::kFloat64);

        DimRed::pretrain(reduction_net, geom_loader, batch_size);
    } else {
        AbInitio::DataSet<AbInitio::geom<float>> * GeomSet;
        AbInitio::read_GeomSet(data_set, origin, intdim, GeomSet);
        std::cout << "Number of geometries = " << GeomSet->size_int() << '\n';
    
        size_t batch_size = 10 * omp_get_max_threads();
        batch_size = batch_size < GeomSet->size_int() ? batch_size : GeomSet->size_int();
        std::cout << "batch size = " << batch_size << '\n';
        auto geom_loader = torch::data::make_data_loader(* GeomSet, batch_size);
    
        auto reduction_net = std::make_shared<DimRed::Net>(symmetry);

        DimRed::pretrain(reduction_net, geom_loader, batch_size);
    }
}

if (job == "train") {
    int NStates = args.retrieve<int>("NStates");
    
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
    auto RegularData_loader = torch::data::make_data_loader(* RegularDataSet, batch_size);
    auto DegenerateData_loader = torch::data::make_data_loader(* DegenerateDataSet, batch_size);

    FL::Chemistry::InitializePhaseFixing(NStates);
}

    std::cout << '\n';
    general::ShowTime();
    std::cout << "Mission success\n";

    return 0;
}