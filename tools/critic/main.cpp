#include <torch/torch.h>

#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>
#include <CppLibrary/chemistry.hpp>
#include <FortranLibrary.hpp>

#include <libSGT.hpp>

argparse::ArgumentParser parse_args(const int & argc, const char ** & argv) {
    CL::utility::EchoCommand(argc, argv); std::cout << '\n';
    argparse::ArgumentParser parser("Surface generation package based on libtorch");

    // Required arguments
    parser.add_argument("-j","--job", 1, false, "min, sad, mex");
    parser.add_argument("-f","--format", 1, false, "Columbus7, default");
    parser.add_argument("-i","--IntCoordDef", 1, false, "an input file to define internal coordinate");
    parser.add_argument("-g","--geom", 1, false, "the initial geometry");
    parser.add_argument("-s","--state", 1, false, "the state of interest");
    parser.add_argument("--SSAIC_in",       1, false, "libSGT requirement: an input file to define scaled and symmetry adapted internal coordinate");
    parser.add_argument("--DimRed_in",      1, false, "libSGT requirement: an input file to define dimensionality reduction network");
    parser.add_argument("--input_layer_in", 1, false, "libSGT requirement: an input file to define the symmetry adapted polynomials");
    parser.add_argument("--Hd_in",          1, false, "libSGT requirement: an input file to define diabatic Hamiltonian (Hd)");

    // Optional arguments for critical geometry search
    parser.add_argument("-d","--diabatic", 0, true, "search diabatic state instead of adiabatic");
    parser.add_argument("-o","--optimizer", 1, true, "min and mex: CG, BFGS (default = BFGS)");

    // Optional arguments for output
    parser.add_argument("-m","--mass", 1, true, "the mass of each atom");

    parser.parse_args(argc, argv);
    return parser;
}

namespace min {
void search_min(double * q, const int & intdim, size_t state, bool diabatic=false, std::string opt="BFGS");
} // namespace min

namespace sad {
void search_sad(double * q, const int & intdim, size_t state, bool diabatic=false);
} // namespace sad

namespace mex {
void search_mex(double * q, const int & intdim, size_t state, std::string opt="BFGS");
} // namespace mex

int main(int argc, const char** argv) {
    // Welcome
    std::cout << "Critic: critical geometry search program\n";
    std::cout << "Yifan Shen 2020\n\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::ShowTime();
    std::cout << '\n';
    srand((unsigned)time(NULL));

    // Retrieve command line arguments and initialize
    std::string job = args.retrieve<std::string>("job");
    std::cout << "Job type: " + job << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::cout << "File format: " + format << '\n';

    std::string IntCoordDef = args.retrieve<std::string>("IntCoordDef");
    std::cout << "Internal coordinate definition is read from " + IntCoordDef << '\n';
    int intdim, DefID;
    std::tie(intdim, DefID) = FL::GT::DefineInternalCoordinate(format, IntCoordDef);
    std::cout << "Number of internal coordinates: " << intdim << '\n';

    std::string geom = args.retrieve<std::string>("geom");
    std::cout << "Initial geometry is read from " + geom << '\n';
    int cartdim;
    std::vector<std::string> symbol;
    std::vector<double> init_geom, mass;
    if (format == "Columbus7") {
        CL::chemistry::xyz_mass<double> molecule(geom, true);
        cartdim = 3 * molecule.NAtoms();
        symbol = molecule.symbol();
        init_geom = molecule.geom();
        mass = molecule.mass();
    }
    else {
        CL::chemistry::xyz<double> molecule(geom, true);
        cartdim = 3 * molecule.NAtoms();
        symbol = molecule.symbol();
        init_geom = molecule.geom();
        if (args.gotArgument("mass")) {
            mass.resize(molecule.NAtoms());
            std::ifstream ifs; ifs.open(args.retrieve<std::string>("mass"));
                for (double & m : mass) ifs >> m;
            ifs.close();
        }
    }

    size_t state = args.retrieve<size_t>("state");
    std::cout << "State " << state << " is of interest\n";

    std::string SSAIC_in = args.retrieve<std::string>("SSAIC_in");
    std::string DimRed_in = args.retrieve<std::string>("DimRed_in");
    std::string input_layer_in = args.retrieve<std::string>("input_layer_in");
    std::string Hd_in = args.retrieve<std::string>("Hd_in");
    libSGT::initialize_libSGT(SSAIC_in, DimRed_in, input_layer_in, Hd_in);

    bool diabatic = args.gotArgument("diabatic");

    std::string opt = "BFGS";
    if (args.gotArgument("optimizer")) opt = args.retrieve<std::string>("optimizer");

    // Search for the desired critical geometry in internal coordinate
    double * q = new double[intdim];
    FL::GT::InternalCoordinate(init_geom.data(), q, cartdim, intdim, DefID);
    std::cout << std::endl;
    if (job == "min") {
        min::search_min(q, intdim, state, diabatic, opt);
    }
    else if (job == "sad") {
        sad::search_sad(q, intdim, state, diabatic);
    }
    else if (job == "mex") {

    }

    // Transform back to Cartesian coordinate then output
    std::vector<double> r(cartdim);
    FL::GT::CartesianCoordinate(q, r.data(), intdim, cartdim, init_geom.data(), DefID);
    if ((job == "min" || job == "sad") && (! mass.empty())) {
        double * BT = new double[cartdim * intdim];
        FL::GT::WilsonBMatrixAndInternalCoordinate(r.data(), BT, q, cartdim, intdim, DefID);
        double * Hessian = new double[intdim * intdim];
        at::Tensor q_tensor = at::from_blob(q, intdim, at::TensorOptions().dtype(torch::kFloat64));
        q_tensor.set_requires_grad(true);
        at::Tensor H, dH, ddH;
        if (diabatic) std::tie(H, dH, ddH) = libSGT::compute_Hd_dHd_ddHd_int(q_tensor);
        else          std::tie(H, dH, ddH) = libSGT::compute_energy_dHa_ddHa_int(q_tensor);
        std::memcpy(Hessian, ddH[state][state].data_ptr<double>(), intdim * intdim * sizeof(double));
        double * freq = new double[intdim];
        double * intmodeT = new double[intdim * intdim];
        double * LinvT = new double[intdim * intdim];
        double * cartmodeT = new double[intdim * cartdim];
        FL::GT::WilsonGFMethod(Hessian, BT, mass.data(), freq, intmodeT, LinvT, cartmodeT, intdim, cartdim/3);
        std::vector<double> r_out = r;
        for (double & el : r_out) el /= AInAU;
        for (size_t i = 0; i < intdim; i++) freq[i] /= cm_1InAu;
        FL::chem::Avogadro_Vibration(cartdim/3, symbol, r_out.data(), intdim, freq, cartmodeT, "min.log");
        delete [] BT;
        delete [] Hessian;
        delete [] freq;
        delete [] intmodeT;
        delete [] LinvT;
        delete [] cartmodeT;
    }
    for (double & el : r) el /= AInAU;
    CL::chemistry::xyz<double> molecule(symbol, r);
    molecule.print(job + ".xyz");
}