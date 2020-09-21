#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>
#include <CppLibrary/chemistry.hpp>
#include <FortranLibrary.hpp>

#include <libSGT.hpp>

namespace basic {
    int intdim, DefID;

    int cartdim;
    double * init_geom, * q;

    size_t state;

    argparse::ArgumentParser parse_args(const int & argc, const char ** & argv) {
        CL::utility::EchoCommand(argc, argv); std::cout << '\n';
        argparse::ArgumentParser parser("Surface generation package based on libtorch");
    
        // Required arguments
        parser.add_argument("-j","--job", 1, false, "min, sad, mex");
        parser.add_argument("-f","--format", 1, false, "Columbus7, default");
        parser.add_argument("-i","--IntCoordDef", 1, false, "an input file to define internal coordinate");
        parser.add_argument("-g","--geom", 1, false, "the initial geometry");
        parser.add_argument("-s","--state", 1, false, "the state of interest");
        parser.add_argument("--SSAIC_in",  1, false, "libSGT requirement: an input file to define scaled and symmetry adapted internal coordinate");
        parser.add_argument("--DimRed_in", 1, false, "libSGT requirement: an input file to define dimensionality reduction network");
        parser.add_argument("--Hd_in",     1, false, "libSGT requirement: an input file to define diabatic Hamiltonian (Hd)");
    
        // Optional arguments
        parser.add_argument("-d","--diabatic", 0, true, "search diabatic state instead of adiabatic");
        parser.add_argument("-o","--optimizer", 1, true, "min and mex: CG, BFGS (default = BFGS)");
    
        parser.parse_args(argc, argv);
        return parser;
    }

    // Parse command line arguments, set global variables
    // Return job type, diabatic, optimizer
    std::tuple<std::string, bool, std::string> initialize(int argc, const char** argv) {
        // Welcome
        std::cout << "Critic: critical geometry search program\n";
        std::cout << "Yifan Shen 2020\n\n";
        argparse::ArgumentParser args = parse_args(argc, argv);
        CL::utility::ShowTime();
        std::cout << '\n';
        srand((unsigned)time(NULL));

        // Retrieve command line arguments
        std::string job = args.retrieve<std::string>("job");
        std::cout << "Job type: " + job << '\n';
        std::string format = args.retrieve<std::string>("format");
        std::cout << "File format: " + format << '\n';
        std::string IntCoordDef = args.retrieve<std::string>("IntCoordDef");
        std::cout << "Internal coordinate definition is read from " + IntCoordDef << '\n';
        std::string geom = args.retrieve<std::string>("geom");
        std::cout << "Initial geometry is read from " + geom << '\n';
        state = args.retrieve<size_t>("state");
        std::cout << "State " << state << " is of interest\n";
        std::string SSAIC_in = args.retrieve<std::string>("SSAIC_in");
        std::string DimRed_in = args.retrieve<std::string>("DimRed_in");
        std::string Hd_in = args.retrieve<std::string>("Hd_in");
        bool diabatic = args.gotArgument("diabatic");
        std::string opt = "BFGS";
        if (args.gotArgument("optimizer")) opt = args.retrieve<std::string>("optimizer");
    
        // Initialize
        std::tie(intdim, DefID) = FL::GT::DefineInternalCoordinate(format, IntCoordDef);
        std::cout << "Number of internal coordinates: " << intdim << '\n';
        if (format == "Columbus7") {
            CL::chemistry::xyz_mass<double> molorigin(geom, true);
            cartdim = 3 * molorigin.NAtoms();
            init_geom = new double[cartdim];
            std::memcpy(init_geom, molorigin.geom().data(), cartdim * sizeof(double));
        }
        else {
            CL::chemistry::xyz<double> molorigin(geom, true);
            cartdim = 3 * molorigin.NAtoms();
            init_geom = new double[cartdim];
            std::memcpy(init_geom, molorigin.geom().data(), cartdim * sizeof(double));
        }
        q = new double[intdim];
        FL::GT::InternalCoordinate(init_geom, q, cartdim, intdim, DefID);
        initialize_libSGT(SSAIC_in, DimRed_in, Hd_in);

        return std::make_tuple(job, diabatic, opt);
    }
} // namespace basic