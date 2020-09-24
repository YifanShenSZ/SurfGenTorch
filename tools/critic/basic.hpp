#ifndef AbInitio_hpp
#define AbInitio_hpp

#include <string>
#include <tuple>

namespace basic {
    extern int intdim, DefID;

    extern int cartdim;
    extern double * init_geom, * q;

    extern size_t state;

    // Parse command line arguments, set global variables
    // Return job type, diabatic, optimizer
    std::tuple<std::string, bool, std::string> initialize(int argc, const char** argv);
} // namespace basic

#endif