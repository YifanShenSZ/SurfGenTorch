#include <torch/torch.h>
#include <FortranLibrary.hpp>
#include "../Cpp-Library_v1.0.0/chemistry.hpp"

// Read the origin_file, return cartdim and origin
std::tuple<int, at::Tensor> cartdim_origin(const std::string & format, const std::string & origin_file, const int & intdim) {
    int cartdim;
    auto top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor origin = at::zeros(intdim, top);
    if (format == "Columbus7") {
        chemistry::xyz_mass<double> molorigin(origin_file, true);
        cartdim = 3 * molorigin.NAtoms();
        FL::GeometryTransformation::InternalCoordinate(molorigin.geom().data(), origin.data_ptr<double>(), cartdim, intdim);
    } else {
        chemistry::xyz<double> molorigin(origin_file, true);
        cartdim = 3 * molorigin.NAtoms();
        FL::GeometryTransformation::InternalCoordinate(molorigin.geom().data(), origin.data_ptr<double>(), cartdim, intdim);
    }
    return std::tie(cartdim, origin);
}

// Check if user inputs are directories (end with /)
// otherwise consider as lists, then read the lists for directories
std::vector<std::string> verify_data_set(const std::vector<std::string> & original_data_set) {
    std::vector<std::string> data_set;
    for (std::string item : original_data_set) {
        if (item[item.size()-1] == '/') data_set.push_back(item);
        else {
            std::string prefix = general::GetPrefix(item);
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
    std::cout << "The training set will be read from: ";
    size_t line_length = 36;
    for (size_t i = 0; i < data_set.size()-1; i++) {
        line_length += data_set[i].size() + 2;
        if (line_length > 90) {
            std::cout << '\n' << "    ";
            line_length = 4;
        }
        std::cout << data_set[i] << ", ";
    }
    line_length += data_set[data_set.size()-1].size() + 2;
    if (line_length > 90) std::cout << '\n' << "    ";
    std::cout << data_set[data_set.size()-1] << '\n';
    return data_set;
}