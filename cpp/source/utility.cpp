#include "../Cpp-Library_v1.0.0/utility.hpp"

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