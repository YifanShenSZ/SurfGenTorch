// General basic routine

#ifndef General_hpp
#define General_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>
#include <string>
#include <vector>
#include <iterator>

namespace general {

inline void ShowTime() {
    time_t now = time(0);
    char * dt = ctime(&now);
    std::cout << dt << '\n';
}

// Get the number of lines in a file
inline size_t NLines(const std::string & file) {
    size_t n = 0;
    std::ifstream ifs; ifs.open(file);
        while (ifs.good()) {
            std::string line;
            std::getline(ifs, line);
            n++;
        } n--;
    ifs.close();
    return n;
}

// Get the number of strings in a file
inline size_t NStrings(const std::string & file) {
    size_t n = 0;
    std::ifstream ifs; ifs.open(file);
        while (ifs.good()) {
            std::string str;
            ifs >> str;
            n++;
        } n--;
    ifs.close();
    return n;
}

// Split text with some delimiter
// text_split harvests the splitted string vector
// White space delimiter
inline void split(const std::string & text, std::vector<std::string> & text_split) {
    std::istringstream iss(text);
    std::vector<std::string> result(std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>());
    text_split = result;
}
// Any single character delimiter
inline void split(const std::string & text, std::vector<std::string> & text_split, char delimiter) {
    size_t current, previous = 0;
    current = text.find(delimiter);
    while (current != std::string::npos) {
        text_split.push_back(text.substr(previous, current - previous));
        previous = current + 1;
        current = text.find(delimiter, previous);
    }
    text_split.push_back(text.substr(previous, current - previous));
}

// Allocate (deallocate) continuous memory for an n-dimensional array A(N1, N2, ..., Nn)
// Matrix
template <typename T> void CreateArray(T ** & A, const int & N1, const int & N2) {
    if (nullptr != A) std::cout << "CreateArray warning: pointer != nullptr, the matrix may has been allocated\n";
    if (N1 == 0) throw std::invalid_argument("CreateArray error: 1st dimensionality = 0");
    if (N2 == 0) throw std::invalid_argument("CreateArray error: 2nd dimensionality = 0");
    try {
        A = new T * [N1];
        A[0] = new T[N1*N2];
        for (int i = 1; i < N1; i++) A[i] = A[i-1] + N2;
    }
    catch (std::bad_alloc & e) {
        throw e;
    }
}
template <typename T> void DeleteArray(T ** & A) {
    if (nullptr == A) {
        std::cout << "DeleteArray warning: pointer == nullptr, the matrix has been deallocated\n";
    }
    else {
        delete [] A[0];
        delete [] A;
        A = nullptr;
    }
}
// 3rd-order tensor
template <typename T> void CreateArray(T *** & A, const int & N1, const int & N2, const int & N3) {
    if (nullptr != A) std::cout << "CreateArray warning: pointer != nullptr, the 3rd-order tensor may has been allocated\n";
    if (N1 == 0) throw std::invalid_argument("CreateArray error: 1st dimensionality = 0");
    if (N2 == 0) throw std::invalid_argument("CreateArray error: 2nd dimensionality = 0");
    if (N3 == 0) throw std::invalid_argument("CreateArray error: 3rd dimensionality = 0");
    try {
        int i, j;
        A = new T ** [N1];
        for (i = 0; i < N1; i++) A[i] = new T * [N2];
        A[0][0] = new T[N1*N2*N3];
        for (j = 1; j < N2; j++) A[0][j] = A[0][j-1] + N3;
        for (i = 1; i < N1; i++) {
            A[i][0] = A[i-1][N2-1] + N3;
            for (j = 1; j < N2; j++) A[i][j] = A[i][j-1] + N3;
        }
    }
    catch (std::bad_alloc & e) {
        throw e;
    }
}
template <typename T> void DeleteArray(T *** & A, const int & N1) {
    if (nullptr == A) {
        std::cout << "DeleteArray warning: pointer == nullptr, the 3rd-order tensor has been deallocated\n";
    }
    else {
        delete [] A[0][0];
        for (int i = 0; i < N1; i++) delete [] A[i];
        delete [] A;
        A = nullptr;
    }
}

} // namespace general

#endif