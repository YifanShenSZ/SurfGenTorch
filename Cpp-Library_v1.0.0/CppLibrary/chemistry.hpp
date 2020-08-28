#ifndef chemistry_hpp
#define chemistry_hpp

#include "utility.hpp"

namespace CL { namespace chemistry {

template <class T> class xyz {
    protected:
        size_t NAtoms_;
        std::vector<std::string> symbol_;
        std::vector<T> geom_;
    public:
        inline xyz() {}
        // Construct from .xyz file
        xyz(const std::string & xyzfile, bool AtomicUnit = false) {
            std::ifstream ifs; ifs.open(xyzfile);
                ifs >> NAtoms_;
                symbol_.resize(NAtoms_);
                geom_.resize(3*NAtoms_);
                for(size_t i = 0; i < NAtoms_; i++) {
                    ifs >> symbol_[i];
                    ifs >> geom_[3*i]  ;
                    ifs >> geom_[3*i+1];
                    ifs >> geom_[3*i+2];
                }
            ifs.close();
            if (AtomicUnit) {
                for(size_t i = 0; i < geom_.size(); i++) geom_[i] *= 1.8897261339212517;
            }
        }
        inline ~xyz() {}

        inline size_t NAtoms() const {return NAtoms_;}
        inline std::vector<std::string> symbol() const {return symbol_;}
        inline std::vector<T> geom() const {return geom_;}
};

template <class T> class xyz_mass : public xyz<T> {
    protected:
        std::vector<T> mass_;
    public:
        inline xyz_mass() {}
        // Construct from .xyz file and mass file
        // mass file can have arbitrary format in atomic mass unit
        xyz_mass(const std::string & xyzfile, const std::string & massfile, bool AtomicUnit = false)
        : xyz<T>(xyzfile, AtomicUnit) {
            std::ifstream ifs; ifs.open(massfile);
                mass_.resize(xyz<T>::NAtoms_);
                for(size_t i = 0; i < xyz<T>::NAtoms_; i++) {
                    ifs >> mass_[i];
                }
            ifs.close();
            if (AtomicUnit) {
                for(size_t i = 0; i < mass_.size(); i++) mass_[i] *= 1822.888486192;
            }
        }
        // Construct from Columbus7 geom file
        xyz_mass(const std::string & geomfile, bool AtomicUnit = false) {
            xyz<T>::NAtoms_ = utility::NLines(geomfile);
            std::ifstream ifs; ifs.open(geomfile);
                xyz<T>::symbol_.resize(xyz<T>::NAtoms_);
                xyz<T>::geom_.resize(3*xyz<T>::NAtoms_);
                mass_.resize(xyz<T>::NAtoms_);
                for(size_t i = 0; i < xyz<T>::NAtoms_; i++) {
                    ifs >> xyz<T>::symbol_[i];
                    T temp; ifs >> temp;
                    ifs >> xyz<T>::geom_[3*i]  ;
                    ifs >> xyz<T>::geom_[3*i+1];
                    ifs >> xyz<T>::geom_[3*i+2];
                    ifs >> mass_[i];
                }
            ifs.close();
            if (AtomicUnit) {
                for(size_t i = 0; i < mass_.size(); i++) mass_[i] *= 1822.888486192;
            }
            else {
                for(size_t i = 0; i < xyz<T>::geom_.size(); i++) xyz<T>::geom_[i] /= 1.8897261339212517;
            }
        }
        inline ~xyz_mass() {}

        inline std::vector<T> mass() const {return mass_;}
};

} // namespace chemistry
} // namespace CL

#endif