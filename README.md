# SurfGenTorch
Surface generation package based on libtorch

SurfGenTorch first contracts scaled and symmetry adapted internal coordinates with a neural network, then casts each element of a diabatic Hamiltonian into a neural network of the contracted coordinates

## Network structure
The networks utilized here are simple feedforward neural networks, with CNPI group symmetry taken into account:
* To maintain symmetry, only odd activation is allowed (except for the totally symmetric irreducible)

Dimensionality reduction network:
* Coordinates of each irreducible representation of the CNPI group are contracted with a network
* This can also be viewed as a pretraining on geometry, so the training of this network is done in a pretraining fashion
* Number of weights for a irreducible with N dimensions <= (N-1)N(N+1)/3

Diabatic Hamiltonian network:
* Symmetry adapted polynomials are made from the contracted coordinates
* The input layer takes these symmetry adapted polynomials
* If 0 hidden layer, this reduces to a polynomial expansion