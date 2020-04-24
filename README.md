# SurfGenTorch
Surface generation package based on libtorch

SurfGenTorch first contracts internal coordinates with a neural network, then casts each element of a diabatic Hamiltonian into a neural network of the contracted coordinates

In principle, any molecular property should be able to be represented by a neural network of the contracted coordinates

## Network structure
The networks utilized here are simple feedforward neural networks, with CNPI group symmetry taken into account:
* The input dimensions are assumed to have been sorted by irreducibles
* To maintain symmetry, only odd activation is allowed (except for the totally symmetric irreducible)

Dimensionality reduction network:
* Coordinates of each irreducible representation of the CNPI group contract into a single neuron
* This can also be viewed as a pretraining on geometry, so the training of this network is done in a pretraining fashion

Molecular property network:
* A property of certain irreducible does not mean it only depends the coordinates of this irreducible, so we use an N-th order polynomial of the N reduced dimensions as the input layer