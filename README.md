# SurfGenTorch
Surface generation package based on libtorch

SurfGenTorch carries out a 2-step training:
1. Dimensionality reduction: contract scaled and symmetry adapted internal coordinates (SSAIC)
2. Fitting observable: cast each matrix element of an observable into a neural network of the contracted coordinates

## Dimensionality reduction
In general, a molecular geometry is specified in Cartesian coordinate. My `Cpp-Library` transforms it to internal coordinate. `SSAIC.cpp` further makes the internal coordinates scaled and symmetry adapted.

Based on SSAIC, `DimRed.cpp` utilizes simple feedforward neural networks to reduce dimensionality, with abelian CNPI group symmetry taken into account:
* Each irreducible owns an autoencoder, whose inputs are the SSAICs belonging to this irreducible
* Only the totally symmetric irreducible has bias
* The activation functions are odd, except for the totally symmetric irreducible

Implementation details:
* The linear layers are fully connected
* The activation function is tanh
* In the encoder (decoder), each layer has one less (more) neuron than the preceding
So for an irreducible with N SSAICs, the depth of the encoder (decoder) <= N - 1, the number of weights in the encoder (decoder) <= (N - 1) * N * (N + 1) / 3 + (N - 1) * N / 2

## Fitting observable
Based on the contracted SSAIC (CSSAIC), `observable_net.cpp` utilizes a simple feedforward neural network for each matrix element of an observable, with abelian CNPI group symmetry taken into account:
* The input layer takes the symmetry adapted polynomials of CSSAICs. If no hidden layer, this network reduces to a polynomial expansion 
* Only the totally symmetric irreducible has bias
* The activation functions are odd, except for the totally symmetric irreducible

Implementation details:
* The linear layers are fully connected
* The activation function is tanh
* For the starting layers, each layer has one less neuron than the preceding. The last layer directly reduces to scalar
So for an N-term symmetry adapted polynomial, the depth of the network <= N - 1, the number of weights <= (N - 1) * N * (N + 1) / 3 + (N - 1) * N / 2
