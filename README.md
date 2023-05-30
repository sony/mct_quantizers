# Model Compression Tollkit (MCT) Quantizers

This is an open-source library that provides tools that enable to easily represent a quantized neural network, both in Keras and in PyTorch.
It provides researchers, developers, and engineers a set of useful quantizers and, in addition, a simple interface for implementing new custom quantizers.

The MCT Quantizers library is developed by researchers and engineers working at Sony Semiconductor Israel.

## High level description

The quantizers interface is composed of two main components:
1. `QuantizationWrapper` - an object that takes a layer with weights and a set of weights quantizers and infer a quantized layer.
2. `ActivationQuantizationHolder` - an object that holds an activation quantizer to be quantized during inference.

The quantizers and all the quantization information for each layer can be set by initializing the weights_quantizer and activation_quantizer API.

Notice that the quantization wrapper and the quantizers are per framework.

<img src="quantization_infra.png" width="700">

## Quantizers 

The library defines the "Inferable Quantizer" interface for implementing new quantizers.
It is based on the basic class [`BaseInferableQuantizer`](common/base_inferable_quantizer.py) which allows to define quantizers that are used for emulating inference-time quantization.

On top of `BaseInferableQuantizer` we define a set of framework-specific quantizers for both weights and activations:
1. [Keras Quantizers](mct_quantizers/keras/quantizers)
2. [Pytorch Quantizers](mct_quantizers/pytorch/quantizers)

### The mark_quantizer Decorator

The [`@mark_quantizer`](mct_quantizers/common/base_inferable_quantizer.py) decorator is used to supply each quantizer with static properties which define its task compatibility. Each quantizer class should be decorated with this decorator. It defines the following properties:
 - [`QuantizationTarget`](mct_quantizers/common/base_inferable_quantizer.py): An Enum that indicates whether the quantizer is designated for weights or activations quantization.
 - [`QuantizationMethod`](mct_quantizers/common/quant_info.py): A list of quantization methods (Uniform, Symmetric, etc.).
 - `quantizer_type`: An optional property that defines the type of the quantization technique. This is a helper property to allow creating advanced quantizers for specific tasks. 
 
## Getting Started

This section provides a quick starting guide. We begin with installation via source code or pip server. Then, we provide a short usage example.

### Installation
See the MCT install guide for the pip package, and build from the source.

#### From Source
```
git clone https://github.com/sony/mct_quantizers.git
python setup.py install
```
#### From PyPi - nightly package
Currently, only a nightly released package (unstable) is available via PyPi.

```
pip install mct-quantizers-nightly
```

### Requirements

To use MCT Quantizers, one of the supported frameworks, Tensorflow/PyTorch, needs to be installed.

For use with Tensorflow please install the packages: 
[tensorflow](https://www.tensorflow.org/install), 
[tensorflow-model-optimization](https://www.tensorflow.org/model_optimization/guide/install)

For use with PyTorch please install the packages: 
[torch](https://pytorch.org/)

Also, a [requirements](requirements.txt) file can be used to set up your environment.

## License
[Apache License 2.0](LICENSE.md).
