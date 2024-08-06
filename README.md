# Model Compression Toolkit (MCT) Quantizers

The MCT Quantizers library is an open-source library developed by researchers and engineers working at Sony Semiconductor Israel. 

It provides tools for easily representing a quantized neural network in both Keras and PyTorch. The library offers researchers, developers, and engineers a set of useful quantizers, along with a simple interface for implementing new custom quantizers.

## High level description:

The library's quantizers interface consists of two main components:

1. `QuantizationWrapper`: This object takes a layer with weights and a set of weight quantizers to infer a quantized layer.
2. `ActivationQuantizationHolder`: An object that holds an activation quantizer to be used during inference.

Users can set the quantizers and all the quantization information for each layer by initializing the weights_quantizer and activation_quantizer API.

Please note that the quantization wrapper and the quantizers are framework-specific.

<img src="https://github.com/sony/mct_quantizers/raw/main/quantization_infra.png" width="700">

## Quantizers:

The library provides the "Inferable Quantizer" interface for implementing new quantizers. 
This interface is based on the [`BaseInferableQuantizer`](https://github.com/sony/mct_quantizers/blob/main/mct_quantizers/common/base_inferable_quantizer.py) class, which allows the definition of quantizers used for emulating inference-time quantization.

On top of `BaseInferableQuantizer` the library defines a set of framework-specific quantizers for both weights and activations:
1. [Keras Quantizers](https://github.com/sony/mct_quantizers/tree/main/mct_quantizers/keras/quantizers)
2. [Pytorch Quantizers](https://github.com/sony/mct_quantizers/tree/main/mct_quantizers/pytorch/quantizers)

### The mark_quantizer Decorator

The [`@mark_quantizer`](https://github.com/sony/mct_quantizers/blob/main/mct_quantizers/common/base_inferable_quantizer.py) decorator is used to assign each quantizer with static properties that define its task compatibility. Each quantizer class should be decorated with this decorator, which defines the following properties:
 - [`QuantizationTarget`](https://github.com/sony/mct_quantizers/blob/main/mct_quantizers/common/base_inferable_quantizer.py): An Enum that indicates whether the quantizer is intended for weights or activations quantization.
 - [`QuantizationMethod`](https://github.com/sony/mct_quantizers/blob/main/mct_quantizers/common/quant_info.py): A list of quantization methods (Uniform, Symmetric, etc.).
 - `identifier`: A unique identifier for the quantizer class. This is a helper property that allows the creation of advanced quantizers for specific tasks.
 
## Getting Started

This section provides a quick guide to getting started. We begin with the installation process, either via source code or the pip server. Then, we provide a short example of usage.

### Installation

#### From PyPi - mct-quantizers package
To install the latest stable release of MCT Quantizer from PyPi, run the following command:
```
pip install mct-quantizers
```

If you prefer to use the nightly package (unstable version), you can install it with the following command:
```
pip install mct-quantizers-nightly
```

#### From Source
To work with the MCT Quantizers source code, follow these steps:
```
git clone https://github.com/sony/mct_quantizers.git
cd mct_quantizers
python setup.py install
```

### Requirements

To use MCT Quantizers, you need to have one of the supported frameworks, Tensorflow or PyTorch, installed.

For use with Tensorflow, please install the following package:
[tensorflow](https://www.tensorflow.org/install),

For use with PyTorch, please install the following package:
[torch](https://pytorch.org/)

You can also use the [requirements](https://github.com/sony/mct_quantizers/blob/main/requirements.txt) file to set up your environment.

## License
[Apache License 2.0](https://github.com/sony/mct_quantizers/blob/main/LICENSE.md).
