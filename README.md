# NumPyML
This a fully extensible, modular library for training CNNs, written only using NumPy. It is meant to help people understand how forward and backpropagation happens through the most popular Neural Network layers. The code is written to be easy to read and modify. No automatic gradient calculation or non-Python code.

## Branches in this project
The **master** branch of this project is the slow but aims to be easy to read. Given a choice between optimization and readability, this branch will always have readable code. The **numba** branch uses just in time compilation provided by numba to execute code significantly faster.
