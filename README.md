# mtuq_cmaes

**mtuq_cmaes**A set of plug-in classes and functions to apply the **C**ovariance **m**atrix **a**daptation **e**volution **s**trategy ([CMA-ES](https://en.wikipedia.org/wiki/CMA-ES)) optimization method in [MTUQ](https://github.com/uafgeotools/mtuq).


--

> **Status**: This repo is in a work-in-progress, and approaching completion. There might be some bugs and issues that need to be resolved. Please feel free to open an issue if you encounter any problems. Please check back soon for the final version.

## Installation

As this repository is intended to be used as a plug-in for MTUQ, it is required to have it installed first. The installation instructions for the latest version of MTUQ can be found [here](https://uafgeotools.github.io/mtuq/install/index.html). We recommend using **conda** / **mamba** to manage your environment for MTUQ.

Once MTUQ is installed, the following steps can be followed to install the plug-in:

1. **Clone this repository:**

   ```bash
   git clone https://github.com/thurinj/mtuq_cmaes.git
   cd mtuq_cmaes
   ```

2. Activate the environment in which MTUQ is installed:
   
   ```bash
    conda activate your_mtuq_env
   ```

3. Install the plug-in:
   
   ```bash
   pip install .
   ```
   
## Usage

After installation, mtuq_cmaes can be imported and used as part of your MTUQ workflows. Refer to the example scripts provided in the `mtuq_cmaes/examples/` directory for a hands-on working script, based on MTUQ default examples.


## Contributing

Contributions to this repository are welcome! If you would like to contribute, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.

Please ensure that your code adheres to the existing style and includes appropriate tests.

If you encounter any issues, have questions, or want to suggest improvements, please do not hesitate to [open an issue](https://github.com/thurinj/mtuq_cmaes/issues).

## License

This repository is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for more information.
