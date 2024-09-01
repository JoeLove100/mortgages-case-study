# Mortgages Case Study

This project comprises an initial look at the mortgage case study data. The repo is structured as follows:
- `common/` - contains python library code, with the intention that this would be spun out into a common library for reuse
- `data/` - contains the initial spreadsheet data, and also data generated in the notebooks for use in later sections
- `notebooks/` - contains the 3 notebooks presenting the basic data analysis and modelling

The notebooks presenting my analysis are as follows:
- `1. Data Analysis and Processing.ipynb` (1 - 2) - basic data analysis and pre-processing, generates a csv file of processed loan data to be used in later sections
- `2. Prepayment and Default Curves.ipynb` (3 - 8) - contains calculation of default/pre-payment/recovery curves, and generates some of these as csv files for use in the final section
- `3. Cashflow Forecasting.ipynb` (9 - 11) - cashflow forecasts using the model set out in the s/s, and reflections on further enhancements


# Development environment set up

## Running the notebooks
This code was written using python 3.12 running on Windows - I have not tested with lower versions or other operating systems. The libraries listed in `requirements.txt` are required to run the notebooks. These can be installed using the `pip` package manager on the terminal as follows:

```bash
pip install -r requirements.txt
```

Users can run the notebooks by calling `jupyter notebook` on the terminal. 

## Virtual environments

Virtual environments make python development easier by allowing you to use separate versions of packages. To create a new virtual environment run the following:

```bash
python -m venv venv
```

This will create a directory called `venv/`. On windows terminal, you can then *activate* the virtual environment by as follows:

```
venv/Scripts/activate
```

For other terminals/operating systems, you may need to call one of the other activate scripts in the `Scripts/` directory. Once the environment is activated you can then install the packages as per the previous section. 

To run your notebook using your virtual environment you need to create a new **kernel**. With your virtual environment activated and packages installed, run the following:

```bash
python -m ipykernel install --user --name=[KERNEL_NAME_HERE]
```

When you run your notebook, you should then be able to select this kernel. This will allow you to run your notebook using the same versions of the packages you have installed in your virtual environment.




