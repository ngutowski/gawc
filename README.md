# Genetic Algorithm with multi-objective Compass (*GAwC*)

This repository aims to share the code of the GAwC method published in the journal *Artificial Intelligence in Medicine (Elsevier)*. Please note that this code is intended for experimental purposes only and is not suitable for production.

## The Algorithm

### A Novel Approach
We propose a novel approach for multi-objective feature selection for medical binary classification. This approach hybrids and extends two recent original methods: A genetic algorithm combined with a Machine Learning classifier that we published in 2018 in Vascular journal https://journals.sagepub.com/home/vas, and an extension of the compass-based fitness computation that we published in 2021 in Information Sciences journal https://www.journals.elsevier.com/information-sciences. The Vascular article entitled "Identification of new factors associated to walking impairment in patients with vascular-type claudication" (DOI: 10.1177/1708538118813708, 9 pages) originally proposed to apply a genetic algorithm combined with a SVM classifier to identify new variables associated to walking limitation in patients with claudication. The Information Sciences article entitled "Gorthaur-EXP3: Bandit-based selection from a portfolio of recommendation algorithms balancing the accuracy-diversity dilemma" (DOI:~10.1016/j.ins.2020.08.106, 18 pages) originally proposed a novel approach for the dynamic selection of recommendation algorithms using the EXP3 adversarial bandit algorithm combined with a 2-Dimensional (Accuracy and Diversity) compass-based fitness calculation.

### Genetic Algorithm + Compass + ELM classification method
In order to deal with multi-objective feature selection for medical binary classification, we adapt and extend the 2D fitness calculation to a 3-Dimensional one, each dimension corresponding to a criteria among accuracy, number of features, and AUC ROC score. Then, we integrate this new 3D compass approach into the original genetic algorithm. Finally, Extrem Machine Learning (ELM) is used for the classification process.

## Datasets

*GAwC* is originally designed for Medicine purpose. With the delivered repository, it can be evaluated over several datasets found in UCI Machine Learning: Heart (Statlog), Diabetes, Musk1, Breast cancer, Cardiotocography. Note that, if needed, you can evaluate the method using other biclasses datasets that are not necessarily in the field of medicine. For this, you will just have to save your own dataset in the *./compass/data/* folder and update the right JSON parameter file.


## Environment
If using Conda, run the following command (where *gawc-env-linux.yml* is) to install the full environment:
```
conda env create -f gawc-env-linux.yml
```


## List of parameters

### JSON file
Input parameters of experiments must be set in JSON files in *./compass/params/*\
JSON parameters file example on Diabetes dataset:
```
{
  "nbCV": 10,
  "norm": "true",
  "mlMethod": "ELM",
  "datasetRes": "myTest",
  "injIndiv": "false",
  "nbIndiv": 128,
  "dimMode": "3D",
  "acc": "true",
  "nbRun": 1,
  "trainRatio": 1,
  "datasetName": "diabetes.xlsx",
  "maxgen": 100,
  "nbFeatRatio": 1,
  "resFile": "diabetesRes.txt",
  "exploreType": "range",
  "range": "[80,80]",
  "seed":-1
}
```
### Parameters description :
* **nbCV** (integer): number of Cross-Validation you need to perform.
* **norm** (Boolean): if true, program will process to features normalization.
* **mlMethod** (String): even though we evaluate with ELM, this code is delivered with different ML classification methods that you can use for comparison => Linear Regression ("RLM"); Lineare Discrimant Analysis ("LDA"); Random Forest Classifier ("RF"); Logistical Regression ("LR"); Support Vector Machine ("SVM"); Gaussian Naive Bayes ("GNB") ; Bayesian Ridge ("BR") ;  Ridge ("RID") ; Elastic Net ("ELAS") ; LASSO ("LASSO").
* **datasetRes** (String): dataset folder we want to save results.
* **injIndiv** (Boolean): the program can be stopped at any time because the individual with the best fitness is stored over the runs. Thus, in the event of a new run, the program offers the possibility of re-injecting this best individual into the starting population or not.
* **nbIndiv** (Integer): the population size of the genetic algorithm reaches 128 individuals. The fitness function will be calculated on each individual.
* **dimMode** (String): default set to "3D". This parameter is available for further evolution and evaluation e.g., 1D, 2D, 4D, ..., nD.
* **acc** (Boolean): default to "true". If "false", program will keep in memory the best results of AUC instead of Accuracy (acc) at final (print display, graph and table).
* **nbRun** (Integer): number of run to perform in order to compute average and standard deviation of the n-CV (e.g., n=10).
* **trainRatio** (Float): the ratio parameter is initialized by default to 1. Thus at 1, the measured performances are carried out in the whole dataset, on the basis of n-cross validations (e.g., n=10). This parameter allows in the future to parameterize the ratio to 0.75 (for example) which means that fitness of the genetic algorithm will be measured in 75% of all the dataset. Then, the remaing 25% data could be used to test the performances of the fitness fonction in this "blind" test set.
* **datasetName** (String): dataset name with extension (.xlsx) on which we want to evaluate GAwC.
* **maxgen** (Interger): number of maximum generations to try before stoping mutations and experiment.
* **nbFeatRatio** (Float): default set to 1. nbFeatRatio denotes a parameter rho. One needs to scale the computed number of features score to bound subset cardinality to a kind of maximum of features to select. Number of features score ScFeat is computed as follows. *ScFeat = 1−rho/|S|* where *|S|* is the total number of features.
* **resFile** (String): results file name that will be stored in datasetRes folder.
* **exploreType** (String): possible values are "range", "range-micro","quarter","low","high","very low", "very high", "mid-high", "mid-low", "mid", "partial": 
  - "range" allows to explore with a step of 1 degree whereas "range-micro" allows to scan with a step of 0.5 degree; 
  - "quarter" allows to explore solutions from 0 to 90 with a step of 1 degree;
  - "partial" allows to explore 9 typical angles (*11pi/24, 5pi/12, 3pi/8, pi/3, pi/4, pi/6, pi/8, pi/12, pi/24*);
  - parameters like e.g., low, mid or high are preset parameters that allow to explore solution in a limited part of the compass (e.g., "low" explores small angles compass solution whereas "high" explores high angles compass solution. "mid" explores balanced solution between objectives).
* **range** (String with integers interval): useful in "range" or "range-micro" modes only. Set *[x,y]* where *x* is the degree to start the exploration and y is the degree to stop. Advices are to explore solution in a range of 1 to 3 degrees (e.g., "[80,80]" or "[80,81]" or "[80,82]").
* **seed** (Integer): default set to -1. -1 means seeds will be generated at random. You can also set any positive value to play the desired seed or replay a seed you witch which you ran before.



## How to run GAwC ?

By using any IDE (e.g., PyCharm, Spyder) run the execute.py with JSON file parameters, or in command line as follows:
```
python3 execute.py jsonParametersFile.json
```


## Quick execution (example on Diabetes dataset, see JSON above)

By using any IDE (e.g., PyCharm, Spyder) run the execute.py with JSON file parameters, or in command line as follows:
```
python3 execute.py paramsTestDiabetes.json
```
*Reminder : example JSON paramaters file (paramsTestDiabetes.json) inputs a single run of 10 CV, evaluates GAwC with 3D compass 80° using ELM classifier on Diabetes dataset, and max generation is set to 100. Results are stored in ./compass/results/myTest/...* 

## Technical settings

This code can be run in a classical computer. Nevertheless, we advise using a minimum of Intel(R) Core(TM) i7 CPU. Note that for all our experiments of our research, evaluation was performed on a high performance computing cluster of 27 nodes (composed of 700 logical CPU, 2 nvidia GPU tesla k20m, 1 nvidia P100 GPU, 120 TB of beegfs scratch storage). Using such cluster allows to paralellize runs and compute average after the final reduce step ends. Hence, using our cluster divided experiments time by 30. If interested by using a cluster, we recommend using *.slurm* file with *sbatch* command. 

## Additional informations  
More information about related work, *GAwC* method, parameterization, results, discussion and perspectives are addressed in a journal paper. *GAwC* has been published in the journal *Artificial Intelligence in Medicine (Elsevier)* https://www.sciencedirect.com/science/article/pii/S0933365722000422 
