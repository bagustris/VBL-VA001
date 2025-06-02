# VBL-V001
Baseline methods for the paper [Lab-scale Vibration Analysis Dataset and Baseline Methods for Machinery Fault Diagnosis with Machine Learning](https://arxiv.org/abs/2212.14732).

# Dataset
Download from here: [https://zenodo.org/record/7006575#.Y3W9lzPP2og](https://zenodo.org/record/7006575#.Y3W9lzPP2og).  
Locate the dataset in a path like `/data/VBL-VA001`.  
Structure of dataset:  
```bash
bagus@m049:VBL-VA001$ tree -L 2 . --filelimit 100
.
├── bearing [1000 entries exceeds filelimit, not opening dir]
├── misalignment [1000 entries exceeds filelimit, not opening dir]
├── normal [1000 entries exceeds filelimit, not opening dir]
└── unbalance [1000 entries exceeds filelimit, not opening dir]

4 directories, 4000 files
```


You can also try the extracted feature under `data` directory and run 
the following codes.


# Running the program
```bash
# First, extract the feature
$ python3 extract_feature.py
# Then you can run any train_* program, i.e.,:
$ python3 train_svm.py
Shape of Train Data : (3200, 27)
Shape of Test Data : (800, 27)
Optimal C: 69
Max test accuracy: 1.0
```

# Note on BPFO/BPFI

The BPFO and BPFI values are obtained from the pump bearing type datasheet, namely type NTN Bearing 6201, which has a BPFO coefficient of 2.62 and a BPFI coefficient of 4.38.


# Citation (Bibtex)
  ```bibtex
  @ARTICLE{Atmaja2023,  
	author = {Atmaja, Bagus Tris and Ihsannur, Haris and Suyanto and Arifianto, Dhany},  
	title = {Lab-Scale Vibration Analysis Dataset and Baseline Methods for Machinery Fault Diagnosis with Machine Learning},  
	year = {2023},  
	journal = {Journal of Vibration Engineering and Technologies},  
	doi = {10.1007/s42417-023-00959-9},  
	type = {Article},  
	publication_stage = {Article in press},  
	source = {Scopus},  
}
```
