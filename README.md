# VBL-V001
Lab-scale Vibration Analysis Dataset and Its Machine Learning Methods


# Dataset
Download from here: [https://zenodo.org/record/7006575#.Y3W9lzPP2og](https://zenodo.org/record/7006575#.Y3W9lzPP2og).  
Locate the dataset to path like '/data/VBL-VA001`.  
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

# Citation
> Bagus Tris Atmaja, Haris Ihsannur, Suyanto, Dhany Arifianto. Lab-scale Vibration Analysis Dataset and Baseline Methods for Machinery Fault Diagnosis with Machine Learning. 2023
