# IoTDevID: A Behaviour-Based Fingerprinting Method for Device Identification in the IoT

# Overview
In this repository you will find a Python implementation of IoTDevID; a fingerprinting method for device identification.

[*Kahraman  Kostas,  Mike  Just,  and  Michael  A.  Lones.   IoTDevID:  A  behaviour-based  fingerprintingmethod for device identification in the IoT, arXiv preprint, arxiv:2102.08866v1, 2021.*](https://arxiv.org/abs/2102.08866v1)

This is the first version of IoTDevID . It is highly recommended that you check out [the second version](https://github.com/kahramankostas/IoTDevIDv2) as well.

# What is IoTDevID?


Device identification is one way to secure a network of IoT devices whereby devices identified as suspicious can subsequently be isolated from a network. We introduce a novel device identification (fingerprinting) method, IoTDevID, that uses machine learning to model the behaviour of IoT devices based on the network packets that they communicate. Our method uses an enhanced combination of features from previous work and includes an approach for dealing with unbalanced device data via data augmentation. We further demonstrate how to enhance device identification via a group-wise data aggregation. We provide a comparative evaluation of our method against two recent identification methods using five public IoT datasets ([ Aalto University ](https://research.aalto.fi/en/datasets/iot-devices-captures),
[ UNSW-Sydney IEEE TMC ](https://iotanalytics.unsw.edu.au/iottraces),
[ IoTFinder ](https://yourthings.info/data/),
[ UNSW-Sydney ACM SOSR*](https://iotanalytics.unsw.edu.au/attack-data), and
[ IoT Network Intrusion Dataset* ](https://ocslab.hksecurity.net/Datasets/iot-network-intrusion-dataset)) which together contain data from over 100 devices, two of which include both benign and malicious data. Through our evaluation we demonstrate improved performance over previous results with F1 scores above 99%, with considerable improvement gained from data aggregation.


# Requirements and Infrastructure: 

Python 3.6 was used to create the application files. Before running the files, it must be ensured that [Python 3.6](https://www.python.org/downloads/) and the following libraries are installed.

| Library | Task |
| ------ | ------ |
|[ Scapy ](https://scapy.net/)| Packet(Pcap) crafting |
|[ Sklearn ](http://scikit-learn.org/stable/install.html)| Machine Learning & Data Preparation |
|[ Imblearn ](https://pypi.org/project/imblearn)| Data Augmentation |
| [ Numpy ](http://www.numpy.org/) |Mathematical Operations|
| [ Pandas  ](https://pandas.pydata.org/pandas-docs/stable/install.html)|  Data Analysis|
| [ Matplotlib ](https://matplotlib.org/users/installing.html) |Graphics and Visuality|
| [Seaborn ](https://seaborn.pydata.org/) |Graphics and Visuality|

The technical features of the computer used for experiments are given below.

|  | |   |
| ------ |--|  ------ |
|Central Processing Unit|:|Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz 2.90 GHz|
| Random Access Memory	|:|	8 GB (7.74 GB usable)|
| Operating System	|:|	Windows 10 Pro 64-bit |
| Graphics Processing Unit	|:|	AMD Readon (TM) 530|

# Implementation: 

The implementation phase consists of 5 steps, which are:

* Fingerprinting
* Initial Fingerprint Method Evaluation
* Data Augmentation
* Augmentated and Aggregated Fingerprint Method Evaluation
* Malicious Device Dataset Evaluation*


Each of these steps contains one or more Python files. The same file was saved with both "py" and "ipynb" extensions. The code they contain is exactly the same. The file with the ipynb extension has the advantage of saving the state of the last run of that file and the screen output. Thus, screen output can be seen without re-running the files. Files with the ipynb extension can be run using the [jupyter notebook](http://jupyter.org/install) program. 









## Fingerprinting

This step contains the [1.1 PCAP2CSV.ipynb](https://github.com/kahramankostas/IoTDevID/blob/master/1.1%20PCAP2CSV.ipynb) file. This file converts the files with pcap extension to single packet-based, csv extension fingerprint files (IoT Sentinel, IoTSense, IoTDevID individual packet based feature sets) and makes labeling.


## Initial Fingerprint Method Evaluation

This step contains the [2.1 Classification of Individual packets for Aalto University Dataset](https://github.com/kahramankostas/IoTDevID/blob/master/2.1%20Classification%20of%20Individual%20packets%20for%20Aalto%20University%20Dataset.ipynb) file. This file makes machine learning application for individual packets for Aalto University and  allows to compare 3 different featuresets (IoT Sentinel, IoTSense, IoTDevID individual packet based feature sets). It uses these algorithms: RF (Random Forest), NB (Naïve Bayes), kNN (k-Nearest Neighbours), GB (Gradient Boosting), DT (Decision Trees), and SVM (Support Vector Machine)

## Data Augmentation
This step contains the [3.1 Data Augmentation.ipynb](https://github.com/kahramankostas/IoTDevID/blob/master/3.1%20Data%20Augmentation.ipynb) file. This file first divides the  datasets into two as train and test. It then applies data augmentation for the required classes using resampling and SMOTE methods.


## Augmentated and Aggregated Fingerprint Method Evaluation
This step contains these 4  files:

[4.1 Aalto university results  with augmentation and aggregation.ipynb](https://github.com/kahramankostas/IoTDevID/blob/master/4.1%20Aalto%20university%20results%20%20with%20augmentation%20and%20aggregation.ipynb) file makes machine learning (RF) application for augmented version of Aalto University dataset based individual packet level using IoTDevID method. It then produces results for 4 different group sizes (3, 6, 9, 12) using the packet aggregation method.

[4.2 IoTfinder results  with augmentation and aggregation](https://github.com/kahramankostas/IoTDevID/blob/master/4.2%20IoTfinder%20results%20%20with%20augmentation%20and%20aggregation.ipynb) file makes machine learning (RF) application for augmented version of IoTfinder dataset based individual packet level using IoTDevID method. It then produces results for 4 different group sizes (3, 6, 9, 12) using the packet aggregation method.


[4.3 UNSW_benign_ results  with augmentation and aggregation](https://github.com/kahramankostas/IoTDevID/blob/master/4.3%20UNSW_benign_%20results%20%20with%20augmentation%20and%20aggregation.ipynb) file makes machine learning (RF) application for augmented version of UNSW-Sydney IEEE TMC dataset based individual packet level using IoTDevID method. It then produces results for 4 different group sizes (3, 6, 9, 12) using the packet aggregation method.


[4.4 Aalto university results  with combined labels.ipynb](https://github.com/kahramankostas/IoTDevID/blob/master/4.4%20Aalto%20university%20results%20%20with%20combined%20labels.ipynb) file makes machine learning (RF) application for augmented version of Aalto University dataset based individual packet level using IoTDevID method. It then produces results for 4 different group sizes (3, 6, 9, 12) using the packet aggregation method. However, in this file, very similar devices are considered as a group in the Aalto University dataset and collected under the same label.



## Malicious Device Dataset Evaluation*
This step contains the [5.1 UNSW_Malicious_ results  with augmentation and aggregation](https://github.com/kahramankostas/IoTDevID/blob/master/5.1%20UNSW_Malicious_%20results%20%20with%20augmentation%20and%20aggregation.ipynb) file. This file makes machine learning (RF) application for  [ UNSW-Sydney ACM SOSR](https://iotanalytics.unsw.edu.au/attack-data) and  [ IoT Network Intrusion ](https://ocslab.hksecurity.net/Datasets/iot-network-intrusion-dataset)datasets based individual packet level using IoTDevID method. It then produces results for 4 different group sizes (3, 6, 9, 12) using the packet aggregation method. However, unlike other steps, this step contains benign and malicious data produced by the same devices. The purpose is not to prevent these attacks, but to show that the device can be detected if it behaves differently. Therefore, not all data of malicious datasets are used. The data used includes only cases where IoT devices are attacker. 
Before creating the fingerprint for this process, we parsed the pcap files as benign and malicious, and then extracted the fingerprints. The information required for the filtering process are clearly stated on the datasets website. You can perform these operations using [Wireshark](https://www.wireshark.org/). You can also use [tshark-filter](https://github.com/kahramankostas/tshark-filter) to automate this process.





# Full Datasets

The processed datasets are shared in depository. However, raw versions of the datasets used in the study and their addresses are given below.

| Dataset | capture year | Number of Devices | Type |
|---|---|---|---|
|[ Aalto University ](https://research.aalto.fi/en/datasets/iot-devices-captures)| 2016|31|Benign|
|[ UNSW-Sydney IEEE TMC ](https://iotanalytics.unsw.edu.au/iottraces)| 2016|31|Benign|
|[ IoTFinder ](https://yourthings.info/data/)| 2018|51|Benign|
|[ UNSW-Sydney ACM SOSR*](https://iotanalytics.unsw.edu.au/attack-data)| 2018|28|Benign & Malicious|
|[ IoT Network Intrusion Dataset* ](https://ocslab.hksecurity.net/Datasets/iot-network-intrusion-dataset)| 2019|2|Benign & Malicious|


# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


# Citations
If you use the source code please cite the following paper:

*Kahraman  Kostas,  Mike  Just,  and  Michael  A.  Lones.   IoTDevID:  A  behaviour-based  fingerprintingmethod for device identification in the IoT, arXiv preprint, arxiv:2102.08866, 2021.*


```
@misc{kostas2021iotdevid,
      title={{IoTDevID}: A Behaviour-Based Fingerprinting Method for Device Identification in the {IoT}}, 
      author={Kahraman Kostas and Mike Just and Michael A. Lones},
      year={2021},
      eprint={2102.08866v1},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```





Contact:
*Kahraman Kostas
kkostas@utexas.edu*

## _____________________________________________________

*Items with the * sign are not included in the paper. They have been prepared for a longer version of it.
