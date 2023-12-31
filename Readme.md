# 机器学习前沿项目一

## 小组信息

|      | 姓名     | 学号          | 完成任务                                |
| ---- | -------- | ------------- | --------------------------------------- |
| 组长 | 欧阳林茁 | 2021040907008 | MLP及对应论文内容撰写，论文排版，组织   |
| 组员 | 刘宸博   | 2021150902016 | SVM以及Random Foreset及对应论文内容撰写 |

## 数据集

### ADNI

在ADNI（Alzheimer's Disease Neuroimaging Initiative）数据集中，缩写AD、MCI、MCIn、MCIp和NC代表以下含义：

1. **AD**：AD代表"Alzheimer's Disease"，即阿尔茨海默病。这是一种神经退行性疾病，通常伴随认知能力逐渐下降、记忆力丧失等症状。在ADNI数据集中，AD表示受试者已被诊断为患有阿尔茨海默病。
2. **MCI**：MCI代表"Mild Cognitive Impairment"，即轻度认知障碍。MCI是一种认知功能受损但尚未达到明显痴呆症状的状态。在ADNI数据集中，MCI表示受试者被诊断为轻度认知障碍患者。
3. **MCIn**：MCIn代表"Stable Mild Cognitive Impairment"，即稳定的轻度认知障碍。这表示患者在一段时间内，其认知功能状况相对稳定，没有显著恶化。
4. **MCIp**：MCIp代表"Progressive Mild Cognitive Impairment"，即进展性轻度认知障碍。这表示患者的认知功能状况在一段时间内有明显的恶化趋势。
5. **NC**：NC代表"Normal Cognition"，即正常认知。在ADNI数据集中，NC表示受试者的认知功能正常，没有认知障碍症状。

这些标签用于描述受试者的临床状态和认知能力水平，以便在研究中进行分类、预测和分析。研究人员可以使用这些标签来了解不同组别受试者的特征以及疾病的发展情况。

### ADNI_90_120_fMRI

在ADNI数据集中，这些缩写代表不同的临床诊断组，用于描述研究参与者的临床状态。这些缩写的含义如下：

1. **AD**：代表"Alzheimer's Disease"，即阿尔茨海默病。阿尔茨海默病是一种神经系统疾病，是老年痴呆症的一种最常见形式，通常与认知功能下降、记忆问题和其他智力和行为方面的障碍有关。AD组通常包括已被诊断为阿尔茨海默病的患者。
2. **EMCI**：代表"Early Mild Cognitive Impairment"，即早期轻度认知障碍。这一组通常包括那些在认知功能上出现轻微问题，但尚未达到明显的老年痴呆症诊断标准的个体。EMCI组的参与者可能处于疾病的早期阶段。
3. **LMCI**：代表"Late Mild Cognitive Impairment"，即晚期轻度认知障碍。这一组包括那些认知功能下降较为明显，但仍未达到明显的老年痴呆症诊断标准的个体。LMCI组的参与者可能在认知功能方面有较明显的问题。
4. **NC**：代表"Normal Control"，即正常对照组。这一组通常包括没有明显认知功能问题的健康个体，作为研究的对照组。

### FTD_90_200_fMRI

在前额叶失调症（Frontotemporal Dementia，FTD）相关研究中，"FTD" 通常代表 Frontotemporal Dementia 自身，这是一种神经系统退行性疾病，主要影响大脑的前额叶和颞叶区域，导致认知功能下降、行为和情感问题以及运动障碍。因此，在研究中，"FTD" 通常指的是患有前额叶失调症的患者。

而 "NC" 代表 "Normal Control"，即正常对照组。在研究中，这是指与患有前额叶失调症的患者进行对照的一组人，通常是年龄、性别等方面与病患组匹配的健康人群。正常对照组的数据通常用于与前额叶失调症患者的数据进行比较，以便研究人员识别潜在的疾病特征、诊断标志物或神经影像学上的差异。

总的来说，"FTD" 代表前额叶失调症，而 "NC" 代表正常对照组。这两个组别在前额叶失调症相关研究中经常用于比较和分析，以便更好地理解该疾病的特点和影响。

### OCD_90_200_fMRI

在 "OCD" 数据集中，"NC" 和 "OCD" 的缩写通常代表以下含义：

1. "NC" - Normal Control：正常对照组。在研究和实验中，正常对照组是一组与患有某种心理障碍或疾病的患者在某些关键特征上相似的健康个体。正常对照组的作用是用来与患有心理障碍或疾病的患者进行对照，以便比较他们之间的差异，帮助研究人员识别与该心理障碍或疾病相关的生理、心理或行为方面的变化。
2. "OCD" - Obsessive-Compulsive Disorder：强迫症。强迫症是一种心理障碍，表现为反复出现的强迫性思维和行为。患有强迫症的个体通常会经历强烈的焦虑，并感到他们需要履行某些仪式、行为或思维来减轻焦虑感。这种疾病可以显著干扰日常生活和功能，并需要治疗和管理。

### PPMI

在 "PPMI"（Parkinson's Progression Markers Initiative）数据集中，"PD" 和 "NC" 分别代表以下含义：

1. "PD" - Parkinson's Disease：帕金森病。帕金森病是一种慢性神经系统退行性疾病，通常表现为肌肉僵硬、震颤、运动缓慢和协调性问题。这种疾病主要影响运动能力，可能导致患者在日常生活中面临各种挑战。
2. "NC" - Normal Control：正常对照组。在研究和实验中，正常对照组是一组与患有帕金森病或其他神经系统障碍的患者在某些关键特征上相似的健康个体。正常对照组的作用是用来与患有帕金森病或其他神经系统障碍的患者进行对照，以便比较他们之间的差异，帮助研究人员识别与该疾病相关的生理、神经学或行为方面的变化。

因此，在 "PPMI" 数据集中，"PD" 表示患有帕金森病的个体，而 "NC" 表示正常对照组，用于进行比较和研究与帕金森病相关的特征和变化。这有助于科学家和医生更好地理解帕金森病的发展和进展。

## 数据维度

-  ADNI: (e.g., AD的shape为(51, 186)：51表示样本数，186表示特征维度).
- ADNI_90_120_fMRI: (e.g., AD的shape为(59, 90, 120)：59表示样本数，90表示脑区数，120表示时间序列).
- FTD_90_200_fMRI: (e.g., NC的shape为(86, 90, 200)：86表示样本数，90表示脑区数，200表示时间序列).
-  OCD_90_200_fMRI: (e.g., NC的shape为(20, 90, 200)：20表示样本数，90表示脑区数，200表示时间序列).
- PPMI: (e.g., PD的shape为(374, 294)：374表示样本数，294表示特征维度).

## 文件解析

### DeepLearning

深度学习多层感知机测试和训练的文件

### MLP

多层感知机定义的文件

### utils

各种工具函数

### dataset

多层感知器使用的nn.data.Dataset的定义

### RandomF

随机森林的定义及使用

### SVM-\*

SVM在对应数据集上的定义与使用

### 代码仓库

[机器学习项目一小组代码仓库](https://github.com/100oylz/machineLearningFutureProject)