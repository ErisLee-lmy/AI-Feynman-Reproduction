# AI-Poincare-Reproduction

本项目为复旦大学石磊老师教授的电动力学的荣誉课项目，尝试复现Max Tegmark教授的prl文章结果

文章：Liu, Ziming, 和Max Tegmark. 《Machine Learning Conservation Laws from Trajectories》. *Physical Review Letters* 126, 期 18 (2021年5月6日): 180604. [https://doi.org/10.1103/PhysRevLett.126.180604](https://doi.org/10.1103/PhysRevLett.126.180604).

其中GnerateData部分用于模拟物理系统，生成相应的相空间轨迹，目前给出两个例子，分别是一维谐振子和双摆。

然后利用LocalMonteCarlo和NLCPA即可得到相应的erd。
