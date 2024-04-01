@article{SMDR-MRC_li2024,
  title = {Semi-Supervised Multi-Label Dimensionality Reduction Learning based on Minimizing Redundant Correlation of Specific and Common Features},
  
  author = {Runxin Li and Gaozhi Zhou and Xiaowu Li and Lianyin Jia and Zhenhong Shang},
  
  year = {2024},
  
  abstract = {Multi-label learning, like other machine learning methods, suffers from dimensionality disaster. However, due to the limitations of multi-label     
  dimensionality reduction frameworks, multi-label dimensionality reduction techniques are difficult to effectively implement semi-supervised models, instance   
  correlation constraints, and feature selection and extraction strategies. In this paper, we propose a novel semi-supervised multi-label dimensionality reduction 
  learning approach based on minimizing redundant correlation of specific and common features (SMDR-MRC, in short). Firstly, we employ matrix factorization 
  technique to transform the HSIC-based dimensionality reduction model MDDM (Y. Zhang and Z.-H. Zhou, 2010) into a least squares problem, so that the label 
  propagation mechanism can be naturally integrated into the model. Secondly, cosine similarity and the k-nearest neighbor technique are used to generate constraint 
  terms for low-dimensional manifold correlations and instance correlations, respectively. Finally, to identify the specific and common features of the low- 
  dimensional manifold structures, we introduce two projection weight matrices constrained by the l1-norm and l2,1-norm. In particular, we define a non-zero 
  correlation constraint that effectively minimizes the redundant correlation between the two matrices. To solve the optimization model with nonlinear binary 
  regular terms, we employ a novel solution approach called S-FISTA. Extensive comparison experiments on 17 multi-label benchmark datasets with the 15 top- 
  performing multi-label dimensionality reduction techniques (including a neural network technique and three feature selection methods) show that SMDR-MRC 
  outperforms all of them.},
  keywords = {Semi-supervised dimensionality reduction; Multi-label learning; Instance correlations; Feature selection; Redundant correlations}
}
