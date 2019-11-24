  The goal of an anomaly (outlier or novelty) detection method is to detect anomalous points within a data set dominated by the presence of ordinary background points. Anomalies are by definition rare and are often generated by different underlying processes [2,3]. Numerous algorithms have been devised toward this goal, the results of which have been applied to a variety of fields to improve upon domain-specific, rule-based detection methods. Anomaly detection has applications in medicine, fraud detection, fault detection, and remote sensing, among others [1, 2].
  
  Background data is often produced by non-linear processes [4]. Kernel-based learning methods are motivated by the idea that there exists a better model of the data in a transformed, nonlinear, feature space (F). A kernel function allows the efficient computation of inner products in F without the explicit calculation of the mapping. The most popular amongst these methods for anomaly detection is the One-Class Support Vector Machine (OC-SVM), which separates the data from the origin in F [5]. For a Gaussian radial-basis-function (rbf) kernel, this process is equivalent to spherically enclosing the data in F.
  
  Hoffman claims that because samples are treated independently a OC-SVM produces a boundary that is too large to tightly model the background data, causing false positives [1].Hoffman draws from the benefits of kernel techniques and the potential limitations of SVMs, by using kernel PCA (kPCA) to better model the relationship between background points [4, 6, 7]. The separation of points in F and the background model serves as an anomaly score. In an Hoffman’s evaluation on a number of real-world and toy data sets, kPCA demonstrated better generalization, accuracy, and robustness over linear PCA, the Parzen density estimator, and OC-SVMs [1].
