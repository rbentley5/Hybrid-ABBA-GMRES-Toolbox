# GPU-CT-Unmatched-back-projectors

This python toolbox provides methods to solve massive scale X-ray computerized tomography. The methods provided includes Hybrid AB-GMRES, Hybrid BA-GMRES, both of which use unmatched projectors and are capable of solving normal and unnormal equations that come from CT problems.

The Hybrid ABBA-GMRES methods work by incorporating Tikhonav regularization into ABBA-GMRES. They support the following features:
- Dense, sparse, or abstract matrices
- Using restarting to improve efficency
- Different methods to choose regularization parameter for Tikhonav
- Multiple automatic stopping criterion
  

## Package Requirements
- numpy
- ASTRA Toolbox
- trips-py
- pylops
- GPUtil
