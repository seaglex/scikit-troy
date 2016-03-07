# scikit-troy

## sktroy
- It's a python machine learning library
- It's a t(r)oy, ignoring most of efficiency issues
- It tries demostrating the key concepts of the models

## convension
The coding conversion
- prefer np.array to np.matrix
  - sometimes you have to use matrix, such as SparseMatrix
- sample/coefs is horizontal oriented, in 1-d array
- samples/coefs collection are vertical stacked, in 2-d array
- Capitalized variable for 2-d array

## models implemented
- LARS: Least Angle Regression laSso
- NAP: Nuisance Attribute projection