# Signal-Processing

## Compressed sensing 

This project was done as a part of the Numerical Analysis research group at GSU under the advisement of Dr. Xiaojing Ye. The project centers around creating a normally distributed 
toy dataset to perform regular and accelerated variation of the proximal gradient method to achieve compressed sensing. The toy data set was created sparse for the orignial problem;
a variation of the problem was implemented for non-sparse signals using a Wavelet transform to first sparsify the data, then perform accelerated compressed sensing and then use an
inverse transform to bring it back into the original domain of the signal. 

The reader is encouraged to read through the [final research report](https://github.com/asuresh213/Signal-Processing/blob/master/Reading_Course_Project___Compressed_Sensing%20(4).pdf) to look through the set up, the mathematics and the results

## What needs work?
1. While most of the mathematics is optimized, the code itself needs some clever optimization to speed up the sampling process.
