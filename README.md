# GPU-accelerated Pearson corellation
A simple GPU based Pearson correlation coefficient finder made using CUDA.
## What is a Pearson correlation?
A Pearson correlation is a statistic which measures a linear correlation of a set of points (x,y).
The domain of this function is **<-1, 1>**, where:
- 1 indicates a positive correlation
- 0 indicates no correlation
- -1 indicates a negative correlation
## Formula
A mathematical formula used to calculate the coefficient:<br/>
![Pearson corr formula](https://wikimedia.org/api/rest_v1/media/math/render/svg/435a23c499a2450f0752112e69a9b808336a7cce)<br/>
*Image courtesy of Wikipedia*
## How does the program work?
The program calculates in parallel each of the components of above formula.
Every sum is calculated and reduced using GPU. This is the most time-demanding process and it's easy to make it parallel. Then, CPU performs a "finishing" process (summing output values of every GPU block) and substitutes every component into the final formula.
## Example
### Input
```
| X  | Y  |
| -- | -- |
| 43 | 99 |
| 21 | 65 |
| 25 | 79 |
| 42 | 75 |
| 57 | 87 |
| 59 | 81 |
```
### Output
```
(sigma)xi * yi = 20485.000000
(sigma)xi^2 = 11409.000000
(sigma)yi^2 = 11409.000000
(sigma)xi = 247.000000
(sigma)yi = 486.000000
r = 0.529809
```