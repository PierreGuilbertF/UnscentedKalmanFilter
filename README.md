# UKF: Manifold Unscented Kalman Filter
### [Project Page](https://www.notion.so/unimarket/Unscented-Kalman-Filter-baf78beaf04746c89283867d52511380) | [Paper](https://kodlab.seas.upenn.edu/uploads/Arun/UKFpaper.pdf)
<br/>

This project provides a C++ implementation of a Manifold Unscented Kalman Filter (UKF). The UKF aims at estimating a set of parameters of a dynamic system with partial and noisy measurements. The parameters set is just assumed to be an element of a Riemannian Manifold (we do not assume that the the set of parameters is a vector of a Vector Space). In addition, the prediction and measurement equations are note assumed to be linear. Finally, since the prediction and measurement functions can come from a black-box function; the gaussian distribution is not propagated using differential calculous (providing the Jacobean of the functions or using automatic-differentiation) but rather by sampling the input distribution, applying the non-linear functions and then compute the moments of the output distribution.

This UKF has been used in different Computer Vision and Robotics task such as:
- Filter the parameters of a camera (orientation, position, focal, distortion, ...) after a SLAM algorithm
- Filter the dynamic parameters of a ball (position, spin)
- Filter a 3D human pose skeletons (set of keypoints with a hierarchie)

The code is providing the general architecture of a Manifold UKF:
- Sampling of the input distribution
- Prediction equations output sampling
- Measdurement equations output sampling
- Fusion of prediction and measure using the novelty and the gain

The provided example is representing the dynamic state of a camera (orientation, position, focal, distortion, ...) and is templated to accept any prediction and measurements equations. To adapt it for a new dynamic system you need to:
- Define the + operator that takes an element of the manifold and a vector of its tangent space to map it to a new element of the manifold
- Define the - operator that takes two elements of the manifold and map it to a vector of its tangent space
- Define a prediction equation

A specific example of how to use it with a specific set of parameters of a manifold which represents the state of a dynamic camera (orientation which is an element of SO(3), position, focal, principal point and distortion).
If one want to use the UKF with to estimate an other dynamic system