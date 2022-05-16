#ifndef ALGORITHMS_UKF_INCLUDE_UNSCENTED_KALMAN_FILTER_H_
#define ALGORITHMS_UKF_INCLUDE_UNSCENTED_KALMAN_FILTER_H_

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>                        // NOLINT(build/include_order)
#include <Eigen/Geometry>                     // NOLINT(build/include_order)
#include <unsupported/Eigen/MatrixFunctions>  // NOLINT(build/include_order)

// The algorithm is inspired by the following papers:
// - https://kodlab.seas.upenn.edu/uploads/Arun/UKFpaper.pdf
// - https://arxiv.org/pdf/1107.1119.pdf
namespace ukf {

/// Dimensions of the UKF internal state manifold and measurements manifold
const unsigned int kStateDims = 14;
const unsigned int kMeasDims = 10;
using StateVector = Eigen::Matrix<double, kStateDims, 1>;
using MeasVector = Eigen::Matrix<double, kMeasDims, 1>;
using StateVariance = Eigen::Matrix<double, kStateDims, kStateDims>;
using MeasVariance = Eigen::Matrix<double, kMeasDims, kMeasDims>;

/// Generates 2N points on a N-manifold that have the same mean, covariance and skewness
/// than the input distribution determined by the two first moments. These points
/// can then be used to estimate the distribution after a non-linear transformation
/// Note that these points are actually located at the 1-sigma isosurface of gaussian distribution
template <typename ManifoldElement, unsigned int N>
std::vector<ManifoldElement> GenerateSigmaPoints(ManifoldElement m, Eigen::Matrix<double, N, N> M);

//--------------------------------------------------------------------------------------------------
// represent an element of the SO3 lie group
template <typename T>
class SO3 {
  public:
    // methods
    SO3() = default;
    explicit SO3(Eigen::Quaternion<T> element) { element_ = element; }
    void FromEulerAngle(T rx, T ry, T rz);
    void SetElement(Eigen::Quaternion<T> element) { element_ = element; }
    Eigen::Matrix<T, 3, 3> ToRotationMatrix() const;
    Eigen::Quaternion<T> ToQuaternion() const;
    SO3<T> operator+(Eigen::Matrix<T, 3, 1> v) const;
    Eigen::Matrix<T, 3, 1> operator-(SO3 y) const;
    Eigen::Matrix<T, 3, 1> ToAxisAngle() const;

  protected:
    Eigen::Quaternion<T> element_;
};

//--------------------------------------------------------------------------------------------------
/// Represents an element of the manifold of all the possible camera calibrations. This
/// is not a vector space since SO3 has a lie algebra structure but not a vector space structure
class CalibrationStateManifoldElement {
  public:
    CalibrationStateManifoldElement() = default;
    CalibrationStateManifoldElement(Eigen::Vector3d position, SO3<double> orientation,
                                    Eigen::Vector3d angular_velocity, double focal,
                                    double focal_velocity, Eigen::Vector2d principal_point,
                                    double distortion);

    // Addition operator: Manifold x Tangent Vector Space -> Manifold
    CalibrationStateManifoldElement operator+(StateVector v);

    // Subtraction operator: Manifold x Manifold -> Tangent Vector Space
    StateVector operator-(CalibrationStateManifoldElement y);

    // Compute the mean of sample points belonging to the manifold. Since the mean is defined as
    // being the element that minimises the sum of the squared distances, the manifold needs to be
    // Riemannian.
    static CalibrationStateManifoldElement ComputeMean(
            std::vector<CalibrationStateManifoldElement> samples);

    std::string PrintSelf();

    Eigen::Vector3d position_;
    SO3<double> orientation_;
    Eigen::Vector3d angular_velocity_;
    double focal_;
    double focal_velocity_;
    Eigen::Vector2d principal_point_;
    double distortion_;
};

//--------------------------------------------------------------------------------------------------
/// Represents an element of the manifold of all the possible calibration measures. This
/// is not a vector space since SO3 has a lie algebra structure but not a vector space structure
class CalibrationMeasureManifoldElement {
  public:
    CalibrationMeasureManifoldElement() = default;
    CalibrationMeasureManifoldElement(Eigen::Vector3d position, SO3<double> orientation,
                                      double focal, Eigen::Vector2d principal_point,
                                      double distortion);

    // Addition operator: Manifold x Tangent Vector Space -> Manifold
    CalibrationMeasureManifoldElement operator+(MeasVector v);

    // Subtraction operator: Manifold x Manifold -> Tangent Vector Space
    MeasVector operator-(CalibrationMeasureManifoldElement y);

    // Compute the mean of samples points belonging to the manifold. Since the mean is defined as
    // being the element thast minimises sum of the squared distances, the manifold needs to be
    // Riemannian.
    static CalibrationMeasureManifoldElement ComputeMean(
            std::vector<CalibrationMeasureManifoldElement> samples);

    std::string PrintSelf();

    Eigen::Vector3d position_;
    SO3<double> orientation_;
    double focal_;
    Eigen::Vector2d principal_point_;
    double distortion_;
};

//--------------------------------------------------------------------------------------------------
/// Represent the state of an estimation (resp measure) by its element on the manifold and its
/// corresponding variance-covariance matrix defines on the tangent vector space
struct PredictionState {
    CalibrationStateManifoldElement predicted_element;
    StateVariance prediction_variance;
    CalibrationMeasureManifoldElement predicted_measure;
    MeasVariance measure_prediction_variance;
    Eigen::Matrix<double, kStateDims, kMeasDims> cross_covariance;
};

//--------------------------------------------------------------------------------------------------
/// Represents the current state of the broadcast camera estimation. The status is determined
/// by the best estimation which is an element of the calibration state manifold and by the
/// estimation uncertainty represented by the minimal-DoF variance-covariance matrix
template <typename F, typename H>
class BroadcastCameraEstimation {
  public:
    BroadcastCameraEstimation() = default;
    BroadcastCameraEstimation(CalibrationStateManifoldElement element,
                              StateVariance variance_matrix = StateVariance::Identity());

    // Methods
    // Add a measure to the kalman filter. The prediction step will be computed
    void NewMeasure(double dt, CalibrationMeasureManifoldElement measure,
                    MeasVariance measure_variance);

    // Predict the next state of the kalman filter: best estimator and and uncertainty
    PredictionState Prediction(double dt);

    // Generate the variance-covariance matrix of the motion model
    StateVariance GenerateMotionModelUncertainty(double dt);

    // Get / Set element
    CalibrationStateManifoldElement & element() { return element_; }
    const CalibrationStateManifoldElement & element() const { return element_; }

    // Get / Set variance_matrix
    StateVariance & variance_matrix() { return variance_matrix_; }
    const StateVariance & variance_matrix() const { return variance_matrix_; }

    // Get / Set uncertainty parameters
    double & max_angular_acceleration() { return max_angular_acceleration_; }
    const double & max_angular_acceleration() const { return max_angular_acceleration_; }

    double & max_focal_acceleration() { return max_focal_acceleration_; }
    const double & max_focal_acceleration() const { return max_focal_acceleration_; }

  protected:
    // represents the the best estimator and the related uncertainty
    CalibrationStateManifoldElement element_;
    StateVariance variance_matrix_ = StateVariance::Identity();

    // Prediction and measure equations
    F prediction_;
    H measure_;

    // uncertainty of motion model
    double max_angular_acceleration_ = 16.0 * M_PI;
    double position_uncertainty_ = 0.0;
    double angular_velocity_uncertainty_ = 0.0;
    double focal_velocity_uncertainty_ = 0.0;
    double max_focal_acceleration_ = 25.0;
    double distortion_uncertainty_ = 0.00;
    double principal_point_uncertainty_ = 0.0;
};

//--------------------------------------------------------------------------------------------------
class FixedPositionConstantAngularVelocityPrediction {
  public:
    FixedPositionConstantAngularVelocityPrediction() = default;

    CalibrationStateManifoldElement operator()(CalibrationStateManifoldElement X, double dt);
};

//--------------------------------------------------------------------------------------------------
class BundleAdjustmentMeasurement {
  public:
    CalibrationMeasureManifoldElement operator()(CalibrationStateManifoldElement X);
};

//--------------------------------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, 3, 3> VectorToSkew(Eigen::Matrix<T, 3, 1> v) {
    Eigen::Matrix<T, 3, 3> skewV;
    skewV << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return skewV;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, 3, 1> SkewToVector(Eigen::Matrix<T, 3, 3> V) {
    Eigen::Matrix<T, 3, 1> v;
    v << V(2, 1), V(0, 2), V(1, 0);
    return v;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
void SO3<T>::FromEulerAngle(T rx, T ry, T rz) {
    Eigen::Matrix<T, 3, 3> R(Eigen::AngleAxis<T>(rz, Eigen::Matrix<T, 3, 1>::UnitZ()) *
                             Eigen::AngleAxis<T>(ry, Eigen::Matrix<T, 3, 1>::UnitY()) *
                             Eigen::AngleAxis<T>(rx, Eigen::Matrix<T, 3, 1>::UnitX()));

    element_ = Eigen::Quaternion<T>(R);
}

//--------------------------------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, 3, 3> SO3<T>::ToRotationMatrix() const {
    return element_.matrix();
}

//--------------------------------------------------------------------------------------------------
template <typename T>
Eigen::Quaternion<T> SO3<T>::ToQuaternion() const {
    return element_;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
SO3<T> SO3<T>::operator+(Eigen::Matrix<T, 3, 1> v) const {
    // TODO(pierreguilbert): replace exponential of skew matrix by rodrigues formula for performance
    Eigen::Matrix<T, 3, 3> skewV = VectorToSkew(v);
    SO3<T> sum;
    sum.SetElement(element_ * Eigen::Quaternion<T>(skewV.exp()));
    return sum;
}

//--------------------------------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, 3, 1> SO3<T>::operator-(SO3<T> y) const {
    Eigen::Matrix<T, 3, 3> log = (y.element_.inverse() * element_).matrix().log();
    return SkewToVector(log);
}

//--------------------------------------------------------------------------------------------------
template <typename T>
Eigen::Matrix<T, 3, 1> SO3<T>::ToAxisAngle() const {
    Eigen::AngleAxis<T> ax(element_);
    Eigen::Matrix<T, 3, 1> w = ax.angle() * ax.axis();
    return w;
}

//--------------------------------------------------------------------------------------------------
template <typename F, typename H>
BroadcastCameraEstimation<F, H>::BroadcastCameraEstimation(CalibrationStateManifoldElement element,
                                                           StateVariance variance_matrix) {
    element_ = element;
    variance_matrix_ = variance_matrix;
}

//--------------------------------------------------------------------------------------------------
template <typename F, typename H>
StateVariance BroadcastCameraEstimation<F, H>::GenerateMotionModelUncertainty(double dt) {
    // TODO(pierreguilbert): replace Monte-Carlo by analytic expression for performance
    // estimates it by using monte-carlo method
    StateVariance Q = StateVariance::Zero();

    const unsigned int n_sample = 250;
    for (unsigned int k = 0; k < n_sample; ++k) {
        // angular error
        Eigen::Vector3d angular_acceleration_error =
                max_angular_acceleration_ / std::sqrt(3) * Eigen::Vector3d::Random(3, 1);
        Eigen::Vector3d angular_velocity_error =
                angular_velocity_uncertainty_ * Eigen::Vector3d::Random(3, 1);
        Eigen::Vector3d angular_error =
                angular_velocity_error * dt + 0.5 * angular_acceleration_error * dt * dt;
        // position error
        Eigen::Vector3d position_error = position_uncertainty_ * Eigen::Vector3d::Random(3, 1);
        // focal error
        double focal_acceleration_error =
                max_focal_acceleration_ * Eigen::Vector2d::Random(2, 1)(0);
        double focal_velocity_error =
                focal_velocity_uncertainty_ * Eigen::Vector2d::Random(2, 1)(0);
        double focal_error = focal_velocity_error * dt + 0.5 * focal_acceleration_error * dt * dt;
        // distortion error
        double distortion_error = distortion_uncertainty_ * Eigen::Vector2d::Random(2, 1)(0);
        // principal point error
        Eigen::Vector2d principal_point_error =
                principal_point_uncertainty_ * Eigen::Vector2d::Random(2, 1);

        StateVector X_error;
        X_error.block(0, 0, 3, 1) = position_error;
        X_error.block(3, 0, 3, 1) = angular_error;
        X_error.block(6, 0, 3, 1) = angular_velocity_error + angular_acceleration_error * dt;
        X_error(9) = focal_error;
        X_error(10) = focal_velocity_error + focal_acceleration_error * dt;
        X_error.block(11, 0, 2, 1) = principal_point_error;
        X_error(13) = distortion_error;

        Q += X_error * X_error.transpose();
    }
    Q /= static_cast<double>(n_sample);
    return Q;
}

//--------------------------------------------------------------------------------------------------
template <typename F, typename H>
PredictionState BroadcastCameraEstimation<F, H>::Prediction(double dt) {
    // First add the prediction noise process
    StateVariance Q = GenerateMotionModelUncertainty(dt);
    StateVariance Incetitude = variance_matrix_ + Q;

    // Unscented Transform Prediction:
    // Generate the sigma points. It represents 2N+1 points of the manifold lying to the sigma-iso
    // probability surface with N being the dimension of the manifold
    std::vector<CalibrationStateManifoldElement> sigma_points =
            GenerateSigmaPoints<CalibrationStateManifoldElement, kStateDims>(element_, Incetitude);
    std::vector<StateVector> vect_bis =
            GenerateSigmaPoints<StateVector, kStateDims>(StateVector::Zero(), Incetitude);

    std::vector<StateVector> vect(sigma_points.size());
    for (unsigned int k = 0; k < sigma_points.size(); ++k) {
        vect[k] = sigma_points[k] - element_;

        // perform sanity check between:
        // - the displacement vector computed from the difference
        //   between the sigma points and the central element
        // - the displacement vectors corresponding to the sigma points
        //   generated from the variance covariance
        if ((vect[k] - vect_bis[k]).norm() > 1e-10) {
            std::wcerr << "UKF prediction sanity test failed. Boundary domain definition reached "
                          "this can be due to a too high variance-covariance matrix. This will "
                          "result in an underestimation of the internal variance";
        }
    }

    // apply the non-linear prediction and measurement equations to sigma points
    std::vector<CalibrationStateManifoldElement> prediction_sigma_points(sigma_points.size());
    std::vector<CalibrationMeasureManifoldElement> measurement_sigma_points(sigma_points.size());
    for (unsigned int k = 0; k < sigma_points.size(); ++k) {
        prediction_sigma_points[k] = prediction_(sigma_points[k], dt);
        measurement_sigma_points[k] = measure_(prediction_sigma_points[k]);
    }

    // from the transformed samples, compute the first two moment of the new distribution
    // Note that the mean is an element of the manifold and it should be computed using the
    // defined manifold Riemannian distance
    CalibrationStateManifoldElement mean_prediction =
            CalibrationStateManifoldElement::ComputeMean(prediction_sigma_points);
    CalibrationMeasureManifoldElement mean_measurement =
            CalibrationMeasureManifoldElement::ComputeMean(measurement_sigma_points);

    // Variance-Covariance is defined on the tangent vector space
    StateVariance variance_prediction = StateVariance::Zero();
    MeasVariance variance_measurement = MeasVariance::Zero();
    Eigen::Matrix<double, kStateDims, kMeasDims> cross_variance =
            Eigen::Matrix<double, kStateDims, kMeasDims>::Zero();
    for (unsigned int k = 0; k < sigma_points.size(); ++k) {
        StateVector V = prediction_sigma_points[k] - mean_prediction;
        variance_prediction += V * V.transpose();

        MeasVector U = measurement_sigma_points[k] - mean_measurement;
        variance_measurement += U * U.transpose();

        cross_variance += V * U.transpose();
    }
    variance_prediction /= static_cast<double>(prediction_sigma_points.size());
    variance_measurement /= static_cast<double>(measurement_sigma_points.size());
    cross_variance /= static_cast<double>(sigma_points.size());

    PredictionState prediction_ret;
    prediction_ret.predicted_element = mean_prediction;
    prediction_ret.prediction_variance = variance_prediction;
    prediction_ret.predicted_measure = mean_measurement;
    prediction_ret.measure_prediction_variance = variance_measurement;
    prediction_ret.cross_covariance = cross_variance;

    return prediction_ret;
}

//--------------------------------------------------------------------------------------------------
template <typename F, typename H>
void BroadcastCameraEstimation<F, H>::NewMeasure(double dt,
                                                 CalibrationMeasureManifoldElement measure,
                                                 MeasVariance measure_variance) {
    // first predict the next estimation and expected measures using the motion model
    PredictionState predicted_state = Prediction(dt);

    // Compute the gain of the kalman filter
    Eigen::Matrix<double, kStateDims, kMeasDims> K =
            predicted_state.cross_covariance *
            (predicted_state.measure_prediction_variance + measure_variance).inverse();

    // now update the kalman filter state by updating:
    // the estimation
    MeasVector error = measure - predicted_state.predicted_measure;  // manifold subtraction
    StateVector novelty = K * error;
    element_ = predicted_state.predicted_element + novelty;  // manifold addition

    // The variance-covariance
    variance_matrix_ =
            predicted_state.prediction_variance -
            K * (predicted_state.measure_prediction_variance + measure_variance) * K.transpose();
}

//--------------------------------------------------------------------------------------------------
template <unsigned int N>
Eigen::Matrix<double, N, N> MatrixSqrt(Eigen::Matrix<double, N, N> A) {
    // perform diagonalisation of A
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, N, N>> es;
    es.compute(A);

    Eigen::Matrix<double, N, N> sqrtEigVal = Eigen::Matrix<double, N, N>::Zero();
    for (unsigned int k = 0; k < N; ++k) {
        sqrtEigVal(k, k) = std::sqrt(es.eigenvalues()(k));
    }

    // Note that we use transpose instead of inverse because we expect A to be symmetric
    // using spectral theorem, we know that the eigen vectors basis is orthonormal
    return es.eigenvectors() * sqrtEigVal * es.eigenvectors().transpose();
}

//--------------------------------------------------------------------------------------------------
template <typename ManifoldElement, unsigned int N>
std::vector<ManifoldElement> GenerateSigmaPoints(ManifoldElement m, Eigen::Matrix<double, N, N> M) {
    // Get square root decomposition of M
    Eigen::Matrix<double, N, N> C = MatrixSqrt<N>(static_cast<double>(N) * M);
    // Fill sigma points
    std::vector<ManifoldElement> sigma_points(2 * N);
    for (unsigned int k = 0; k < N; ++k) {
        // Nothe that this is the addition define between the manifold and its tangent vector space
        sigma_points[2 * k] = m + C.col(k);
        sigma_points[2 * k + 1] = m + (-1.0 * C.col(k));
    }
    return sigma_points;
}

}  // namespace ukf

#endif  // ALGORITHMS_UKF_INCLUDE_UNSCENTED_KALMAN_FILTER_H_
