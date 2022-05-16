#include "unscented_kalman_filter.h"

namespace ukf {

//--------------------------------------------------------------------------------------------------
CalibrationStateManifoldElement::CalibrationStateManifoldElement(
        Eigen::Vector3d position, SO3<double> orientation, Eigen::Vector3d angular_velocity,
        double focal, double focal_velocity, Eigen::Vector2d principal_point, double distortion) {
    position_ = position;
    orientation_ = orientation;
    angular_velocity_ = angular_velocity;
    focal_ = focal;
    focal_velocity_ = focal_velocity;
    principal_point_ = principal_point;
    distortion_ = distortion;
}

//--------------------------------------------------------------------------------------------------
CalibrationStateManifoldElement CalibrationStateManifoldElement::operator+(StateVector v) {
    Eigen::Vector3d position = position_ + v.block(0, 0, 3, 1);
    SO3<double> orientation = orientation_ + v.block(3, 0, 3, 1);  // manifold addition
    Eigen::Vector3d angular_velocity = angular_velocity_ + v.block(6, 0, 3, 1);
    double focal = focal_ + v(9);
    double focal_velocity = focal_velocity_ + v(10);
    Eigen::Vector2d principal_point = principal_point_ + v.block(11, 0, 2, 1);
    double distortion = distortion_ + v(13);

    return CalibrationStateManifoldElement(position, orientation, angular_velocity, focal,
                                           focal_velocity, principal_point, distortion);
}

//--------------------------------------------------------------------------------------------------
StateVector CalibrationStateManifoldElement::operator-(CalibrationStateManifoldElement y) {
    StateVector v;
    v.block(0, 0, 3, 1) = position_ - y.position_;
    v.block(3, 0, 3, 1) = orientation_ - y.orientation_;  // manifold substraction
    v.block(6, 0, 3, 1) = angular_velocity_ - y.angular_velocity_;
    v(9) = focal_ - y.focal_;
    v(10) = focal_velocity_ - y.focal_velocity_;
    v.block(11, 0, 2, 1) = principal_point_ - y.principal_point_;
    v(13) = distortion_ - y.distortion_;

    return v;
}

//--------------------------------------------------------------------------------------------------
std::string CalibrationStateManifoldElement::PrintSelf() {
    std::stringstream ss;
    ss << "Position: " << position_.transpose() << std::endl;
    ss << "Orientation: " << std::endl << orientation_.ToQuaternion().matrix() << std::endl;
    ss << "angular velocity: " << angular_velocity_.transpose() << std::endl;
    ss << "focal: " << focal_ << std::endl;
    ss << "focal velocity: " << focal_velocity_ << std::endl;
    ss << "principal point: " << principal_point_.transpose() << std::endl;
    ss << "distortion: " << distortion_;

    return ss.str();
}

//--------------------------------------------------------------------------------------------------
CalibrationStateManifoldElement CalibrationStateManifoldElement::ComputeMean(
        std::vector<CalibrationStateManifoldElement> samples) {
    CalibrationStateManifoldElement mean;

    // all parameters lying in a vector space are handles as usual using the distance
    // derived from the canonical dot product
    mean.focal_ = 0;
    mean.position_ = Eigen::Vector3d::Zero();
    mean.distortion_ = 0;
    mean.focal_velocity_ = 0;
    mean.principal_point_ = Eigen::Vector2d::Zero();
    mean.angular_velocity_ = Eigen::Vector3d::Zero();
    for (unsigned int k = 0; k < samples.size(); ++k) {
        mean.focal_ += samples[k].focal_;
        mean.position_ += samples[k].position_;
        mean.distortion_ += samples[k].distortion_;
        mean.focal_velocity_ += samples[k].focal_velocity_;
        mean.principal_point_ += samples[k].principal_point_;
        mean.angular_velocity_ += samples[k].angular_velocity_;
    }
    mean.focal_ /= static_cast<double>(samples.size());
    mean.position_ /= static_cast<double>(samples.size());
    mean.distortion_ /= static_cast<double>(samples.size());
    mean.focal_velocity_ /= static_cast<double>(samples.size());
    mean.principal_point_ /= static_cast<double>(samples.size());
    mean.angular_velocity_ /= static_cast<double>(samples.size());

    // for the orientation part, we will compute the mean using the
    // angle distance metric
    Eigen::Quaterniond mean_quat(1, 0, 0, 0);
    int count = 0;
    while (count < 15) {
        // compute the mean (relative to quaternion vector space) of the relative rotation
        // between the current mean (relative to unitary quaternion lie algebra) estimation
        // and the samples
        Eigen::Vector3d mean_err_vector(0, 0, 0);
        double sum = 0;
        for (unsigned int k = 0; k < samples.size(); ++k) {
            Eigen::Quaterniond error = samples[k].orientation_.ToQuaternion() * mean_quat.inverse();
            Eigen::AngleAxisd ax(error);
            Eigen::Vector3d w = ax.angle() * ax.axis();
            mean_err_vector += w;
        }
        mean_err_vector /= static_cast<double>(samples.size());

        Eigen::AngleAxisd ax;
        ax.axis() = mean_err_vector.normalized();
        ax.angle() = mean_err_vector.norm();
        mean_quat = Eigen::Quaterniond(ax) * mean_quat;
        count++;
    }

    mean.orientation_ = SO3<double>(mean_quat);
    return mean;
}

//--------------------------------------------------------------------------------------------------
CalibrationMeasureManifoldElement::CalibrationMeasureManifoldElement(
        Eigen::Vector3d position, SO3<double> orientation, double focal,
        Eigen::Vector2d principal_point, double distortion) {
    position_ = position;
    orientation_ = orientation;
    focal_ = focal;
    principal_point_ = principal_point;
    distortion_ = distortion;
}

//--------------------------------------------------------------------------------------------------
CalibrationMeasureManifoldElement CalibrationMeasureManifoldElement::operator+(MeasVector v) {
    Eigen::Vector3d position = position_ + v.block(0, 0, 3, 1);
    SO3<double> orientation = orientation_ + v.block(3, 0, 3, 1);  // manifold addition
    double focal = focal_ + v(6);
    Eigen::Vector2d principal_point = principal_point_ + v.block(7, 0, 2, 1);
    double distortion = distortion_ + v(9);

    return CalibrationMeasureManifoldElement(position, orientation, focal, principal_point,
                                             distortion);
}

//--------------------------------------------------------------------------------------------------
MeasVector CalibrationMeasureManifoldElement::operator-(CalibrationMeasureManifoldElement y) {
    MeasVector v;
    v.block(0, 0, 3, 1) = position_ - y.position_;
    v.block(3, 0, 3, 1) = orientation_ - y.orientation_;  // manifold substraction
    v(6) = focal_ - y.focal_;
    v.block(7, 0, 2, 1) = principal_point_ - y.principal_point_;
    v(9) = distortion_ - y.distortion_;

    return v;
}

//--------------------------------------------------------------------------------------------------
std::string CalibrationMeasureManifoldElement::PrintSelf() {
    std::stringstream ss;
    ss << "Position: " << position_.transpose() << std::endl;
    ss << "Orientation: " << std::endl << orientation_.ToQuaternion().matrix() << std::endl;
    ss << "focal: " << focal_ << std::endl;
    ss << "principal point: " << principal_point_.transpose() << std::endl;
    ss << "distortion: " << distortion_;

    return ss.str();
}

//--------------------------------------------------------------------------------------------------
CalibrationMeasureManifoldElement CalibrationMeasureManifoldElement::ComputeMean(
        std::vector<CalibrationMeasureManifoldElement> samples) {
    CalibrationMeasureManifoldElement mean;

    // all parameters lying in a vector space are handles as usual using the distance
    // derived from the canonical dot product
    mean.focal_ = 0;
    mean.position_ = Eigen::Vector3d::Zero();
    mean.distortion_ = 0;
    mean.principal_point_ = Eigen::Vector2d::Zero();
    for (unsigned int k = 0; k < samples.size(); ++k) {
        mean.focal_ += samples[k].focal_;
        mean.position_ += samples[k].position_;
        mean.distortion_ += samples[k].distortion_;
        mean.principal_point_ += samples[k].principal_point_;
    }
    mean.focal_ /= static_cast<double>(samples.size());
    mean.position_ /= static_cast<double>(samples.size());
    mean.distortion_ /= static_cast<double>(samples.size());
    mean.principal_point_ /= static_cast<double>(samples.size());

    // for the orientation part, we will compute the mean using the
    // angle distance metric
    Eigen::Quaterniond mean_quat(1, 0, 0, 0);
    int count = 0;
    while (count < 15) {
        // compute the mean (relative to quaternion vector space) of the relative rotation
        // between the current mean (relative to unitary quaternion lie algebra) estimation
        // and the samples
        Eigen::Vector3d mean_err_vector(0, 0, 0);
        double sum = 0;
        for (unsigned int k = 0; k < samples.size(); ++k) {
            Eigen::Quaterniond error = samples[k].orientation_.ToQuaternion() * mean_quat.inverse();
            Eigen::AngleAxisd ax(error);
            Eigen::Vector3d w = ax.angle() * ax.axis();
            mean_err_vector += w;
        }
        mean_err_vector /= static_cast<double>(samples.size());

        Eigen::AngleAxisd ax;
        ax.axis() = mean_err_vector.normalized();
        ax.angle() = mean_err_vector.norm();
        mean_quat = Eigen::Quaterniond(ax) * mean_quat;
        count++;
    }

    mean.orientation_ = SO3<double>(mean_quat);
    return mean;
}
CalibrationStateManifoldElement FixedPositionConstantAngularVelocityPrediction::operator()(
        CalibrationStateManifoldElement X, double dt) {
    CalibrationStateManifoldElement X_dt;

    // position
    X_dt.position_ = X.position_;  // no motion regarding position
    // angular velocity
    X_dt.angular_velocity_ = X.angular_velocity_;  // constant angular velocity assumption
    // distortion
    X_dt.distortion_ = X.distortion_;
    // focal velocity
    X_dt.focal_velocity_ = X.focal_velocity_;  // constant angular velocity assumption
    // principal point
    X_dt.principal_point_ = X.principal_point_;
    // focal
    X_dt.focal_ = X.focal_ + X.focal_velocity_ * dt;
    // orientation
    X_dt.orientation_ = X.orientation_ + (X.angular_velocity_ * dt);  // Manifold addition

    return X_dt;
}

CalibrationMeasureManifoldElement BundleAdjustmentMeasurement::operator()(
        CalibrationStateManifoldElement X) {
    CalibrationMeasureManifoldElement Y;

    Y.position_ = X.position_;
    Y.orientation_ = X.orientation_;
    Y.focal_ = X.focal_;
    Y.principal_point_ = X.principal_point_;
    Y.distortion_ = X.distortion_;

    return Y;
}

}  // namespace ukf
