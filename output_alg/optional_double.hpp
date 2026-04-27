#pragma once

namespace output_alg {

// Lightweight C++14 replacement for std::optional<double>.
class OptionalDouble {
public:
    OptionalDouble() = default;
    OptionalDouble(double value) : has_value_(true), value_(value) {}

    bool has_value() const { return has_value_; }

    void reset() { has_value_ = false; }

    double& operator*() { return value_; }
    const double& operator*() const { return value_; }

private:
    bool has_value_ = false;
    double value_ = 0.0;
};

}  // namespace output_alg
