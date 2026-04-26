#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "one_euro_z.hpp"
#include "ukf.hpp"

namespace {

using Pose = output_alg::OneEuroZRealtimeFilter::Pose;

struct CsvTrajectory {
    std::vector<Pose> poses;
    std::vector<double> timestamps;
    bool has_timestamps = false;
};

struct Options {
    std::filesystem::path input;
    std::filesystem::path output;
    std::string algorithm = "ukf";
    output_alg::OneEuroZParameters one_euro;
    output_alg::UkfParameters ukf;
};

std::string Lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string Trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> Split(const std::string& line, char delimiter) {
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        parts.push_back(Trim(item));
    }
    if (!line.empty() && line.back() == delimiter) {
        parts.emplace_back();
    }
    return parts;
}

double ParseDouble(const std::string& value, const std::string& name) {
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid numeric value for " + name + ": " + value);
    }
}

std::array<double, 3> ParseVector3(const std::string& value, const std::string& name) {
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), ';', ',');
    const auto parts = Split(normalized, ',');
    if (parts.size() != 3) {
        throw std::invalid_argument(name + " must contain 3 comma-separated values");
    }
    return {
        ParseDouble(parts[0], name),
        ParseDouble(parts[1], name),
        ParseDouble(parts[2], name)};
}

std::array<double, 6> ParseVector6(const std::string& value, const std::string& name) {
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), ';', ',');
    const auto parts = Split(normalized, ',');
    if (parts.size() != 6) {
        throw std::invalid_argument(name + " must contain 6 comma-separated values");
    }
    return {
        ParseDouble(parts[0], name),
        ParseDouble(parts[1], name),
        ParseDouble(parts[2], name),
        ParseDouble(parts[3], name),
        ParseDouble(parts[4], name),
        ParseDouble(parts[5], name)};
}

void PrintUsage() {
    std::cerr
        << "usage: rt_filter_cpp_demo --input in.csv --output out.csv "
        << "--algorithm one_euro_z|ukf [params]\n\n"
        << "One Euro params:\n"
        << "  --min-cutoff 0.02 --beta 6.0 --d-cutoff 2.0 --derivative-deadband 1.0\n"
        << "UKF params:\n"
        << "  --motion-model constant_velocity --process-noise 1000 --measurement-noise 0.001\n"
        << "  --initial-linear-velocity vx,vy,vz --initial-angular-velocity wx,wy,wz\n";
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value after " + key);
            }
            return argv[++i];
        };

        if (key == "--help" || key == "-h") {
            PrintUsage();
            std::exit(0);
        } else if (key == "--input") {
            options.input = require_value();
        } else if (key == "--output") {
            options.output = require_value();
        } else if (key == "--algorithm") {
            options.algorithm = Lower(require_value());
        } else if (key == "--sample-rate-hz") {
            const double value = ParseDouble(require_value(), key);
            options.one_euro.sample_rate_hz = value;
            options.ukf.sample_rate_hz = value;
        } else if (key == "--strict-timestamps") {
            options.one_euro.strict_timestamps = true;
            options.ukf.strict_timestamps = true;
        } else if (key == "--min-cutoff") {
            options.one_euro.min_cutoff = ParseDouble(require_value(), key);
        } else if (key == "--beta") {
            options.one_euro.beta = ParseDouble(require_value(), key);
        } else if (key == "--d-cutoff") {
            options.one_euro.d_cutoff = ParseDouble(require_value(), key);
        } else if (key == "--derivative-deadband") {
            options.one_euro.derivative_deadband = ParseDouble(require_value(), key);
        } else if (key == "--motion-model") {
            options.ukf.motion_model = require_value();
        } else if (key == "--process-noise") {
            options.ukf.process_noise = ParseDouble(require_value(), key);
        } else if (key == "--measurement-noise") {
            options.ukf.measurement_noise = ParseDouble(require_value(), key);
        } else if (key == "--initial-covariance") {
            options.ukf.initial_covariance = ParseDouble(require_value(), key);
        } else if (key == "--alpha") {
            options.ukf.alpha = ParseDouble(require_value(), key);
        } else if (key == "--ukf-beta") {
            options.ukf.beta = ParseDouble(require_value(), key);
        } else if (key == "--kappa") {
            options.ukf.kappa = ParseDouble(require_value(), key);
        } else if (key == "--initial-velocity") {
            options.ukf.initial_velocity = ParseVector6(require_value(), key);
            options.ukf.use_initial_velocity = true;
        } else if (key == "--initial-linear-velocity") {
            options.ukf.initial_linear_velocity = ParseVector3(require_value(), key);
        } else if (key == "--initial-angular-velocity") {
            options.ukf.initial_angular_velocity = ParseVector3(require_value(), key);
        } else {
            throw std::invalid_argument("unknown option: " + key);
        }
    }

    if (options.input.empty()) {
        throw std::invalid_argument("--input is required");
    }
    if (options.output.empty()) {
        throw std::invalid_argument("--output is required");
    }
    if (options.algorithm != "one_euro_z" && options.algorithm != "ukf") {
        throw std::invalid_argument("--algorithm must be one_euro_z or ukf");
    }
    return options;
}

std::map<std::string, std::size_t> HeaderIndex(const std::vector<std::string>& header) {
    std::map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < header.size(); ++i) {
        index[Lower(header[i])] = i;
    }
    return index;
}

bool HasColumns(const std::map<std::string, std::size_t>& index, const std::vector<std::string>& columns) {
    return std::all_of(columns.begin(), columns.end(), [&](const std::string& col) {
        return index.find(col) != index.end();
    });
}

double CellDouble(
    const std::vector<std::string>& cells,
    const std::map<std::string, std::size_t>& index,
    const std::string& column) {
    const auto found = index.find(column);
    if (found == index.end() || found->second >= cells.size()) {
        throw std::invalid_argument("missing column: " + column);
    }
    return ParseDouble(cells[found->second], column);
}

Pose PoseFromMatrixColumns(
    const std::vector<std::string>& cells,
    const std::map<std::string, std::size_t>& index) {
    Pose pose{};
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            const std::string name = "m" + std::to_string(row) + std::to_string(col);
            pose[static_cast<std::size_t>(row * 4 + col)] = CellDouble(cells, index, name);
        }
    }
    return pose;
}

Pose PoseFromCompactColumns(
    const std::vector<std::string>& cells,
    const std::map<std::string, std::size_t>& index) {
    const double x = CellDouble(cells, index, "x");
    const double y = CellDouble(cells, index, "y");
    const double z = CellDouble(cells, index, "z");
    const double qw = CellDouble(cells, index, "qw");
    const double qx = CellDouble(cells, index, "qx");
    const double qy = CellDouble(cells, index, "qy");
    const double qz = CellDouble(cells, index, "qz");
    Eigen::Quaterniond quaternion(qw, qx, qy, qz);
    quaternion.normalize();
    const Eigen::Matrix3d rotation = quaternion.toRotationMatrix();

    Pose pose = {
        rotation(0, 0), rotation(0, 1), rotation(0, 2), x,
        rotation(1, 0), rotation(1, 1), rotation(1, 2), y,
        rotation(2, 0), rotation(2, 1), rotation(2, 2), z,
        0.0, 0.0, 0.0, 1.0};
    return pose;
}

CsvTrajectory ReadCsvTrajectory(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open input: " + path.string());
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("input CSV is empty: " + path.string());
    }
    const auto header = Split(line, ',');
    const auto index = HeaderIndex(header);

    std::vector<std::string> matrix_columns;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            matrix_columns.push_back("m" + std::to_string(row) + std::to_string(col));
        }
    }
    const bool has_matrix = HasColumns(index, matrix_columns);
    const bool has_compact = HasColumns(index, {"x", "y", "z", "qw", "qx", "qy", "qz"});
    if (!has_matrix && !has_compact) {
        throw std::invalid_argument("CSV must contain m00..m33 or x,y,z,qw,qx,qy,qz columns");
    }

    CsvTrajectory trajectory;
    trajectory.has_timestamps =
        index.find("timestamp") != index.end() || index.find("time") != index.end() || index.find("t") != index.end();
    std::string timestamp_column;
    for (const auto& candidate : {"timestamp", "time", "t"}) {
        if (index.find(candidate) != index.end()) {
            timestamp_column = candidate;
            break;
        }
    }

    while (std::getline(input, line)) {
        if (Trim(line).empty()) {
            continue;
        }
        const auto cells = Split(line, ',');
        trajectory.poses.push_back(
            has_matrix ? PoseFromMatrixColumns(cells, index) : PoseFromCompactColumns(cells, index));
        if (!timestamp_column.empty()) {
            trajectory.timestamps.push_back(CellDouble(cells, index, timestamp_column));
        }
    }
    if (trajectory.poses.empty()) {
        throw std::runtime_error("input CSV contains no trajectory rows");
    }
    return trajectory;
}

void WriteCsvTrajectory(
    const std::filesystem::path& path,
    const std::vector<Pose>& poses,
    const std::vector<double>& timestamps,
    bool has_timestamps) {
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }

    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open output: " + path.string());
    }
    output << std::setprecision(17);
    if (has_timestamps) {
        output << "timestamp,";
    }
    output << "x,y,z,qw,qx,qy,qz";
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            output << ",m" << row << col;
        }
    }
    output << '\n';

    for (std::size_t i = 0; i < poses.size(); ++i) {
        const Pose& pose = poses[i];
        if (has_timestamps) {
            output << timestamps[i] << ',';
        }

        Eigen::Matrix3d rotation;
        rotation << pose[0], pose[1], pose[2],
            pose[4], pose[5], pose[6],
            pose[8], pose[9], pose[10];
        Eigen::Quaterniond quaternion(rotation);
        quaternion.normalize();
        if (quaternion.w() < 0.0) {
            quaternion.coeffs() *= -1.0;
        }

        output << pose[3] << ',' << pose[7] << ',' << pose[11] << ','
               << quaternion.w() << ',' << quaternion.x() << ','
               << quaternion.y() << ',' << quaternion.z();
        for (double value : pose) {
            output << ',' << value;
        }
        output << '\n';
    }
}

std::vector<Pose> RunFilter(const Options& options, const CsvTrajectory& trajectory) {
    const std::vector<double>* timestamps =
        trajectory.has_timestamps ? &trajectory.timestamps : nullptr;
    if (options.algorithm == "one_euro_z") {
        return output_alg::FilterOneEuroZTrajectory(
            trajectory.poses,
            timestamps,
            options.one_euro);
    }
    return output_alg::FilterUkfTrajectory(
        trajectory.poses,
        timestamps,
        options.ukf);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        const CsvTrajectory input = ReadCsvTrajectory(options.input);
        const std::vector<Pose> filtered = RunFilter(options, input);
        WriteCsvTrajectory(options.output, filtered, input.timestamps, input.has_timestamps);
        std::cout << "wrote " << options.output.string() << " frames=" << filtered.size() << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        PrintUsage();
        return 2;
    }
}
