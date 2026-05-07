#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include <Eigen/Geometry>

#include "adaptive_local_line.hpp"
#include "butterworth.hpp"
#include "one_euro_z.hpp"
#include "ukf.hpp"

namespace {

// 两个 C++ 滤波器统一使用 RigidMatrix 表示刚体变换。
// demo 中先把 CSV 或“坐标+欧拉角”转换成 RigidMatrix，再调用实际滤波器。
using RigidMatrix = Sn3DAlgorithm::RigidMatrix;

struct CsvTrajectory {
    std::vector<RigidMatrix> rigids;
    std::vector<double> timestamps;
    // 输入 CSV 可能没有时间戳；没有时间戳时滤波器会使用 sample_rate_hz 计算固定 dt。
    bool has_timestamps = false;
};

struct Options {
    std::string input;
    std::string output;
    std::string algorithm = "ukf";
    // 滤波参数统一放在 lambda / 单帧接口外部配置。实际工程中可把这些成员
    // 放到扫描界面类或轨迹处理类里，避免每帧重复构造参数对象。
    output_alg::ButterworthParameters butterworth;
    output_alg::OneEuroZParameters one_euro;
    output_alg::UkfParameters ukf;
    output_alg::AdaptiveLocalLineParameters adaptive_local_line;
};

std::string DefaultOutputForAlgorithm(const std::string& algorithm) {
    if (algorithm == "butterworth") {
        return "outputs/cpp_demo/noisy_line_butterworth.csv";
    }
    if (algorithm == "butterworth_z") {
        return "outputs/cpp_demo/noisy_line_butterworth_z.csv";
    }
    if (algorithm == "one_euro_z") {
        return "outputs/cpp_demo/noisy_line_one_euro_z.csv";
    }
    if (algorithm == "adaptive_local_line") {
        return "outputs/cpp_demo/noisy_line_adaptive_local_line.csv";
    }
    return "outputs/cpp_demo/noisy_line_ukf.csv";
}

Options DefaultOptions() {
    Options options;

    // 默认输入输出用于“无命令行参数”直接运行 demo。输入文件来自当前示例数据；
    // 输出文件仍写到 outputs/cpp_demo，便于继续用 Python 框架评估。
    options.input = "examples/demo_data/noisy_line.csv";
    options.algorithm = "ukf";
    options.output = DefaultOutputForAlgorithm(options.algorithm);

    // Realtime Butterworth 当前推荐参数：面向 100 Hz 采样下的轻量实时低通。
    // 这是因果 IIR，和 Python 侧离线 zero-phase Butterworth 并不等价。
    options.butterworth.cutoff_hz = 20.0;
    options.butterworth.order = 2;
    options.butterworth.sample_rate_hz = 100.0;
    options.butterworth.history_size = 0;
    options.butterworth.delay_frames = 0;
    options.butterworth.strict_timestamps = false;

    // One Euro Z 当前推荐参数：偏向 SN 数据中 Z 方向静止降噪，同时保留速度自适应。
    options.one_euro.min_cutoff = 1.0;
    options.one_euro.beta = 10.0;
    options.one_euro.d_cutoff = 8.0;
    options.one_euro.derivative_deadband = 0.02;
    options.one_euro.sample_rate_hz = 100.0;
    options.one_euro.history_size = 0;
    options.one_euro.delay_frames = 0;
    options.one_euro.strict_timestamps = false;

    // UKF 当前推荐参数：匀速模型，过程噪声较大以保证跟手，观测噪声较小以避免
    // 前期出现明显偏移；初始线速度/角速度默认未知，设为 0。
    options.ukf.motion_model = "constant_velocity";
    options.ukf.process_noise = 1000.0;
    options.ukf.measurement_noise = 0.001;
    options.ukf.initial_covariance = 1.0;
    options.ukf.use_initial_velocity = false;
    options.ukf.initial_velocity = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    options.ukf.initial_linear_velocity = {0.0, 0.0, 0.0};
    options.ukf.initial_angular_velocity = {0.0, 0.0, 0.0};
    options.ukf.alpha = 1e-3;
    options.ukf.beta = 2.0;
    options.ukf.kappa = 0.0;
    options.ukf.sample_rate_hz = 100.0;
    options.ukf.history_size = 0;
    options.ukf.strict_timestamps = false;

    // Adaptive Local Line：5 帧窗口、2 帧固定延迟。沿直线方向保持原始坐标，
    // 只按局部垂直跳动自适应削弱线外残差。默认参数按 0507_sn/scan 的 mm 量级整定。
    options.adaptive_local_line.window = 5;
    options.adaptive_local_line.target_noise_mm = 0.26;
    options.adaptive_local_line.max_strength = 0.5;
    options.adaptive_local_line.min_strength = 0.0;
    options.adaptive_local_line.response = 1.0;
    options.adaptive_local_line.reference_mode = "global";
    options.adaptive_local_line.sample_rate_hz = 100.0;
    options.adaptive_local_line.strict_timestamps = false;
    options.adaptive_local_line.history_size = 0;

    return options;
}

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

bool IsPathSeparator(char ch) {
    return ch == '/' || ch == '\\';
}

std::string ParentPath(const std::string& path) {
    const std::size_t end = path.find_last_not_of("/\\");
    if (end == std::string::npos) {
        return "";
    }

    const std::size_t separator = path.find_last_of("/\\", end);
    if (separator == std::string::npos) {
        return "";
    }
    if (separator == 0) {
        return path.substr(0, 1);
    }
    if (separator == 2 && path.size() >= 3 && path[1] == ':' && IsPathSeparator(path[2])) {
        return path.substr(0, 3);
    }
    return path.substr(0, separator);
}

std::string ReplaceExtensionWithSuffix(const std::string& path, const std::string& suffix) {
    const std::size_t separator = path.find_last_of("/\\");
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || (separator != std::string::npos && dot < separator)) {
        return path + suffix;
    }
    return path.substr(0, dot) + suffix;
}

void MakeDirectory(const std::string& directory) {
    if (directory.empty() || directory == "." ||
        (directory.size() == 2 && directory[1] == ':')) {
        return;
    }

    errno = 0;
#ifdef _WIN32
    const int result = _mkdir(directory.c_str());
#else
    const int result = mkdir(directory.c_str(), 0755);
#endif
    if (result != 0 && errno != EEXIST) {
        throw std::runtime_error("failed to create directory: " + directory);
    }
}

void CreateDirectories(const std::string& directory) {
    std::string normalized = directory;
    while (!normalized.empty() && IsPathSeparator(normalized[normalized.size() - 1])) {
        normalized.erase(normalized.size() - 1);
    }
    if (normalized.empty()) {
        return;
    }

    std::string current;
    std::size_t start = 0;
    if (normalized.size() >= 2 && normalized[1] == ':') {
        current = normalized.substr(0, 2);
        start = 2;
        if (start < normalized.size() && IsPathSeparator(normalized[start])) {
            current += normalized[start];
            ++start;
        }
    } else if (IsPathSeparator(normalized[0])) {
        current = normalized.substr(0, 1);
        start = 1;
    }

    while (start <= normalized.size()) {
        const std::size_t separator = normalized.find_first_of("/\\", start);
        const std::size_t count =
            separator == std::string::npos ? std::string::npos : separator - start;
        const std::string part = normalized.substr(start, count);
        if (!part.empty() && part != ".") {
            if (!current.empty() && !IsPathSeparator(current[current.size() - 1])) {
                current += '/';
            }
            current += part;
            MakeDirectory(current);
        }
        if (separator == std::string::npos) {
            break;
        }
        start = separator + 1;
    }
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

int ParseInt(const std::string& value, const std::string& name) {
    try {
        std::size_t consumed = 0;
        const long parsed = std::stol(value, &consumed, 10);
        if (consumed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<int>(parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid integer value for " + name + ": " + value);
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
        << "--algorithm butterworth|butterworth_z|one_euro_z|adaptive_local_line|ukf [params]\n\n"
        << "Defaults when no arguments are supplied:\n"
        << "  --input examples/demo_data/noisy_line.csv\n"
        << "  --algorithm ukf\n"
        << "  --output outputs/cpp_demo/noisy_line_ukf.csv\n\n"
        << "Butterworth realtime params:\n"
        << "  --cutoff-hz 20.0 --order 2\n"
        << "  butterworth_z also supports --delay-frames 0\n"
        << "One Euro params:\n"
        << "  --min-cutoff 1.0 --beta 10.0 --d-cutoff 8.0 --derivative-deadband 0.02 --delay-frames 0\n"
        << "UKF params:\n"
        << "  --motion-model constant_velocity --process-noise 1000 --measurement-noise 0.001\n"
        << "  --initial-linear-velocity vx,vy,vz --initial-angular-velocity wx,wy,wz\n"
        << "Adaptive Local Line params:\n"
        << "  --window 5 --target-noise-mm 0.26 --max-strength 0.5 --reference-mode global\n"
        << "  --line-origin x,y,z --line-direction dx,dy,dz\n";
}

Options ParseOptions(int argc, char** argv) {
    Options options = DefaultOptions();
    bool output_provided = false;
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
            output_provided = true;
        } else if (key == "--algorithm") {
            options.algorithm = Lower(require_value());
        } else if (key == "--sample-rate-hz") {
            // 三个实时滤波器都支持“无时间戳输入”。此时使用采样率推导 dt。
            const double value = ParseDouble(require_value(), key);
            options.butterworth.sample_rate_hz = value;
            options.one_euro.sample_rate_hz = value;
            options.ukf.sample_rate_hz = value;
            options.adaptive_local_line.sample_rate_hz = value;
        } else if (key == "--strict-timestamps") {
            // 默认情况下，异常时间戳会退回到固定采样率；开启 strict 后会直接报错。
            options.butterworth.strict_timestamps = true;
            options.one_euro.strict_timestamps = true;
            options.ukf.strict_timestamps = true;
            options.adaptive_local_line.strict_timestamps = true;
        } else if (key == "--delay-frames") {
            const int value = ParseInt(require_value(), key);
            if (value < 0) {
                throw std::invalid_argument("--delay-frames must be >= 0");
            }
            options.butterworth.delay_frames = static_cast<std::size_t>(value);
            options.one_euro.delay_frames = static_cast<std::size_t>(value);
        } else if (key == "--cutoff-hz") {
            // Realtime Butterworth：因果低通截止频率，越低越平滑，但真实局部波峰也会被更强压制。
            options.butterworth.cutoff_hz = ParseDouble(require_value(), key);
        } else if (key == "--order") {
            // Realtime Butterworth：Butterworth 阶数；当前建议先从 2 阶开始。
            options.butterworth.order = ParseInt(require_value(), key);
        } else if (key == "--min-cutoff") {
            // One Euro Z：静止/低速时的最小截止频率，越小越稳但拖影越明显。
            options.one_euro.min_cutoff = ParseDouble(require_value(), key);
        } else if (key == "--beta") {
            // One Euro Z：速度越大时放开低通的强度，越大越跟手。
            options.one_euro.beta = ParseDouble(require_value(), key);
        } else if (key == "--d-cutoff") {
            // One Euro Z：Z 方向速度估计本身的低通截止频率。
            options.one_euro.d_cutoff = ParseDouble(require_value(), key);
        } else if (key == "--derivative-deadband") {
            // One Euro Z：Z 速度死区，用于避免静止噪声触发“运动中”参数。
            options.one_euro.derivative_deadband = ParseDouble(require_value(), key);
        } else if (key == "--motion-model") {
            // UKF：当前支持匀速和匀加速两种通用运动模型。
            options.ukf.motion_model = require_value();
        } else if (key == "--process-noise") {
            // UKF：过程噪声越大，越允许运动状态快速变化，滤波越跟手。
            options.ukf.process_noise = ParseDouble(require_value(), key);
        } else if (key == "--measurement-noise") {
            // UKF：观测噪声越大，越不相信输入位姿，平滑更强但可能更滞后。
            options.ukf.measurement_noise = ParseDouble(require_value(), key);
        } else if (key == "--initial-covariance") {
            // UKF：初始状态协方差，影响前几帧对初始状态的信任程度。
            options.ukf.initial_covariance = ParseDouble(require_value(), key);
        } else if (key == "--alpha") {
            options.ukf.alpha = ParseDouble(require_value(), key);
        } else if (key == "--ukf-beta") {
            options.ukf.beta = ParseDouble(require_value(), key);
        } else if (key == "--kappa") {
            options.ukf.kappa = ParseDouble(require_value(), key);
        } else if (key == "--initial-velocity") {
            // UKF：整体 6 维初始速度 [vx,vy,vz,wx,wy,wz]。
            // 线速度单位与输入坐标一致/秒，角速度单位为 rad/s。
            options.ukf.initial_velocity = ParseVector6(require_value(), key);
            options.ukf.use_initial_velocity = true;
        } else if (key == "--initial-linear-velocity") {
            // UKF：拆分形式的初始线速度 [vx,vy,vz]。
            options.ukf.initial_linear_velocity = ParseVector3(require_value(), key);
        } else if (key == "--initial-angular-velocity") {
            // UKF：拆分形式的初始角速度 [wx,wy,wz]，单位 rad/s。
            options.ukf.initial_angular_velocity = ParseVector3(require_value(), key);
        } else if (key == "--window") {
            const int value = ParseInt(require_value(), key);
            if (value < 0) {
                throw std::invalid_argument("--window must be positive");
            }
            options.adaptive_local_line.window = static_cast<std::size_t>(value);
        } else if (key == "--target-noise-mm") {
            options.adaptive_local_line.target_noise_mm = ParseDouble(require_value(), key);
        } else if (key == "--max-strength") {
            options.adaptive_local_line.max_strength = ParseDouble(require_value(), key);
        } else if (key == "--min-strength") {
            options.adaptive_local_line.min_strength = ParseDouble(require_value(), key);
        } else if (key == "--response") {
            options.adaptive_local_line.response = ParseDouble(require_value(), key);
        } else if (key == "--reference-mode") {
            options.adaptive_local_line.reference_mode = Lower(require_value());
        } else if (key == "--line-origin") {
            const auto value = ParseVector3(require_value(), key);
            options.adaptive_local_line.line_origin =
                Eigen::Vector3d(value[0], value[1], value[2]);
            options.adaptive_local_line.use_line_origin = true;
        } else if (key == "--line-direction") {
            const auto value = ParseVector3(require_value(), key);
            options.adaptive_local_line.line_direction =
                Eigen::Vector3d(value[0], value[1], value[2]);
            options.adaptive_local_line.use_line_direction = true;
        } else {
            throw std::invalid_argument("unknown option: " + key);
        }
    }

    if (
        options.algorithm != "butterworth" &&
        options.algorithm != "butterworth_z" &&
        options.algorithm != "one_euro_z" &&
        options.algorithm != "adaptive_local_line" &&
        options.algorithm != "ukf") {
        throw std::invalid_argument(
            "--algorithm must be butterworth, butterworth_z, one_euro_z, adaptive_local_line, or ukf");
    }
    if (!output_provided) {
        options.output = DefaultOutputForAlgorithm(options.algorithm);
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

RigidMatrix RigidFromMatrixColumns(
    const std::vector<std::string>& cells,
    const std::map<std::string, std::size_t>& index) {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            const std::string name = "m" + std::to_string(row) + std::to_string(col);
            const double value = CellDouble(cells, index, name);
            if (row < 3 && col < 3) {
                rotation(row, col) = value;
            } else if (row < 3 && col == 3) {
                translation(row) = value;
            }
        }
    }
    return RigidMatrix(rotation, translation);
}

RigidMatrix RigidFromCompactColumns(
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
    return RigidMatrix(rotation, Eigen::Vector3d(x, y, z));
}

CsvTrajectory ReadCsvTrajectory(const std::string& path) {
    std::ifstream input(path.c_str());
    if (!input) {
        throw std::runtime_error("failed to open input: " + path);
    }

    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("input CSV is empty: " + path);
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
        trajectory.rigids.push_back(
            has_matrix ? RigidFromMatrixColumns(cells, index) : RigidFromCompactColumns(cells, index));
        if (!timestamp_column.empty()) {
            trajectory.timestamps.push_back(CellDouble(cells, index, timestamp_column));
        }
    }
    if (trajectory.rigids.empty()) {
        throw std::runtime_error("input CSV contains no trajectory rows");
    }
    return trajectory;
}

std::string JsonEscape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
        case '\\':
            escaped += "\\\\";
            break;
        case '"':
            escaped += "\\\"";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped += ch;
            break;
        }
    }
    return escaped;
}

void WriteCsvTrajectory(
    const std::string& path,
    const std::vector<RigidMatrix>& rigids,
    const std::vector<double>& timestamps,
    bool has_timestamps) {
    const std::string parent = ParentPath(path);
    if (!parent.empty()) {
        CreateDirectories(parent);
    }

    std::ofstream output(path.c_str());
    if (!output) {
        throw std::runtime_error("failed to open output: " + path);
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

    for (std::size_t i = 0; i < rigids.size(); ++i) {
        const RigidMatrix& rigid = rigids[i];
        if (has_timestamps) {
            output << timestamps[i] << ',';
        }

        const Eigen::Matrix3d& rotation = rigid.get_rotation();
        const Eigen::Vector3d& translation = rigid.get_translation();
        Eigen::Quaterniond quaternion(rotation);
        quaternion.normalize();
        if (quaternion.w() < 0.0) {
            quaternion.coeffs() *= -1.0;
        }

        output << translation.x() << ',' << translation.y() << ',' << translation.z() << ','
               << quaternion.w() << ',' << quaternion.x() << ','
               << quaternion.y() << ',' << quaternion.z();
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                double value = 0.0;
                if (row < 3 && col < 3) {
                    value = rotation(row, col);
                } else if (row < 3 && col == 3) {
                    value = translation(row);
                } else if (row == 3 && col == 3) {
                    value = 1.0;
                }
                output << ',' << value;
            }
        }
        output << '\n';
    }
}

void WriteTimingCsv(
    const std::string& path,
    const std::vector<std::int64_t>& per_pose_time_ns,
    const std::vector<double>& timestamps,
    bool has_timestamps) {
    const std::string parent = ParentPath(path);
    if (!parent.empty()) {
        CreateDirectories(parent);
    }

    std::ofstream output(path.c_str());
    if (!output) {
        throw std::runtime_error("failed to open output: " + path);
    }
    output << std::setprecision(17);
    output << "frame_index,timestamp,compute_time_ns,compute_time_us,compute_time_ms\n";
    for (std::size_t i = 0; i < per_pose_time_ns.size(); ++i) {
        const double elapsed_ns = static_cast<double>(per_pose_time_ns[i]);
        output << i << ',';
        if (has_timestamps) {
            output << timestamps[i];
        }
        output << ',' << per_pose_time_ns[i] << ','
               << elapsed_ns / 1000.0 << ',' << elapsed_ns / 1000000.0 << '\n';
    }
}

void WriteMetricsJson(
    const std::string& path,
    const Options& options,
    const CsvTrajectory& input,
    const output_alg::TimedFilterTrajectoryResult& result,
    const std::string& timing_path) {
    const std::string parent = ParentPath(path);
    if (!parent.empty()) {
        CreateDirectories(parent);
    }

    const output_alg::TimingSummary timing =
        output_alg::SummarizeTiming(result.per_pose_time_ns, result.total_time_ns);
    std::ofstream output(path.c_str());
    if (!output) {
        throw std::runtime_error("failed to open output: " + path);
    }
    output << std::setprecision(17);
    output << "{\n"
           << "  \"algorithm\": \"" << JsonEscape(options.algorithm) << "\",\n"
           << "  \"input_path\": \"" << JsonEscape(options.input) << "\",\n"
           << "  \"output_path\": \"" << JsonEscape(options.output) << "\",\n"
           << "  \"timing_path\": \"" << JsonEscape(timing_path) << "\",\n"
           << "  \"frames\": " << result.rigids.size() << ",\n"
           << "  \"has_timestamps\": " << (input.has_timestamps ? "true" : "false") << ",\n"
           << "  \"compute_total_ns\": " << timing.total_time_ns << ",\n"
           << "  \"compute_total_ms\": " << timing.total_time_ms << ",\n"
           << "  \"compute_mean_us\": " << timing.mean_time_us << ",\n"
           << "  \"compute_p95_us\": " << timing.p95_time_us << ",\n"
           << "  \"compute_max_us\": " << timing.max_time_us << "\n"
           << "}\n";
}

output_alg::TimedFilterTrajectoryResult RunRealtimeFrameFilterTimed(
    const Options& options,
    const CsvTrajectory& trajectory) {
    // 实时帧滤波入口：这里用 CSV 中的每一行模拟实时系统收到的一帧 RigidMatrix。
    // 关键点是滤波器对象只创建一次，并在循环中跨帧复用。这样 Butterworth 的 IIR
    // 节状态、One Euro 的上一帧输出、UKF 的状态向量/协方差/参考姿态都会被保留下来。
    const std::vector<double>* timestamps =
        trajectory.has_timestamps ? &trajectory.timestamps : nullptr;

    if (options.algorithm == "butterworth") {
        output_alg::ButterworthRealtimeFilter filter(options.butterworth);
        return filter.FilterTrajectoryTimed(trajectory.rigids, timestamps, true);
    }

    if (options.algorithm == "butterworth_z") {
        output_alg::ButterworthZRealtimeFilter filter(options.butterworth);
        return filter.FilterTrajectoryTimed(trajectory.rigids, timestamps, true);
    }

    if (options.algorithm == "one_euro_z") {
        output_alg::OneEuroZRealtimeFilter filter(options.one_euro);
        return filter.FilterTrajectoryTimed(trajectory.rigids, timestamps, true);
    }

    if (options.algorithm == "adaptive_local_line") {
        output_alg::AdaptiveLocalLineRealtimeFilter filter(options.adaptive_local_line);
        return filter.FilterTrajectoryTimed(trajectory.rigids, timestamps, true);
    }

    output_alg::UkfRealtimeFilter filter(options.ukf);
    return filter.FilterTrajectoryTimed(trajectory.rigids, timestamps, true);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        const CsvTrajectory input = ReadCsvTrajectory(options.input);

        // 实时接口演示：每读到一帧 RigidMatrix，就调用一次 Update 并得到滤波后的 RigidMatrix。
        const output_alg::TimedFilterTrajectoryResult filtered =
            RunRealtimeFrameFilterTimed(options, input);
        const std::string timing_output =
            ReplaceExtensionWithSuffix(options.output, ".timing.csv");
        const std::string metrics_output =
            ReplaceExtensionWithSuffix(options.output, ".metrics.json");
        const output_alg::TimingSummary timing =
            output_alg::SummarizeTiming(filtered.per_pose_time_ns, filtered.total_time_ns);

        WriteCsvTrajectory(options.output, filtered.rigids, input.timestamps, input.has_timestamps);
        WriteTimingCsv(
            timing_output,
            filtered.per_pose_time_ns,
            input.timestamps,
            input.has_timestamps);
        WriteMetricsJson(metrics_output, options, input, filtered, timing_output);

        std::cout << "wrote " << options.output
                  << " frames=" << filtered.rigids.size()
                  << " timing=" << timing_output
                  << " metrics=" << metrics_output
                  << " mean_us=" << timing.mean_time_us
                  << " p95_us=" << timing.p95_time_us
                  << " max_us=" << timing.max_time_us
                  << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        PrintUsage();
        return 2;
    }
}
