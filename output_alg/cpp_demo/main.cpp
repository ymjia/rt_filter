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
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

#include "euler_zyx_interface.hpp"
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
    std::filesystem::path input;
    std::filesystem::path output;
    std::string algorithm = "ukf";
    // 滤波参数统一放在 lambda / 单帧接口外部配置。实际工程中可把这些成员
    // 放到扫描界面类或轨迹处理类里，避免每帧重复构造参数对象。
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
            // 两个实时滤波器都支持“无时间戳输入”。此时使用采样率推导 dt。
            const double value = ParseDouble(require_value(), key);
            options.one_euro.sample_rate_hz = value;
            options.ukf.sample_rate_hz = value;
        } else if (key == "--strict-timestamps") {
            // 默认情况下，异常时间戳会退回到固定采样率；开启 strict 后会直接报错。
            options.one_euro.strict_timestamps = true;
            options.ukf.strict_timestamps = true;
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

void WriteCsvTrajectory(
    const std::filesystem::path& path,
    const std::vector<RigidMatrix>& rigids,
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

std::vector<RigidMatrix> RunFilter(const Options& options, const CsvTrajectory& trajectory) {
    // 批量轨迹滤波入口：从 CSV 读出的整段 RigidMatrix 序列一次性传给滤波器。
    // 输出的每一帧仍然是一一对应的 4x4 位姿，可直接写回 CSV 并交给 Python 框架评估。
    const std::vector<double>* timestamps =
        trajectory.has_timestamps ? &trajectory.timestamps : nullptr;
    if (options.algorithm == "one_euro_z") {
        // One Euro Z 只修改 Z 平移；X/Y 和旋转保持输入值。
        return output_alg::FilterOneEuroZTrajectory(
            trajectory.rigids,
            timestamps,
            options.one_euro);
    }
    // UKF 会同时处理平移和相对初始姿态的旋转向量，适合轨迹整体降噪。
    return output_alg::FilterUkfTrajectory(
        trajectory.rigids,
        timestamps,
        options.ukf);
}

void CallEulerZyxInterfaces(const Options& options, const CsvTrajectory& trajectory) {
    // 这个函数演示“实时扫描界面”一类场景下的单帧接口。
    // 真实工程中，下面两个参数对象和两个 m_groupedXXXFilters 通常是外部类成员；
    // lambda 每来一帧就按 groupId 找到对应滤波器状态，直接使用并更新 pos/eulerZyx。
    output_alg::OneEuroZParameters oneEuroParams = options.one_euro;
    output_alg::UkfParameters ukfParams = options.ukf;

    // 每个 groupId 保留独立滤波状态。这样多目标、多轨迹或多扫描分组之间不会互相污染。
    // 注意：滤波器必须跨帧复用，不能在每帧 lambda 内临时创建，否则实时滤波会退化为无历史状态。
    std::unordered_map<int, output_alg::OneEuroZRealtimeFilter> m_groupedOneEuroZFilters;
    std::unordered_map<int, output_alg::UkfRealtimeFilter> m_groupedUkfFilters;

    // One Euro Z 的“坐标 + zyx 欧拉角”接口。
    // 输入:
    //   groupId   分组 id，用于找到该组的历史滤波状态。
    //   pos       当前帧坐标，函数成功后会被原地改成滤波后的坐标。
    //   eulerZyx  当前帧欧拉角，顺序固定为 [z,y,x]，单位 rad；One Euro Z 不改旋转，
    //             但接口仍会把转换后的结果写回，方便调用方统一处理。
    auto calcOneEuroZEulerZyx = [&](int& groupId, Eigen::Vector3d& pos, Eigen::Vector3d& eulerZyx) -> bool {
        auto iter = m_groupedOneEuroZFilters.find(groupId);
        if (iter == m_groupedOneEuroZFilters.end()) {
            // 第一次遇到某个 groupId 时才创建滤波器，后续帧复用同一个对象中的历史状态。
            iter = m_groupedOneEuroZFilters
                       .emplace(groupId, output_alg::OneEuroZRealtimeFilter(oneEuroParams))
                       .first;
        }
        // 内部会把 pos/eulerZyx 转成 RigidMatrix，滤波后再转回 zyx 欧拉角并写回引用。
        return output_alg::FilterOneEuroZEulerZyx(iter->second, pos, eulerZyx);
    };

    // UKF 的“坐标 + zyx 欧拉角”接口。
    // 与 One Euro Z 不同，UKF 会同时滤波坐标和姿态，因此 eulerZyx 输出可能变化。
    // 欧拉角顺序同样固定为 [z,y,x]，对应 R = Rz * Ry * Rx。
    auto calcUkfEulerZyx = [&](int& groupId, Eigen::Vector3d& pos, Eigen::Vector3d& eulerZyx) -> bool {
        auto iter = m_groupedUkfFilters.find(groupId);
        if (iter == m_groupedUkfFilters.end()) {
            // UKF 的速度、协方差和参考姿态都保存在滤波器对象中，必须按 groupId 长期复用。
            iter = m_groupedUkfFilters
                       .emplace(groupId, output_alg::UkfRealtimeFilter(ukfParams))
                       .first;
        }
        // 输出同样通过 pos/eulerZyx 引用返回，便于直接写回外部成员或界面显示变量。
        return output_alg::FilterUkfEulerZyx(iter->second, pos, eulerZyx);
    };

    // demo 只取第一帧做一次单帧接口 smoke test。
    // 实际实时系统中，这段应放在每帧处理逻辑里，并把真实 groupId、pos、eulerZyx 传进 lambda。
    int groupId = 0;
    const auto sample = output_alg::PositionEulerZyxFromRigid(trajectory.rigids.front());
    Eigen::Vector3d oneEuroPos = sample.position;
    Eigen::Vector3d oneEuroEulerZyx = sample.euler_zyx;
    Eigen::Vector3d ukfPos = sample.position;
    Eigen::Vector3d ukfEulerZyx = sample.euler_zyx;

    const bool oneEuroOk = calcOneEuroZEulerZyx(groupId, oneEuroPos, oneEuroEulerZyx);
    const bool ukfOk = calcUkfEulerZyx(groupId, ukfPos, ukfEulerZyx);
    std::cout << "euler_zyx interface smoke one_euro_z=" << oneEuroOk
              << " ukf=" << ukfOk
              << " first_z=(" << oneEuroPos.z() << ", " << ukfPos.z() << ")\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        const CsvTrajectory input = ReadCsvTrajectory(options.input);
        // 单帧接口演示：模拟外部类成员 m_groupedXXXFilters + lambda 的调用方式。
        CallEulerZyxInterfaces(options, input);

        // 批量接口演示：处理完整 CSV 轨迹并写出结果，用于离线评估。
        const std::vector<RigidMatrix> filtered = RunFilter(options, input);
        WriteCsvTrajectory(options.output, filtered, input.timestamps, input.has_timestamps);
        std::cout << "wrote " << options.output.string() << " frames=" << filtered.size() << '\n';
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << '\n';
        PrintUsage();
        return 2;
    }
}
