#pragma once

#include "neural_runtime.hpp"
#include <locale>
#include <numeric>

namespace neural {

struct DatasetRow {
    std::vector<double> features;
    int label = 0;
};

// The shape is supplied by checked IR, not inferred from the file. Reading
// the complete dataset first keeps malformed input from triggering SGD.
// Streaming row parser shared by training, validation and inference.
// The caller controls whether rows are retained in memory.
template<class RowSink>
inline void visit_dataset_rows(const std::string& path, int samples, int width,
                               int classes, RowSink&& consume) {
    Model::safe_model_filename(path);
    require(width > 0 && width <= 2048 && samples > 0 && samples <= 100000 &&
            static_cast<uint64_t>(samples) * width <= 8000000, "dataset shape exceeds limits");
    std::error_code ec;
    const auto status = std::filesystem::symlink_status(path, ec);
    require(!ec && std::filesystem::is_regular_file(status), "dataset must be a regular local file");
    const auto bytes = std::filesystem::file_size(path, ec);
    require(!ec && bytes <= 128ull * 1024 * 1024, "dataset file exceeds byte limit");
    std::ifstream stream(path);
    require(bool(stream), "cannot open dataset");
    std::string line;
    for (int row = 0; row < samples; ++row) {
        const auto location = "dataset row " + std::to_string(row + 1);
        require(bool(std::getline(stream, line)), location + ": missing");
        require(line.size() <= static_cast<size_t>(width) * 64 + 32, location + ": too long");
        std::istringstream values(line);
        values.imbue(std::locale::classic());
        DatasetRow sample;
        if (classes) {
            require(bool(values >> sample.label) && sample.label >= 0 && sample.label < classes,
                    location + ": invalid class label");
        }
        sample.features.resize(static_cast<size_t>(width));
        for (size_t col = 0; col < sample.features.size(); ++col) {
            if (classes || col != 0) {
                const int separator = values.peek();
                if (separator == std::char_traits<char>::eof() ||
                    !std::isspace(static_cast<char>(separator), std::locale::classic()))
                    throw std::runtime_error(location + ": missing feature separator");
            }
            double& value = sample.features[col];
            if (!(values >> value) || !std::isfinite(value) || std::abs(value) >= 1.e9)
                throw std::runtime_error(location + ": invalid or missing feature");
        }
        values >> std::ws;
        require(values.eof(), location + ": extra field");
        consume(std::move(sample));
    }
    require(stream.peek() == std::char_traits<char>::eof() && !stream.bad(),
            "dataset contains extra rows or a read failed");
}

inline std::vector<DatasetRow> read_dataset(const std::string& path, int samples,
                                            int width, int classes = 0) {
    std::vector<DatasetRow> rows;
    rows.reserve(static_cast<size_t>(samples));
    visit_dataset_rows(path, samples, width, classes, [&](DatasetRow&& row) {
        rows.push_back(std::move(row));
    });
    return rows;
}

inline void train_data(Model& model, const std::string& path, int samples, int epochs,
                       double rate, uint32_t seed) {
    require(!model.layers.empty() && model.layers.back().out >= 2, "dataset needs two or more classes");
    require(samples > 0 && epochs > 0 && epochs <= 1000 &&
            static_cast<uint64_t>(samples) * epochs <= 1000000, "dataset training exceeds step limit");
    require(rate > 0 && rate <= 1, "dataset learning rate must be (0,1]");
    const int classes = model.layers.back().out;
    const auto rows = read_dataset(path, samples, model.ninput, classes);
    std::vector<int> order(rows.size());
    std::vector<double> target(static_cast<size_t>(classes), 0.);
    std::mt19937 random(seed);
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::iota(order.begin(), order.end(), 0);
        // Fixed Fisher-Yates index mapping avoids library-dependent std::shuffle.
        for (size_t i = order.size(); i > 1; --i) {
            const size_t j = static_cast<size_t>((uint64_t(random()) * i) >> 32);
            std::swap(order[i - 1], order[j]);
        }
        for (int index : order) {
            const auto& row = rows[static_cast<size_t>(index)];
            model.observe(row.features);
            target[static_cast<size_t>(row.label)] = 1.;
            model.learn(target, rate);
            target[static_cast<size_t>(row.label)] = 0.;
        }
    }
    std::cout << "DATA_TRAIN samples=" << samples << " epochs=" << epochs
              << " steps=" << static_cast<uint64_t>(samples) * epochs << " seed=" << seed << '\n';
}

// Existing fast one-pass behavior: validated dataset retained in memory.
inline void predict_data(Model& model, const std::string& path, int samples) {
    const auto rows = read_dataset(path, samples, model.ninput);
    for (size_t i = 0; i < rows.size(); ++i) {
        model.observe(rows[i].features);
        const auto& prediction = model.infer(false);
        std::cout << "DATA_PREDICT index=" << i << " values=";
        for (size_t j = 0; j < prediction.size(); ++j)
            std::cout << (j ? "," : "") << std::setprecision(17) << prediction[j];
        std::cout << '\n';
    }
}

// Opt-in memory profile: pre-validate every row before producing output,
// then stream input one row at a time. This trades more file parsing for
// O(input width) row storage instead of O(samples * input width).
inline void predict_data_streaming(Model& model, const std::string& path, int samples) {
    visit_dataset_rows(path, samples, model.ninput, 0, [](DatasetRow&&) {});
    size_t index = 0;
    visit_dataset_rows(path, samples, model.ninput, 0, [&](DatasetRow&& row) {
        model.observe(std::move(row.features));
        const auto& prediction = model.infer(false);
        std::cout << "DATA_PREDICT index=" << index++ << " values=";
        for (size_t j = 0; j < prediction.size(); ++j)
            std::cout << (j ? "," : "") << std::setprecision(17) << prediction[j];
        std::cout << '\n';
    });
}

} // namespace neural
