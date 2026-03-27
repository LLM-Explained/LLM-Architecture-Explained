#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using std::size_t;

static constexpr int VALUES_PER_BYTE = 5;
static constexpr int POW3[VALUES_PER_BYTE] = {1, 3, 9, 27, 81};

struct Sample {
    std::array<float, 2> x;
    int y;
};

float rand_uniform(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

std::vector<Sample> make_xor_data(int n, std::mt19937& rng) {
    std::vector<Sample> data;
    data.reserve(n);
    for (int i = 0; i < n; ++i) {
        float x0 = rand_uniform(rng, -1.0f, 1.0f);
        float x1 = rand_uniform(rng, -1.0f, 1.0f);
        int y = (x0 * x1 < 0.0f) ? 1 : 0;
        data.push_back({{x0, x1}, y});
    }
    return data;
}

float tanh_derivative_from_output(float y) {
    return 1.0f - y * y;
}

std::array<float, 2> softmax2(float a, float b) {
    float m = std::max(a, b);
    float ea = std::exp(a - m);
    float eb = std::exp(b - m);
    float s = ea + eb;
    return {ea / s, eb / s};
}

float cross_entropy_from_logits(float a, float b, int target) {
    auto p = softmax2(a, b);
    return -std::log(std::max(1e-8f, p[target]));
}

struct PackedTernaryWeight {
    int out_features = 0;
    int in_features = 0;
    std::vector<std::vector<uint8_t>> packed_rows;
    std::vector<float> scales;
};

int ternary_code_from_value(int t) {
    // {-1,0,1} -> {0,1,2}
    return t + 1;
}

int ternary_value_from_code(int code) {
    // {0,1,2} -> {-1,0,1}
    return code - 1;
}

std::vector<uint8_t> pack_ternary_row(const std::vector<int>& ternary_vals) {
    std::vector<int> codes;
    codes.reserve(ternary_vals.size());
    for (int v : ternary_vals) {
        if (v < -1 || v > 1) {
            throw std::runtime_error("ternary value out of range");
        }
        codes.push_back(ternary_code_from_value(v));
    }

    int padded = ((static_cast<int>(codes.size()) + VALUES_PER_BYTE - 1) / VALUES_PER_BYTE) * VALUES_PER_BYTE;
    while (static_cast<int>(codes.size()) < padded) {
        codes.push_back(1); // pad with ternary zero
    }

    std::vector<uint8_t> packed;
    packed.reserve(codes.size() / VALUES_PER_BYTE);

    for (int i = 0; i < padded; i += VALUES_PER_BYTE) {
        int byte = 0;
        for (int j = 0; j < VALUES_PER_BYTE; ++j) {
            byte += codes[i + j] * POW3[j];
        }
        packed.push_back(static_cast<uint8_t>(byte));
    }

    return packed;
}

std::vector<int> unpack_ternary_row(const std::vector<uint8_t>& packed, int original_len) {
    std::vector<int> codes;
    codes.reserve(packed.size() * VALUES_PER_BYTE);

    for (uint8_t byte : packed) {
        int x = static_cast<int>(byte);
        for (int i = 0; i < VALUES_PER_BYTE; ++i) {
            codes.push_back(x % 3);
            x /= 3;
        }
    }

    codes.resize(original_len);

    std::vector<int> ternary_vals;
    ternary_vals.reserve(original_len);
    for (int c : codes) {
        ternary_vals.push_back(ternary_value_from_code(c));
    }
    return ternary_vals;
}

struct BitLinearTrain {
    int in_features;
    int out_features;
    std::vector<std::vector<float>> weight;
    std::vector<float> bias;

    explicit BitLinearTrain(int in_f, int out_f, std::mt19937& rng)
        : in_features(in_f), out_features(out_f),
          weight(out_f, std::vector<float>(in_f)),
          bias(out_f, 0.0f) {
        std::normal_distribution<float> dist(0.0f, 0.05f);
        for (int o = 0; o < out_features; ++o) {
            for (int i = 0; i < in_features; ++i) {
                weight[o][i] = dist(rng);
            }
        }
    }

    std::pair<std::vector<std::vector<int>>, std::vector<float>> ternarize() const {
        std::vector<std::vector<int>> wt(out_features, std::vector<int>(in_features, 0));
        std::vector<float> scales(out_features, 1e-5f);

        for (int o = 0; o < out_features; ++o) {
            float mean_abs = 0.0f;
            for (int i = 0; i < in_features; ++i) {
                mean_abs += std::fabs(weight[o][i]);
            }
            mean_abs /= static_cast<float>(in_features);
            mean_abs = std::max(mean_abs, 1e-5f);
            scales[o] = mean_abs;

            for (int i = 0; i < in_features; ++i) {
                float q = std::round(weight[o][i] / mean_abs);
                q = std::max(-1.0f, std::min(1.0f, q));
                wt[o][i] = static_cast<int>(q);
            }
        }

        return {wt, scales};
    }

    std::vector<float> forward(const std::vector<float>& x) const {
        auto [wt, scales] = ternarize();
        std::vector<float> y(out_features, 0.0f);

        for (int o = 0; o < out_features; ++o) {
            float acc = bias[o];
            for (int i = 0; i < in_features; ++i) {
                acc += x[i] * (static_cast<float>(wt[o][i]) * scales[o]);
            }
            y[o] = acc;
        }
        return y;
    }

    PackedTernaryWeight export_packed() const {
        auto [wt, scales] = ternarize();
        PackedTernaryWeight packed;
        packed.out_features = out_features;
        packed.in_features = in_features;
        packed.scales = scales;
        packed.packed_rows.reserve(out_features);

        for (int o = 0; o < out_features; ++o) {
            packed.packed_rows.push_back(pack_ternary_row(wt[o]));
        }
        return packed;
    }
};

struct BitLinearInfer {
    PackedTernaryWeight packed;
    std::vector<float> bias;

    explicit BitLinearInfer(PackedTernaryWeight p, std::vector<float> b)
        : packed(std::move(p)), bias(std::move(b)) {}

    std::vector<float> forward(const std::vector<float>& x) const {
        std::vector<float> y(packed.out_features, 0.0f);
        for (int o = 0; o < packed.out_features; ++o) {
            auto ternary = unpack_ternary_row(packed.packed_rows[o], packed.in_features);
            float acc = bias[o];
            for (int i = 0; i < packed.in_features; ++i) {
                acc += x[i] * (static_cast<float>(ternary[i]) * packed.scales[o]);
            }
            y[o] = acc;
        }
        return y;
    }
};

struct TinyBitNetTrain {
    BitLinearTrain fc1;
    BitLinearTrain fc2;

    explicit TinyBitNetTrain(std::mt19937& rng)
        : fc1(2, 32, rng), fc2(32, 2, rng) {}

    std::pair<std::vector<float>, std::vector<float>> forward_hidden(const std::vector<float>& x) const {
        std::vector<float> h = fc1.forward(x);
        for (float& v : h) v = std::tanh(v);
        std::vector<float> logits = fc2.forward(h);
        return {h, logits};
    }
};

struct TinyBitNetInfer {
    BitLinearInfer fc1;
    BitLinearInfer fc2;

    TinyBitNetInfer(BitLinearInfer a, BitLinearInfer b)
        : fc1(std::move(a)), fc2(std::move(b)) {}

    std::vector<float> forward(const std::vector<float>& x) const {
        std::vector<float> h = fc1.forward(x);
        for (float& v : h) v = std::tanh(v);
        return fc2.forward(h);
    }
};

TinyBitNetInfer export_infer_model(const TinyBitNetTrain& model) {
    return TinyBitNetInfer(
        BitLinearInfer(model.fc1.export_packed(), model.fc1.bias),
        BitLinearInfer(model.fc2.export_packed(), model.fc2.bias)
    );
}

float evaluate_train_model(const TinyBitNetTrain& model, const std::vector<Sample>& data) {
    int correct = 0;
    for (const auto& s : data) {
        std::vector<float> x = {s.x[0], s.x[1]};
        auto [h, logits] = model.forward_hidden(x);
        int pred = (logits[1] > logits[0]) ? 1 : 0;
        correct += (pred == s.y);
    }
    return static_cast<float>(correct) / static_cast<float>(data.size());
}

float evaluate_infer_model(const TinyBitNetInfer& model, const std::vector<Sample>& data) {
    int correct = 0;
    for (const auto& s : data) {
        std::vector<float> x = {s.x[0], s.x[1]};
        auto logits = model.forward(x);
        int pred = (logits[1] > logits[0]) ? 1 : 0;
        correct += (pred == s.y);
    }
    return static_cast<float>(correct) / static_cast<float>(data.size());
}

int estimate_storage_bits(const PackedTernaryWeight& packed) {
    int code_bits = 0;
    for (const auto& row : packed.packed_rows) {
        code_bits += static_cast<int>(row.size()) * 8;
    }
    int scale_bits = static_cast<int>(packed.scales.size()) * 32;
    return code_bits + scale_bits;
}

int main() {
    std::mt19937 rng(0);

    auto train_data = make_xor_data(2048, rng);
    auto test_data = make_xor_data(512, rng);

    TinyBitNetTrain model(rng);

    const float lr = 0.02f;
    const float weight_decay = 1e-3f;
    const int steps = 300;

    for (int step = 0; step < steps; ++step) {
        // Full-batch gradients
        std::vector<std::vector<float>> g1_w(32, std::vector<float>(2, 0.0f));
        std::vector<float> g1_b(32, 0.0f);

        std::vector<std::vector<float>> g2_w(2, std::vector<float>(32, 0.0f));
        std::vector<float> g2_b(2, 0.0f);

        float total_loss = 0.0f;

        for (const auto& s : train_data) {
            std::vector<float> x = {s.x[0], s.x[1]};

            // Forward
            auto [wt1, sc1] = model.fc1.ternarize();
            std::vector<float> z1(32, 0.0f);
            std::vector<float> h(32, 0.0f);

            for (int o = 0; o < 32; ++o) {
                float acc = model.fc1.bias[o];
                for (int i = 0; i < 2; ++i) {
                    acc += x[i] * (static_cast<float>(wt1[o][i]) * sc1[o]);
                }
                z1[o] = acc;
                h[o] = std::tanh(acc);
            }

            auto [wt2, sc2] = model.fc2.ternarize();
            std::vector<float> logits(2, 0.0f);
            for (int o = 0; o < 2; ++o) {
                float acc = model.fc2.bias[o];
                for (int i = 0; i < 32; ++i) {
                    acc += h[i] * (static_cast<float>(wt2[o][i]) * sc2[o]);
                }
                logits[o] = acc;
            }

            total_loss += cross_entropy_from_logits(logits[0], logits[1], s.y);

            // Backward through softmax CE
            auto probs = softmax2(logits[0], logits[1]);
            std::vector<float> dlogits = {probs[0], probs[1]};
            dlogits[s.y] -= 1.0f;

            // fc2 gradients (STE: apply grads to master weights as if quantization were identity)
            std::vector<float> dh(32, 0.0f);
            for (int o = 0; o < 2; ++o) {
                g2_b[o] += dlogits[o];
                for (int i = 0; i < 32; ++i) {
                    g2_w[o][i] += dlogits[o] * h[i];
                    dh[i] += dlogits[o] * (static_cast<float>(wt2[o][i]) * sc2[o]);
                }
            }

            // tanh backward
            std::vector<float> dz1(32, 0.0f);
            for (int i = 0; i < 32; ++i) {
                dz1[i] = dh[i] * tanh_derivative_from_output(h[i]);
            }

            // fc1 gradients
            for (int o = 0; o < 32; ++o) {
                g1_b[o] += dz1[o];
                for (int i = 0; i < 2; ++i) {
                    g1_w[o][i] += dz1[o] * x[i];
                }
            }
        }

        const float inv_n = 1.0f / static_cast<float>(train_data.size());

        // SGD update
        for (int o = 0; o < 32; ++o) {
            model.fc1.bias[o] -= lr * (g1_b[o] * inv_n);
            for (int i = 0; i < 2; ++i) {
                float grad = g1_w[o][i] * inv_n + weight_decay * model.fc1.weight[o][i];
                model.fc1.weight[o][i] -= lr * grad;
            }
        }

        for (int o = 0; o < 2; ++o) {
            model.fc2.bias[o] -= lr * (g2_b[o] * inv_n);
            for (int i = 0; i < 32; ++i) {
                float grad = g2_w[o][i] * inv_n + weight_decay * model.fc2.weight[o][i];
                model.fc2.weight[o][i] -= lr * grad;
            }
        }

        if (step % 50 == 0 || step == steps - 1) {
            float acc = evaluate_train_model(model, test_data);
            std::cout << "train step=" << std::setw(3) << std::setfill('0') << step
                      << " loss=" << std::fixed << std::setprecision(4)
                      << total_loss * inv_n
                      << " test_acc=" << acc << "\n";
        }
    }

    auto infer_model = export_infer_model(model);
    float infer_acc = evaluate_infer_model(infer_model, test_data);
    std::cout << "\ninference accuracy with packed ternary weights: "
              << std::fixed << std::setprecision(4) << infer_acc << "\n";

    auto pw1 = model.fc1.export_packed();
    int dense_bits_fc1 = pw1.out_features * pw1.in_features * 32;
    int packed_bits_fc1 = estimate_storage_bits(pw1);

    std::cout << "\nFirst layer storage comparison:\n";
    std::cout << "dense FP32 bits : " << dense_bits_fc1 << "\n";
    std::cout << "packed ternary+scale bits : " << packed_bits_fc1 << "\n";
    std::cout << "effective bits/weight incl shared scales: "
              << std::fixed << std::setprecision(3)
              << static_cast<float>(packed_bits_fc1) /
                     static_cast<float>(pw1.out_features * pw1.in_features)
              << "\n";

    std::cout << "\nExample unpacked first 4 rows of first layer:\n";
    for (int r = 0; r < std::min(4, pw1.out_features); ++r) {
        auto ternary = unpack_ternary_row(pw1.packed_rows[r], pw1.in_features);
        std::cout << "row " << r << " scale=" << pw1.scales[r] << " values=[";
        for (int i = 0; i < pw1.in_features; ++i) {
            float v = static_cast<float>(ternary[i]) * pw1.scales[r];
            std::cout << v;
            if (i + 1 < pw1.in_features) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    return 0;
}
