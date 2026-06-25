// bench_embedding.cpp — measures serial vs batch embedding throughput.
//
// Serial path: one embed() call at a time, waiting for each result before
//              submitting the next — no batching opportunity for the worker.
// Batch  path: kBatch embed() calls submitted concurrently via threads so
//              the EmbeddingWorker can drain them in a single llama_decode().
//
// Usage: ./bench_embedding <model.gguf> [n_texts=64] [gpu_layers=99]
#include "memorylayer/embedding.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using memorylayer::EmbedPriority;
using memorylayer::EmbeddingWorker;
using Clock = std::chrono::high_resolution_clock;

static std::vector<std::string> make_texts(int n) {
    static const char* s[] = {
        "How does the attention mechanism work in transformers?",
        "What is the difference between TCP and UDP protocols?",
        "Explain gradient descent optimization in machine learning.",
        "How do LRU caches reduce database load in high-traffic systems?",
        "What are the SOLID principles in object-oriented design?",
        "Describe how a B-tree index speeds up SQL queries.",
        "What is the role of the KV cache in autoregressive decoding?",
        "How does Metal GPU compute work on Apple Silicon processors?",
    };
    std::vector<std::string> out;
    out.reserve(n);
    for (int i = 0; i < n; i++) out.push_back(s[i % 8]);
    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: bench_embedding <model.gguf> [n_texts=64] [gpu_layers=99]\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const int n_texts    = argc >= 3 ? std::atoi(argv[2]) : 64;
    const int gpu_layers = argc >= 4 ? std::atoi(argv[3]) : 99;

    std::cout << "Loading: " << model_path << "\n";
    EmbeddingWorker worker(model_path, gpu_layers);
    if (!worker.is_ready()) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    const int dim = worker.dimension();
    std::cout << "Model ready. dim=" << dim << "  n_texts=" << n_texts << "\n\n";

    auto texts = make_texts(n_texts);

    // ── Serial: one embed at a time (no batching) ─────────────────────────
    auto t0 = Clock::now();
    for (int i = 0; i < n_texts; i++) {
        auto emb = worker.embed(texts[i], EmbedPriority::NORMAL);
        (void)emb;
    }
    double serial_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    // ── Batch: submit kBatch concurrently → worker batches in 1 decode ───
    constexpr int kBatch = 8;
    auto t2 = Clock::now();
    for (int i = 0; i < n_texts; i += kBatch) {
        int sz = std::min(kBatch, n_texts - i);
        std::vector<std::thread> threads;
        std::vector<std::vector<float>> results(sz);
        for (int j = 0; j < sz; j++) {
            threads.emplace_back([&worker, &texts, &results, i, j]() {
                results[j] = worker.embed(texts[i + j], EmbedPriority::NORMAL);
            });
        }
        for (auto& t : threads) t.join();
    }
    double batch_ms = std::chrono::duration<double, std::milli>(Clock::now() - t2).count();

    const double sp      = serial_ms / n_texts;
    const double bp      = batch_ms  / n_texts;
    const double speedup = serial_ms / batch_ms;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "┌──────────────────────────────────────┐\n";
    std::cout << "│     Embedding Benchmark Results      │\n";
    std::cout << "├──────────────────────────────────────┤\n";
    std::cout << "│ n_texts  : " << std::setw(5) << n_texts << "                       │\n";
    std::cout << "│ dim      : " << std::setw(5) << dim     << "                       │\n";
    std::cout << "├──────────────────────────────────────┤\n";
    std::cout << "│ SERIAL total : " << std::setw(8) << serial_ms        << " ms      │\n";
    std::cout << "│ SERIAL /text : " << std::setw(8) << sp               << " ms/text │\n";
    std::cout << "│ SERIAL rate  : " << std::setw(7) << (1000.0 / sp)    << " texts/s │\n";
    std::cout << "├──────────────────────────────────────┤\n";
    std::cout << "│ BATCH  total : " << std::setw(8) << batch_ms         << " ms      │\n";
    std::cout << "│ BATCH  /text : " << std::setw(8) << bp               << " ms/text │\n";
    std::cout << "│ BATCH  rate  : " << std::setw(7) << (1000.0 / bp)    << " texts/s │\n";
    std::cout << "├──────────────────────────────────────┤\n";
    std::cout << "│ SPEEDUP      : " << std::setw(7) << speedup          << "x        │\n";
    std::cout << "└──────────────────────────────────────┘\n";
    return 0;
}
