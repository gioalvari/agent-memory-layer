// Benchmark: serial vs concurrent batch embedding throughput.
//
// Usage: ./bench_embedding <model.gguf> [gpu_layers=99]
//
// Single-thread serial:   embed texts one-by-one (N × llama_decode)
// Multi-thread concurrent: submit N texts from N threads simultaneously;
//   the worker drains up to kMaxBatchTexts=8 in one llama_decode.
//
// Reports: ms/text and speedup for N = 1, 2, 4, 8.

#include "memorylayer/embedding.h"

#include <chrono>
#include <cmath>
#include <future>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using namespace memorylayer;
using Clock = std::chrono::high_resolution_clock;
using Ms    = std::chrono::duration<double, std::milli>;

static const std::vector<std::string> kTexts = {
    "The quick brown fox jumps over the lazy dog.",
    "How do I implement a radix tree in C++?",
    "What are the differences between PostgreSQL and SQLite?",
    "Explain the attention mechanism in transformer models.",
    "How does Apple Metal differ from CUDA for GPU compute?",
    "What is the best way to handle memory management in C++?",
    "Describe the architecture of a microservices system.",
    "What are the trade-offs between synchronous and asynchronous I/O?",
};

static double cosine_similarity(const std::vector<float>& lhs,
                                const std::vector<float>& rhs) {
    if (lhs.size() != rhs.size() || lhs.empty()) return 0.0;
    double dot = 0.0;
    double lhs_norm = 0.0;
    double rhs_norm = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        dot += static_cast<double>(lhs[i]) * rhs[i];
        lhs_norm += static_cast<double>(lhs[i]) * lhs[i];
        rhs_norm += static_cast<double>(rhs[i]) * rhs[i];
    }
    return dot / std::sqrt(lhs_norm * rhs_norm);
}

static double measure_serial(EmbeddingWorker& w, int n, int runs) {
    double total = 0;
    for (int r = 0; r < runs; r++) {
        auto t0 = Clock::now();
        for (int i = 0; i < n; i++) {
            w.embed(kTexts[i % kTexts.size()], EmbedPriority::HIGH);
        }
        total += Ms(Clock::now() - t0).count();
    }
    return total / runs / n;  // ms per text
}

// Submit N embed() calls from N threads simultaneously so the worker
// sees them all queued at once and can batch them in a single decode.
static double measure_concurrent(EmbeddingWorker& w, int n, int runs) {
    double total = 0;
    for (int r = 0; r < runs; r++) {
        std::vector<std::future<std::vector<float>>> futures;
        futures.reserve(n);

        auto t0 = Clock::now();
        for (int i = 0; i < n; i++) {
            futures.push_back(std::async(std::launch::async, [&w, i] {
                return w.embed(kTexts[i % kTexts.size()], EmbedPriority::HIGH);
            }));
        }
        for (auto& f : futures) f.get();  // wait for all
        total += Ms(Clock::now() - t0).count();
    }
    return total / runs / n;  // ms per text
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf> [gpu_layers=99]\n";
        return 1;
    }
    const std::string model_path = argv[1];
    const int gpu_layers = (argc >= 3) ? std::stoi(argv[2]) : 99;

    std::cout << "Loading model: " << model_path << " (gpu_layers=" << gpu_layers << ")\n";
    EmbeddingWorker worker(model_path, gpu_layers);
    if (!worker.is_ready()) {
        std::cerr << "Failed to load model.\n";
        return 1;
    }

    // Warm up: two embeds to initialize Metal pipelines.
    worker.embed(kTexts[0], EmbedPriority::HIGH);
    worker.embed(kTexts[1], EmbedPriority::HIGH);

    // Correctness check: compare independently decoded embeddings with one
    // eight-request concurrent batch. This specifically validates sequence-id
    // mapping in EmbeddingWorker's multi-text path.
    std::vector<std::vector<float>> serial_embeddings;
    serial_embeddings.reserve(kTexts.size());
    for (const auto& text : kTexts) {
        serial_embeddings.push_back(worker.embed(text, EmbedPriority::HIGH));
    }
    std::vector<std::future<std::vector<float>>> batch_futures;
    batch_futures.reserve(kTexts.size());
    for (size_t i = 0; i < kTexts.size(); ++i) {
        batch_futures.push_back(std::async(std::launch::async, [&worker, i] {
            return worker.embed(kTexts[i], EmbedPriority::HIGH);
        }));
    }
    double min_cosine = 1.0;
    for (size_t i = 0; i < batch_futures.size(); ++i) {
        min_cosine = std::min(min_cosine,
                              cosine_similarity(serial_embeddings[i], batch_futures[i].get()));
    }
    std::cout << "Batch correctness: minimum serial/batch cosine = "
              << std::fixed << std::setprecision(6) << min_cosine << "\n";
    if (min_cosine <= 0.999) {
        std::cerr << "Batch correctness FAILED (expected cosine > 0.999)\n";
        return 2;
    }

    const int kRuns = 5;

    std::cout << "\n=== Embedding Benchmark (average over " << kRuns << " runs) ===\n";
    std::cout << std::setw(4)  << "N"
              << std::setw(16) << "serial ms/text"
              << std::setw(16) << "batch ms/text"
              << std::setw(12) << "speedup"
              << "\n"
              << std::string(48, '-') << "\n";

    for (int n : {1, 2, 4, 8}) {
        double serial = measure_serial(worker, n, kRuns);
        double batch  = measure_concurrent(worker, n, kRuns);
        double speedup = serial / batch;

        std::cout << std::setw(4)  << n
                  << std::setw(16) << std::fixed << std::setprecision(2) << serial
                  << std::setw(16) << std::fixed << std::setprecision(2) << batch
                  << std::setw(11) << std::fixed << std::setprecision(2) << speedup
                  << "x\n";
    }

    std::cout << "\nNote: batch speedup relies on concurrent submission (N threads).\n"
              << "Serial baseline uses single-threaded sequential embed().\n";
    return 0;
}
