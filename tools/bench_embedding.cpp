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
