#pragma once
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <future>
#include <atomic>

namespace memorylayer {

enum class EmbedPriority { HIGH, NORMAL };

class EmbeddingWorker {
public:
    EmbeddingWorker(const std::string& model_path, int gpu_layers);
    ~EmbeddingWorker();

    EmbeddingWorker(const EmbeddingWorker&) = delete;
    EmbeddingWorker& operator=(const EmbeddingWorker&) = delete;

    std::vector<float> embed(const std::string& text, EmbedPriority priority = EmbedPriority::NORMAL);
    int dimension() const { return n_embd_; }
    bool is_ready() const { return ready_; }
    void shutdown();

private:
    struct EmbedJob {
        std::string text;
        std::promise<std::vector<float>> promise;
    };

    void worker_loop();

    void* model_ = nullptr;
    void* ctx_ = nullptr;
    int n_embd_ = 0;
    int n_batch_max_ = 2048;
    bool ready_ = false;

    static constexpr int kMaxBatchTexts = 8;  // max texts per llama_decode call

    std::thread worker_thread_;
    std::queue<EmbedJob> jobs_high_;   // Priority queue for search
    std::queue<EmbedJob> jobs_normal_; // Normal queue for save
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
};

} // namespace memorylayer
