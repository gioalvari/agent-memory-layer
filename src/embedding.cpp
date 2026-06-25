#include "memorylayer/embedding.h"
#include "llama.h"
#include <cstring>
#include <cmath>
#include <iostream>

namespace memorylayer {

static void batch_add(llama_batch& batch, llama_token id, llama_pos pos,
                      const std::vector<llama_seq_id>& seq_ids, bool logits) {
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = static_cast<int32_t>(seq_ids.size());
    for (size_t i = 0; i < seq_ids.size(); i++) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

EmbeddingWorker::EmbeddingWorker(const std::string& model_path, int gpu_layers) {
    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;

    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "[embedding] Failed to load model: " << model_path << "\n";
        return;
    }
    model_ = model;

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 2048;
    ctx_params.embeddings = true;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "[embedding] Failed to create context\n";
        llama_model_free(model);
        model_ = nullptr;
        return;
    }
    ctx_ = ctx;

    n_embd_ = llama_model_n_embd(model);
    ready_ = true;

    std::cout << "[embedding] Model loaded: " << model_path
              << " (dim=" << n_embd_ << ", gpu_layers=" << gpu_layers << ")\n";

    worker_thread_ = std::thread(&EmbeddingWorker::worker_loop, this);
}

EmbeddingWorker::~EmbeddingWorker() {
    shutdown();
    if (ctx_) llama_free(static_cast<llama_context*>(ctx_));
    if (model_) llama_model_free(static_cast<llama_model*>(model_));
    llama_backend_free();
}

void EmbeddingWorker::shutdown() {
    stop_ = true;
    cv_.notify_all();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

std::vector<float> EmbeddingWorker::embed(const std::string& text) {
    if (!ready_) return {};

    std::promise<std::vector<float>> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_.push({text, std::move(promise)});
    }
    cv_.notify_one();

    return future.get();
}

void EmbeddingWorker::worker_loop() {
    auto* model = static_cast<llama_model*>(model_);
    auto* ctx = static_cast<llama_context*>(ctx_);
    const auto* vocab = llama_model_get_vocab(model);

    while (!stop_) {
        EmbedJob job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] { return !jobs_.empty() || stop_; });
            if (stop_ && jobs_.empty()) return;
            job = std::move(jobs_.front());
            jobs_.pop();
        }

        // Tokenize
        int max_tokens = static_cast<int>(job.text.size()) + 32;
        std::vector<llama_token> tokens(max_tokens);
        int n_tokens = llama_tokenize(vocab, job.text.c_str(), job.text.size(),
                                      tokens.data(), max_tokens, true, true);
        if (n_tokens < 0) {
            job.promise.set_value({});
            continue;
        }
        tokens.resize(n_tokens);

        // Clear KV cache
        llama_memory_clear(llama_get_memory(ctx), true);

        // Create batch
        llama_batch batch = llama_batch_init(n_tokens, 0, 1);
        for (int i = 0; i < n_tokens; i++) {
            batch_add(batch, tokens[i], i, {0}, (i == n_tokens - 1));
        }

        // Decode
        if (llama_decode(ctx, batch) != 0) {
            job.promise.set_value({});
            llama_batch_free(batch);
            continue;
        }

        // Get embeddings
        std::vector<float> result(n_embd_, 0.0f);
        const float* emb = llama_get_embeddings_seq(ctx, 0);
        if (emb) {
            std::memcpy(result.data(), emb, n_embd_ * sizeof(float));
        } else {
            const float* last_emb = llama_get_embeddings_ith(ctx, n_tokens - 1);
            if (last_emb) {
                std::memcpy(result.data(), last_emb, n_embd_ * sizeof(float));
            }
        }

        // L2 normalize
        float norm = 0.0f;
        for (float v : result) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 0.0f) {
            for (float& v : result) v /= norm;
        }

        llama_batch_free(batch);
        job.promise.set_value(std::move(result));
    }
}

} // namespace memorylayer
