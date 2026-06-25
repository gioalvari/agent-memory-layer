#include "memorylayer/embedding.h"
#include "memorylayer/logger.h"
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
        LOG_ERROR("embedding", "Failed to load model: " + model_path);
        return;
    }
    model_ = model;

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 2048;
    ctx_params.embeddings = true;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERROR("embedding", "Failed to create context");
        llama_model_free(model);
        model_ = nullptr;
        return;
    }
    ctx_ = ctx;

    n_embd_ = llama_model_n_embd(model);
    n_batch_max_ = ctx_params.n_batch;
    ready_ = true;

    LOG_INFO("embedding", "Model loaded: " + model_path + " (dim=" + std::to_string(n_embd_) + ", gpu_layers=" + std::to_string(gpu_layers) + ")");

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

std::vector<float> EmbeddingWorker::embed(const std::string& text, EmbedPriority priority) {
    if (!ready_) return {};

    std::promise<std::vector<float>> promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (priority == EmbedPriority::HIGH) {
            jobs_high_.push({text, std::move(promise)});
        } else {
            jobs_normal_.push({text, std::move(promise)});
        }
    }
    cv_.notify_one();

    return future.get();
}

void EmbeddingWorker::worker_loop() {
    auto* model = static_cast<llama_model*>(model_);
    auto* ctx = static_cast<llama_context*>(ctx_);
    const auto* vocab = llama_model_get_vocab(model);

    // Encode a single text and return a unit-normalized embedding (serial fallback).
    auto encode_single = [&](const std::string& text) -> std::vector<float> {
        int max_t = static_cast<int>(text.size()) + 32;
        std::vector<llama_token> tokens(max_t);
        int n = llama_tokenize(vocab, text.c_str(), text.size(),
                               tokens.data(), max_t, true, true);
        if (n <= 0) return {};
        tokens.resize(n);

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) {
            batch_add(batch, tokens[i], i, {0}, i == n - 1);
        }
        std::vector<float> result;
        if (llama_decode(ctx, batch) == 0) {
            result.resize(n_embd_, 0.0f);
            const float* emb = llama_get_embeddings_seq(ctx, 0);
            if (!emb) emb = llama_get_embeddings_ith(ctx, n - 1);
            if (emb) {
                std::memcpy(result.data(), emb, n_embd_ * sizeof(float));
                float norm = 0.0f;
                for (float v : result) norm += v * v;
                norm = std::sqrt(norm);
                if (norm > 0.0f) for (float& v : result) v /= norm;
            } else {
                result.clear();
            }
        }
        llama_batch_free(batch);
        return result;
    };

    while (!stop_) {
        std::vector<EmbedJob> jobs;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return !jobs_high_.empty() || !jobs_normal_.empty() || stop_;
            });
            if (stop_ && jobs_high_.empty() && jobs_normal_.empty()) return;

            // Drain one priority level at a time — HIGH drains fully before NORMAL.
            auto& src = jobs_high_.empty() ? jobs_normal_ : jobs_high_;
            while (!src.empty() && static_cast<int>(jobs.size()) < kMaxBatchTexts) {
                jobs.push_back(std::move(src.front()));
                src.pop();
            }
        }

        if (jobs.size() == 1) {
            jobs[0].promise.set_value(encode_single(jobs[0].text));
            continue;
        }

        // Multi-text batch path: tokenize all and check if they fit in one llama_decode.
        std::vector<std::vector<llama_token>> all_tokens(jobs.size());
        int total_tokens = 0;
        for (int i = 0; i < static_cast<int>(jobs.size()); i++) {
            const auto& text = jobs[i].text;
            int max_t = static_cast<int>(text.size()) + 32;
            all_tokens[i].resize(max_t);
            int n = llama_tokenize(vocab, text.c_str(), text.size(),
                                   all_tokens[i].data(), max_t, true, true);
            if (n > 0) {
                all_tokens[i].resize(n);
                total_tokens += n;
            } else {
                all_tokens[i].clear();
            }
        }

        if (total_tokens > n_batch_max_) {
            // Batch would overflow the context — process each text serially.
            for (auto& job : jobs) {
                job.promise.set_value(encode_single(job.text));
            }
            continue;
        }

        // All texts fit: build one multi-sequence batch and decode once.
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init(std::max(total_tokens, 1), 0, 1);

        std::vector<int> job_seq(jobs.size(), -1);
        int seq_id = 0;
        for (int i = 0; i < static_cast<int>(jobs.size()); i++) {
            if (all_tokens[i].empty()) continue;
            job_seq[i] = seq_id;
            const int n = static_cast<int>(all_tokens[i].size());
            for (int t = 0; t < n; t++) {
                batch_add(batch, all_tokens[i][t], t,
                          {static_cast<llama_seq_id>(seq_id)}, t == n - 1);
            }
            seq_id++;
        }

        bool decode_ok = (batch.n_tokens > 0) && (llama_decode(ctx, batch) == 0);
        for (int i = 0; i < static_cast<int>(jobs.size()); i++) {
            if (!decode_ok || job_seq[i] < 0) {
                jobs[i].promise.set_value({});
                continue;
            }
            std::vector<float> result(n_embd_, 0.0f);
            const float* emb = llama_get_embeddings_seq(ctx, job_seq[i]);
            if (emb) {
                std::memcpy(result.data(), emb, n_embd_ * sizeof(float));
                float norm = 0.0f;
                for (float v : result) norm += v * v;
                norm = std::sqrt(norm);
                if (norm > 0.0f) for (float& v : result) v /= norm;
            }
            jobs[i].promise.set_value(std::move(result));
        }
        llama_batch_free(batch);
    }
}

} // namespace memorylayer
