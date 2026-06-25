#include "memorylayer/proxy.h"
#include "memorylayer/injector.h"
#include "httplib.h"
#include "json.hpp"
#include <iostream>
#include <sstream>
#include <ctime>

using json = nlohmann::json;

namespace memorylayer {

MemoryProxy::MemoryProxy(const Config& cfg, MemoryStore& store, EmbeddingWorker& embedder)
    : cfg_(cfg), store_(store), embedder_(embedder) {
    saver_thread_ = std::thread(&MemoryProxy::saver_loop, this);
}

MemoryProxy::~MemoryProxy() {
    shutdown();
}

void MemoryProxy::shutdown() {
    stop_ = true;
    save_cv_.notify_all();
    if (saver_thread_.joinable()) saver_thread_.join();
}

std::string MemoryProxy::extract_last_user_message(const json& messages) const {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if ((*it).value("role", "") == "user") {
            return (*it).value("content", "");
        }
    }
    return "";
}

std::string MemoryProxy::retrieve_and_inject(const std::string& agent_id,
                                              const std::string& user_text,
                                              json& messages) {
    if (user_text.empty() || !embedder_.is_ready()) return "";

    auto query_emb = embedder_.embed(user_text);
    if (query_emb.empty()) return "";

    auto memories = store_.search(query_emb, agent_id, cfg_.top_k,
                                   cfg_.min_score_threshold, cfg_.decay_days,
                                   cfg_.agent_boost);

    if (memories.empty()) return "";

    double now = static_cast<double>(std::time(nullptr));
    std::string context = format_memory_context(memories, now);
    inject_memories(messages, context);

    std::cout << "[proxy] Injected " << memories.size() << " memories for agent='"
              << (agent_id.empty() ? "global" : agent_id) << "'\n";

    return context;
}

void MemoryProxy::enqueue_save(const std::string& agent_id,
                                const std::string& user_text,
                                const std::string& assist_text) {
    {
        std::lock_guard<std::mutex> lock(save_mutex_);
        save_queue_.push({agent_id, user_text, assist_text});
    }
    save_cv_.notify_one();
}

void MemoryProxy::saver_loop() {
    while (!stop_) {
        SaveJob job;
        {
            std::unique_lock<std::mutex> lock(save_mutex_);
            save_cv_.wait(lock, [this] { return !save_queue_.empty() || stop_; });
            if (stop_ && save_queue_.empty()) return;
            job = std::move(save_queue_.front());
            save_queue_.pop();
        }

        if (!embedder_.is_ready()) continue;

        auto user_emb = embedder_.embed(job.user_text);
        auto assist_emb = embedder_.embed(job.assist_text);
        if (user_emb.empty() || assist_emb.empty()) continue;

        int64_t dup_id = store_.find_duplicate(user_emb, cfg_.dedup_threshold);
        if (dup_id > 0) {
            store_.touch(dup_id);
            std::cout << "[memory] Deduplicated: merged with memory #" << dup_id << "\n";
        } else {
            store_.insert(job.agent_id, job.user_text, job.assist_text, user_emb, assist_emb);
            store_.evict(job.agent_id);
            std::cout << "[memory] Saved new memory for agent='"
                      << (job.agent_id.empty() ? "global" : job.agent_id) << "'\n";
        }
    }
}

void MemoryProxy::run() {
    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"Invalid JSON"})", "application/json");
            return;
        }

        std::string agent_id;
        if (req.has_header("X-Agent-Id")) {
            agent_id = req.get_header_value("X-Agent-Id");
        }

        if (!body.contains("messages") || !body["messages"].is_array()) {
            res.status = 400;
            res.set_content(R"({"error":"Missing messages array"})", "application/json");
            return;
        }

        auto& messages = body["messages"];
        std::string user_text = extract_last_user_message(messages);

        retrieve_and_inject(agent_id, user_text, messages);

        bool streaming = body.value("stream", false);

        // Parse backend host and port
        httplib::Client cli(cfg_.backend_url);
        cli.set_read_timeout(300);
        cli.set_connection_timeout(10);

        std::string modified_body = body.dump();

        if (!streaming) {
            auto backend_res = cli.Post("/v1/chat/completions",
                                         modified_body, "application/json");
            if (!backend_res) {
                res.status = 502;
                res.set_content(R"({"error":"Backend unreachable"})", "application/json");
                return;
            }

            res.status = backend_res->status;
            res.set_content(backend_res->body, backend_res->get_header_value("Content-Type"));

            if (backend_res->status == 200 && !user_text.empty()) {
                try {
                    auto resp_json = json::parse(backend_res->body);
                    std::string assist_text = resp_json["choices"][0]["message"]["content"].get<std::string>();
                    enqueue_save(agent_id, user_text, assist_text);
                } catch (...) {}
            }
        } else {
            // Streaming: POST normally, buffer response, parse SSE for content, forward
            auto backend_res = cli.Post("/v1/chat/completions",
                                         modified_body, "application/json");
            if (!backend_res) {
                res.status = 502;
                res.set_content(R"({"error":"Backend unreachable"})", "application/json");
                return;
            }

            res.set_header("Content-Type", "text/event-stream");
            res.set_header("Cache-Control", "no-cache");
            res.status = backend_res->status;
            res.set_content(backend_res->body, "text/event-stream");

            // Parse SSE to extract assistant content for memory saving
            if (backend_res->status == 200 && !user_text.empty()) {
                std::string accumulated_content;
                std::istringstream stream(backend_res->body);
                std::string line;
                while (std::getline(stream, line)) {
                    if (line.size() > 6 && line.substr(0, 6) == "data: ") {
                        std::string payload = line.substr(6);
                        if (payload == "[DONE]") continue;
                        try {
                            auto j = json::parse(payload);
                            if (j.contains("choices") && !j["choices"].empty()) {
                                auto& delta = j["choices"][0]["delta"];
                                if (delta.contains("content")) {
                                    accumulated_content += delta["content"].get<std::string>();
                                }
                            }
                        } catch (...) {}
                    }
                }
                if (!accumulated_content.empty()) {
                    enqueue_save(agent_id, user_text, accumulated_content);
                }
            }
        }
    });

    // Catch-all for other /v1/* endpoints
    svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) -> httplib::Server::HandlerResponse {
        // Only catch unhandled paths under /v1/
        if (req.path.substr(0, 3) == "/v1" && req.path != "/v1/chat/completions") {
            httplib::Client cli(cfg_.backend_url);
            cli.set_read_timeout(60);
            httplib::Result result = (req.method == "POST")
                ? cli.Post(req.path, req.body, req.get_header_value("Content-Type"))
                : cli.Get(req.path);

            if (result) {
                res.status = result->status;
                res.set_content(result->body, result->get_header_value("Content-Type"));
            } else {
                res.status = 502;
                res.set_content(R"({"error":"Backend unreachable"})", "application/json");
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    std::cout << "[proxy] Listening on http://0.0.0.0:" << cfg_.port << "\n";
    std::cout << "[proxy] Backend: " << cfg_.backend_url << "\n";
    std::cout << "[proxy] Memory injection: top_k=" << cfg_.top_k
              << ", decay_days=" << cfg_.decay_days << "\n";

    svr.listen("0.0.0.0", cfg_.port);
}

} // namespace memorylayer
