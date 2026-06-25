#include "memorylayer/proxy.h"
#include "memorylayer/injector.h"
#include "memorylayer/logger.h"
#include "httplib.h"
#include "json.hpp"
#include <iostream>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <vector>
#include <utility>

using json = nlohmann::json;

namespace memorylayer {

MemoryProxy::MemoryProxy(const Config& cfg, MemoryStore& store, EmbeddingWorker& embedder)
    : cfg_(cfg), store_(store), embedder_(embedder) {
    saver_thread_ = std::thread(&MemoryProxy::saver_loop, this);
}

MemoryProxy::~MemoryProxy() {
    shutdown();
}

void MemoryProxy::stop() {
    auto* svr = svr_ptr_.load();
    if (svr) svr->stop();
}

void MemoryProxy::shutdown() {
    stop_ = true;
    save_cv_.notify_all();
    if (saver_thread_.joinable()) saver_thread_.join();
}

bool MemoryProxy::check_admin_auth(const httplib::Request& req, httplib::Response& res) const {
    if (cfg_.admin_token.empty()) return true;  // Auth disabled
    std::string auth = req.get_header_value("Authorization");
    if (auth == "Bearer " + cfg_.admin_token) return true;
    res.status = 401;
    res.set_content(R"({"error":"Unauthorized — set Authorization: Bearer <admin-token>"})",
                    "application/json");
    return false;
}

std::string MemoryProxy::extract_last_user_message(const json& messages) const {    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if ((*it).value("role", "") == "user") {
            return (*it).value("content", "");
        }
    }
    return "";
}

std::string MemoryProxy::extract_conversation_context(const json& messages, int max_turns) const {
    std::vector<std::pair<std::string, std::string>> turns;

    for (const auto& msg : messages) {
        std::string role = msg.value("role", "");
        std::string content = msg.value("content", "");
        if (role == "user" || role == "assistant") {
            turns.push_back({role, content});
        }
    }

    int start = std::max(0, static_cast<int>(turns.size()) - max_turns * 2);

    std::string context;
    for (int i = start; i < static_cast<int>(turns.size()); i++) {
        if (!context.empty()) context += " | ";
        context += turns[i].first + ": " + turns[i].second.substr(0, 200);
    }

    return context;
}

std::string MemoryProxy::retrieve_and_inject(const std::string& agent_id,
                                              const std::string& user_text,
                                              json& messages) {
    if (user_text.empty() || !embedder_.is_ready()) return "";

    auto query_emb = embedder_.embed(user_text, EmbedPriority::HIGH);
    if (query_emb.empty()) return "";

    auto memories = store_.search(query_emb, agent_id, cfg_.top_k,
                                   cfg_.min_score_threshold, cfg_.decay_days,
                                   cfg_.agent_boost);

    if (memories.empty()) return "";

    double now = static_cast<double>(std::time(nullptr));
    std::string context = format_memory_context_budgeted(memories, now, cfg_.max_inject_tokens);
    inject_memories(messages, context);

    {
        std::lock_guard<std::mutex> lock(debug_mutex_);
        last_injection_ = {agent_id, user_text, context, static_cast<double>(std::time(nullptr))};
    }

    LOG_INFO("proxy", "Injected " + std::to_string(memories.size()) + " memories for agent='" + (agent_id.empty() ? "global" : agent_id) + "'");

    return context;
}

void MemoryProxy::enqueue_save(const std::string& agent_id,
                                const std::string& user_text,
                                const std::string& assist_text) {
    {
        std::lock_guard<std::mutex> lock(save_mutex_);
        if (save_queue_.size() >= 100) {
            LOG_WARN("proxy", "Save queue full, dropping oldest");
            save_queue_.pop();
        }
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

        int64_t dup_id = store_.find_duplicate(user_emb, cfg_.dedup_threshold, job.agent_id);
        if (dup_id > 0) {
            store_.touch(dup_id);
            LOG_INFO("memory", "Deduplicated: merged with memory #" + std::to_string(dup_id));
        } else {
            store_.insert(job.agent_id, job.user_text, job.assist_text, user_emb, assist_emb);
            store_.evict(job.agent_id);
            LOG_INFO("memory", "Saved new memory for agent='" + (job.agent_id.empty() ? "global" : job.agent_id) + "'");
        }
    }
}

void MemoryProxy::run() {
    httplib::Server svr;
    svr_ptr_.store(&svr);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });

    // Admin API: list memories
    svr.Get("/admin/memories", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(req, res)) return;
        std::string agent_id = req.has_param("agent_id") ? req.get_param_value("agent_id") : "";
        int limit = 50;
        int offset = 0;
        if (req.has_param("limit")) {
            try { limit = std::stoi(req.get_param_value("limit")); } catch (...) {}
        }
        if (req.has_param("offset")) {
            try { offset = std::stoi(req.get_param_value("offset")); } catch (...) {}
        }

        auto memories = store_.list_memories(agent_id, limit, offset);

        json result = json::array();
        for (const auto& m : memories) {
            result.push_back({
                {"id", m.id},
                {"agent_id", m.agent_id},
                {"created_at", m.created_at},
                {"user_text", m.user_text},
                {"assist_text", m.assist_text},
                {"access_count", m.access_count}
            });
        }
        res.set_content(result.dump(2), "application/json");
    });

    // Admin API: delete a memory
    svr.Delete(R"(/admin/memories/(\d+))", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(req, res)) return;
        int64_t id;
        try {
            id = std::stoll(req.matches[1]);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"Invalid memory ID"})", "application/json");
            return;
        }
        bool removed = store_.remove(id);
        if (removed) {
            res.set_content(R"({"status":"deleted"})", "application/json");
        } else {
            res.status = 404;
            res.set_content(R"({"error":"Memory not found"})", "application/json");
        }
    });

    // Admin API: stats
    svr.Get("/admin/stats", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(req, res)) return;
        auto stats = store_.get_stats();
        json result = {
            {"total_memories", stats.total_memories},
            {"total_agents", stats.total_agents},
            {"per_agent", json::object()}
        };
        for (const auto& [agent, count] : stats.per_agent_counts) {
            result["per_agent"][agent] = count;
        }
        res.set_content(result.dump(2), "application/json");
    });

    // Admin API: last injection debug info
    svr.Get("/admin/debug/last-injection", [this](const httplib::Request& req, httplib::Response& res) {
        if (!check_admin_auth(req, res)) return;
        std::lock_guard<std::mutex> lock(debug_mutex_);
        json result = {
            {"last_agent_id", last_injection_.agent_id},
            {"last_query", last_injection_.query},
            {"last_injected_context", last_injection_.injected_context},
            {"last_timestamp", last_injection_.timestamp}
        };
        res.set_content(result.dump(2), "application/json");
    });

    svr.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
        // Body size limit: 1MB
        if (req.body.size() > 1024 * 1024) {
            res.status = 413;
            res.set_content(R"({"error":"Request body too large"})", "application/json");
            return;
        }

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
                    std::string conv_context = extract_conversation_context(messages);
                    std::string save_user = conv_context.empty() ? user_text : conv_context;
                    enqueue_save(agent_id, save_user, assist_text);
                } catch (const std::exception& e) {
                    LOG_WARN("proxy", std::string("Failed to parse backend response for memory save: ") + e.what());
                }
            }
        } else {
            // TRUE streaming: forward SSE chunks in real-time via chunked response
            std::string agent_id_copy = agent_id;
            std::string user_text_copy = user_text;
            std::string backend_url = cfg_.backend_url;
            std::string mod_body = modified_body;
            std::string conv_context = extract_conversation_context(messages);
            std::string save_user = conv_context.empty() ? user_text_copy : conv_context;

            res.set_chunked_content_provider(
                "text/event-stream",
                [this, mod_body, backend_url, agent_id_copy, save_user]
                (size_t /*offset*/, httplib::DataSink& sink) -> bool {
                    httplib::Client backend(backend_url);
                    backend.set_read_timeout(300);
                    backend.set_connection_timeout(10);

                    std::string accumulated_content;

                    auto result = backend.Post(
                        "/v1/chat/completions",
                        mod_body.size(),
                        [&mod_body](size_t offset, size_t length, httplib::DataSink& body_sink) -> bool {
                            size_t remaining = mod_body.size() - offset;
                            size_t to_write = std::min(length, remaining);
                            body_sink.write(mod_body.data() + offset, to_write);
                            return true;
                        },
                        "application/json",
                        [&sink, &accumulated_content](const char* data, size_t len) -> bool {
                            // Forward chunk to client in real-time
                            sink.write(data, len);

                            // Parse SSE lines to accumulate assistant content
                            std::string chunk(data, len);
                            std::istringstream stream(chunk);
                            std::string line;
                            while (std::getline(stream, line)) {
                                if (line.size() > 6 && line.substr(0, 6) == "data: ") {
                                    std::string payload = line.substr(6);
                                    if (payload == "[DONE]") continue;
                                    try {
                                        auto j = nlohmann::json::parse(payload);
                                        if (j.contains("choices") && !j["choices"].empty()) {
                                            auto& delta = j["choices"][0]["delta"];
                                            if (delta.contains("content")) {
                                                accumulated_content += delta["content"].get<std::string>();
                                            }
                                        }
                                    } catch (...) {
                                        // Partial SSE chunks commonly fail JSON parse — skip silently
                                    }                                }
                            }
                            return true;
                        }
                    );

                    // Save accumulated content to memory after streaming completes
                    if (!accumulated_content.empty() && !save_user.empty()) {
                        enqueue_save(agent_id_copy, save_user, accumulated_content);
                    }

                    sink.done();
                    return false;
                },
                [](bool) {}
            );
        }
    });

    // Catch-all for other /v1/* endpoints
    svr.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) -> httplib::Server::HandlerResponse {
        // Only catch unhandled paths under /v1/
        if (req.path.compare(0, 4, "/v1/") == 0 && req.path != "/v1/chat/completions") {
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

    LOG_INFO("proxy", "Listening on http://0.0.0.0:" + std::to_string(cfg_.port));
    LOG_INFO("proxy", "Backend: " + cfg_.backend_url);
    LOG_INFO("proxy", "Memory injection: top_k=" + std::to_string(cfg_.top_k) + ", decay_days=" + std::to_string(cfg_.decay_days));

    svr.listen("0.0.0.0", cfg_.port);
    svr_ptr_.store(nullptr);  // Clear before svr goes out of scope (stop() safety)
}

} // namespace memorylayer
