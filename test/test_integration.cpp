/**
 * Integration test for the Memory Proxy HTTP endpoints.
 *
 * Spins up a mock LLM backend and the real proxy (with embeddings disabled),
 * then validates every HTTP endpoint end-to-end.
 */
#include "memorylayer/config.h"
#include "memorylayer/memory_store.h"
#include "memorylayer/proxy.h"
#include "httplib.h"
#include "json.hpp"
#include <cassert>
#include <iostream>
#include <cstdio>
#include <thread>
#include <chrono>
#include <atomic>

using json = nlohmann::json;

static constexpr int BACKEND_PORT = 19999;
static constexpr int PROXY_PORT = 19900;

// --------------- Mock LLM Backend ---------------
class MockBackend {
public:
    void start() {
        thread_ = std::thread([this] {
            svr_.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
                request_count_++;
                last_body_ = req.body;

                auto body = json::parse(req.body);
                bool streaming = body.value("stream", false);

                if (!streaming) {
                    json response = {
                        {"id", "chatcmpl-test123"},
                        {"object", "chat.completion"},
                        {"choices", json::array({
                            {{"index", 0}, {"message", {{"role", "assistant"}, {"content", "Hello from mock!"}}}, {"finish_reason", "stop"}}
                        })}
                    };
                    res.set_content(response.dump(), "application/json");
                } else {
                    // SSE streaming response
                    std::string chunk1 = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n";
                    std::string chunk2 = "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"index\":0}]}\n\n";
                    std::string done = "data: [DONE]\n\n";
                    std::string full = chunk1 + chunk2 + done;
                    res.set_content(full, "text/event-stream");
                }
            });

            svr_.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
                json response = {{"data", json::array({
                    {{"id", "test-model"}, {"object", "model"}}
                })}};
                res.set_content(response.dump(), "application/json");
            });

            svr_.listen("0.0.0.0", BACKEND_PORT);
        });

        // Wait for server to be ready
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void stop() {
        svr_.stop();
        if (thread_.joinable()) thread_.join();
    }

    int request_count() const { return request_count_; }
    std::string last_body() const { return last_body_; }

private:
    httplib::Server svr_;
    std::thread thread_;
    std::atomic<int> request_count_{0};
    std::string last_body_;
};

// --------------- Proxy Under Test ---------------
class TestProxy {
public:
    TestProxy() {
        std::remove("test_integration.sqlite");

        cfg_.port = PROXY_PORT;
        cfg_.backend_url = "http://localhost:" + std::to_string(BACKEND_PORT);
        cfg_.db_path = "test_integration.sqlite";
        cfg_.top_k = 3;
        cfg_.decay_days = 30;

        store_ = std::make_unique<memorylayer::MemoryStore>(cfg_.db_path);
        // Construct embedding worker on main thread (llama needs main thread for Metal init)
        // Non-existent model → ready_=false, embed() returns {} → no injection
        embedder_ = std::make_unique<memorylayer::EmbeddingWorker>("/nonexistent_model.gguf", 0);
    }

    void start() {
        thread_ = std::thread([this] {
            memorylayer::MemoryProxy proxy(cfg_, *store_, *embedder_);
            proxy_ = &proxy;
            proxy.run();
        });

        // Wait for proxy to be ready
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }

    void stop() {
        if (proxy_) proxy_->stop();
        if (thread_.joinable()) thread_.join();
    }

    memorylayer::MemoryStore& store() { return *store_; }

private:
    memorylayer::Config cfg_;
    std::unique_ptr<memorylayer::MemoryStore> store_;
    std::unique_ptr<memorylayer::EmbeddingWorker> embedder_;
    std::thread thread_;
    memorylayer::MemoryProxy* proxy_ = nullptr;
};

// --------------- Tests ---------------

void test_health_endpoint() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    auto res = cli.Get("/health");
    assert(res);
    assert(res->status == 200);
    auto body = json::parse(res->body);
    assert(body["status"] == "ok");
    std::cout << "test_health_endpoint PASSED\n";
}

void test_admin_stats_empty() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    auto res = cli.Get("/admin/stats");
    assert(res);
    assert(res->status == 200);
    auto body = json::parse(res->body);
    assert(body["total_memories"] == 0);
    assert(body["total_agents"] == 0);
    std::cout << "test_admin_stats_empty PASSED\n";
}

void test_admin_list_memories_empty() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    auto res = cli.Get("/admin/memories");
    assert(res);
    assert(res->status == 200);
    auto body = json::parse(res->body);
    assert(body.is_array());
    assert(body.empty());
    std::cout << "test_admin_list_memories_empty PASSED\n";
}

void test_admin_debug_injection() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    auto res = cli.Get("/admin/debug/last-injection");
    assert(res);
    assert(res->status == 200);
    auto body = json::parse(res->body);
    assert(body.contains("last_agent_id"));
    assert(body.contains("last_query"));
    std::cout << "test_admin_debug_injection PASSED\n";
}

void test_chat_completions_passthrough() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    json request = {
        {"model", "test-model"},
        {"messages", json::array({
            {{"role", "user"}, {"content", "Hello!"}}
        })}
    };

    auto res = cli.Post("/v1/chat/completions", request.dump(), "application/json");
    assert(res);
    assert(res->status == 200);
    auto body = json::parse(res->body);
    assert(body["choices"][0]["message"]["content"] == "Hello from mock!");
    std::cout << "test_chat_completions_passthrough PASSED\n";
}

void test_chat_completions_with_agent_header() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    json request = {
        {"model", "test-model"},
        {"messages", json::array({
            {{"role", "user"}, {"content", "How are you?"}}
        })}
    };

    httplib::Headers headers = {{"X-Agent-Id", "test-agent"}};
    auto res = cli.Post("/v1/chat/completions", headers, request.dump(), "application/json");
    assert(res);
    assert(res->status == 200);
    std::cout << "test_chat_completions_with_agent_header PASSED\n";
}

void test_invalid_json_returns_400() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    auto res = cli.Post("/v1/chat/completions", "not valid json {{{{", "application/json");
    assert(res);
    assert(res->status == 400);
    auto body = json::parse(res->body);
    assert(body["error"] == "Invalid JSON");
    std::cout << "test_invalid_json_returns_400 PASSED\n";
}

void test_missing_messages_returns_400() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    json request = {{"model", "test-model"}};  // no "messages" key
    auto res = cli.Post("/v1/chat/completions", request.dump(), "application/json");
    assert(res);
    assert(res->status == 400);
    auto body = json::parse(res->body);
    assert(body["error"] == "Missing messages array");
    std::cout << "test_missing_messages_returns_400 PASSED\n";
}

void test_body_too_large_returns_413() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    std::string huge_body(2 * 1024 * 1024, 'x');  // 2MB
    auto res = cli.Post("/v1/chat/completions", huge_body, "application/json");
    assert(res);
    assert(res->status == 413);
    auto body = json::parse(res->body);
    assert(body["error"] == "Request body too large");
    std::cout << "test_body_too_large_returns_413 PASSED\n";
}

void test_delete_nonexistent_returns_404() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    auto res = cli.Delete("/admin/memories/99999");
    assert(res);
    assert(res->status == 404);
    auto body = json::parse(res->body);
    assert(body["error"] == "Memory not found");
    std::cout << "test_delete_nonexistent_returns_404 PASSED\n";
}

void test_passthrough_other_v1_endpoints() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    auto res = cli.Get("/v1/models");
    assert(res);
    assert(res->status == 200);
    auto body = json::parse(res->body);
    assert(body["data"][0]["id"] == "test-model");
    std::cout << "test_passthrough_other_v1_endpoints PASSED\n";
}

void test_admin_memories_pagination() {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    // With bad limit param - should use default (no crash)
    auto res = cli.Get("/admin/memories?limit=abc&offset=xyz");
    assert(res);
    assert(res->status == 200);
    std::cout << "test_admin_memories_pagination PASSED\n";
}

// --------------- Main ---------------
int main() {
    MockBackend backend;
    backend.start();

    TestProxy proxy;
    proxy.start();

    // Give everything time to settle
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    test_health_endpoint();
    test_admin_stats_empty();
    test_admin_list_memories_empty();
    test_admin_debug_injection();
    test_chat_completions_passthrough();
    test_chat_completions_with_agent_header();
    test_invalid_json_returns_400();
    test_missing_messages_returns_400();
    test_body_too_large_returns_413();
    test_delete_nonexistent_returns_404();
    test_passthrough_other_v1_endpoints();
    test_admin_memories_pagination();

    proxy.stop();
    backend.stop();

    std::remove("test_integration.sqlite");
    std::cout << "All integration tests PASSED (12 tests)\n";
    return 0;
}
