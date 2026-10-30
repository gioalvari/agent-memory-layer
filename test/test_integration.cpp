/**
 * Integration test for the Memory Proxy HTTP endpoints.
 *
 * Spins up a mock LLM backend and the real proxy, then validates every HTTP
 * endpoint end-to-end. Fact tests use an injected deterministic embedder.
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
#include <functional>
#include <mutex>
#include <vector>

using json = nlohmann::json;

static constexpr int BACKEND_PORT = 19999;
static constexpr int PROXY_PORT = 19900;
static constexpr int FACT_PROXY_PORT = 19901;

template <typename Predicate>
bool wait_until(Predicate predicate, std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return predicate();
}

// --------------- Mock LLM Backend ---------------
class MockBackend {
public:
    void start() {
        thread_ = std::thread([this] {
            svr_.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
                auto body = json::parse(req.body);
                const bool extraction = body.value("temperature", -1) == 0 &&
                    body.contains("messages") && !body["messages"].empty() &&
                    body["messages"][0].value("content", "").find("Extract only durable facts") != std::string::npos;
                {
                    std::lock_guard<std::mutex> lock(request_mutex_);
                    request_count_++;
                    request_bodies_.push_back(req.body);
                    if (extraction) extraction_count_++;
                    else last_normal_body_ = req.body;
                }
                bool streaming = body.value("stream", false);

                if (extraction) {
                    const std::string prompt = body["messages"][0]["content"].get<std::string>();
                    std::string facts = "[]";
                    if (prompt.find("I prefer vim") != std::string::npos) {
                        facts = R"([{"subject":"user","attribute":"favorite editor","value":"vim"}])";
                    } else if (prompt.find("I now prefer helix") != std::string::npos) {
                        facts = R"([{"subject":"user","attribute":"favorite editor","value":"helix"}])";
                    }
                    json response = {{"choices", json::array({{{"message", {{"role", "assistant"}, {"content", facts}}}}})}};
                    res.set_content(response.dump(), "application/json");
                } else if (!streaming) {
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

        assert(wait_until([this] {
            httplib::Client cli("http://localhost:" + std::to_string(BACKEND_PORT));
            const auto response = cli.Get("/v1/models");
            return response && response->status == 200;
        }));
    }

    void stop() {
        svr_.stop();
        if (thread_.joinable()) thread_.join();
    }

    int request_count() const { return request_count_; }
    int extraction_count() const { return extraction_count_; }
    std::string last_normal_body() const {
        std::lock_guard<std::mutex> lock(request_mutex_);
        return last_normal_body_;
    }
    std::vector<std::string> request_bodies() const {
        std::lock_guard<std::mutex> lock(request_mutex_);
        return request_bodies_;
    }
    void reset_requests() {
        std::lock_guard<std::mutex> lock(request_mutex_);
        request_count_ = 0;
        extraction_count_ = 0;
        request_bodies_.clear();
        last_normal_body_.clear();
    }

private:
    httplib::Server svr_;
    std::thread thread_;
    std::atomic<int> request_count_{0};
    std::atomic<int> extraction_count_{0};
    mutable std::mutex request_mutex_;
    std::vector<std::string> request_bodies_;
    std::string last_normal_body_;
};

// --------------- Proxy Under Test ---------------
class TestProxy {
public:
    TestProxy(int port = PROXY_PORT, bool extract_facts = false,
              memorylayer::MemoryProxy::EmbedHook embed_hook = {})
        : embed_hook_(std::move(embed_hook)) {
        db_path_ = "test_integration_" + std::to_string(port) + ".sqlite";
        std::remove(db_path_.c_str());

        cfg_.port = port;
        cfg_.backend_url = "http://localhost:" + std::to_string(BACKEND_PORT);
        cfg_.db_path = db_path_;
        cfg_.top_k = 3;
        cfg_.decay_days = 30;
        cfg_.history_budget = 80;
        cfg_.extract_facts = extract_facts;
        if (extract_facts) cfg_.dedup_threshold = 1.1f;

        store_ = std::make_unique<memorylayer::MemoryStore>(cfg_.db_path);
        // Construct embedding worker on main thread (llama needs main thread for Metal init)
        // Non-existent model → ready_=false, embed() returns {} → no injection
        embedder_ = std::make_unique<memorylayer::EmbeddingWorker>("/nonexistent_model.gguf", 0);
    }

    void start() {
        thread_ = std::thread([this] {
            memorylayer::MemoryProxy proxy(cfg_, *store_, *embedder_, embed_hook_);
            proxy_ = &proxy;
            proxy.run();
        });

        assert(wait_until([this] {
            httplib::Client cli("http://localhost:" + std::to_string(cfg_.port));
            const auto response = cli.Get("/health");
            return response && response->status == 200;
        }));
    }

    void stop() {
        if (proxy_) proxy_->stop();
        if (thread_.joinable()) thread_.join();
    }

    memorylayer::MemoryStore& store() { return *store_; }
    int port() const { return cfg_.port; }
    const std::string& db_path() const { return db_path_; }

private:
    memorylayer::Config cfg_;
    std::unique_ptr<memorylayer::MemoryStore> store_;
    std::unique_ptr<memorylayer::EmbeddingWorker> embedder_;
    memorylayer::MemoryProxy::EmbedHook embed_hook_;
    std::string db_path_;
    std::thread thread_;
    memorylayer::MemoryProxy* proxy_ = nullptr;
};

std::vector<float> deterministic_embedding(const std::string&, memorylayer::EmbedPriority) {
    return {1.0f, 0.0f, 0.0f, 0.0f};
}

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
    assert(body.is_array());  // injection ring, newest first
    for (const auto& item : body) {
        assert(item.contains("agent_id"));
        assert(item.contains("query"));
        assert(item.contains("injected_context"));
    }
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

void test_chat_completions_trims_history(const MockBackend& backend) {
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    json request = {
        {"model", "test-model"},
        {"messages", json::array({
            {{"role", "system"}, {"content", "Keep this instruction."}},
            {{"role", "user"}, {"content", std::string(80, 'a')}},
            {{"role", "assistant"}, {"content", std::string(80, 'b')}},
            {{"role", "user"}, {"content", std::string(80, 'c')}},
            {{"role", "assistant"}, {"content", std::string(80, 'd')}},
            {{"role", "user"}, {"content", "latest request"}}
        })}
    };

    auto res = cli.Post("/v1/chat/completions", request.dump(), "application/json");
    assert(res);
    assert(res->status == 200);
    const auto forwarded = json::parse(backend.last_body());
    const auto& messages = forwarded["messages"];
    assert(messages.size() == 2);
    assert(messages[0]["role"] == "system");
    assert(messages[0]["content"].get<std::string>().find(
               "4 earlier messages of this conversation were omitted") != std::string::npos);
    assert(messages[1]["role"] == "user");
    assert(messages[1]["content"] == "latest request");
    std::cout << "test_chat_completions_trims_history PASSED\n";
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

void test_admin_facts_endpoints(TestProxy& proxy) {
    const std::vector<float> embedding(4, 0.5f);
    const auto id = proxy.store().insert_fact("test-agent", "Project", "language", "C++", 1, embedding);
    assert(id > 0);
    httplib::Client cli("http://localhost:" + std::to_string(PROXY_PORT));
    auto listed = cli.Get("/admin/facts?agent_id=test-agent&include_superseded=0&limit=5");
    assert(listed && listed->status == 200);
    const auto facts = json::parse(listed->body);
    assert(facts.size() == 1 && facts[0]["value"] == "C++");
    auto deleted = cli.Delete("/admin/facts/" + std::to_string(id));
    assert(deleted && deleted->status == 200);
    auto missing = cli.Delete("/admin/facts/" + std::to_string(id));
    assert(missing && missing->status == 404);
    std::cout << "test_admin_facts_endpoints PASSED\n";
}

json post_chat(int port, const std::string& content, const std::string& agent_id = "fact-agent") {
    httplib::Client cli("http://localhost:" + std::to_string(port));
    const json request = {{"model", "test-model"}, {"messages", json::array({{{"role", "user"}, {"content", content}}})}};
    const auto response = cli.Post("/v1/chat/completions", {{"X-Agent-Id", agent_id}}, request.dump(), "application/json");
    assert(response && response->status == 200);
    return json::parse(response->body);
}

json list_facts(int port, bool include_superseded = false) {
    httplib::Client cli("http://localhost:" + std::to_string(port));
    const auto response = cli.Get("/admin/facts?agent_id=fact-agent&include_superseded=" +
                                  std::string(include_superseded ? "1" : "0"));
    assert(response && response->status == 200);
    return json::parse(response->body);
}

void test_fact_extraction_end_to_end(MockBackend& backend) {
    backend.reset_requests();
    TestProxy proxy(FACT_PROXY_PORT, true, deterministic_embedding);
    proxy.start();

    post_chat(proxy.port(), "I prefer vim for editing.");
    assert(wait_until([&] {
        const auto facts = list_facts(proxy.port());
        return facts.size() == 1 && facts[0]["value"] == "vim";
    }));
    assert(backend.extraction_count() == 1);

    post_chat(proxy.port(), "What editor do I use?");
    assert(backend.last_normal_body().find("favorite editor: vim") != std::string::npos);

    post_chat(proxy.port(), "I now prefer helix for editing.");
    assert(wait_until([&] {
        const auto facts = list_facts(proxy.port(), true);
        return facts.size() == 2 && facts[0]["value"] == "helix" && !facts[0]["superseded_by"].is_number() &&
               facts[1]["value"] == "vim" && facts[1]["superseded_by"].is_number();
    }));

    post_chat(proxy.port(), "Which editor is my favorite now?");
    const auto injected = backend.last_normal_body();
    assert(injected.find("favorite editor: helix") != std::string::npos);
    assert(injected.find("favorite editor: vim") == std::string::npos);

    assert(wait_until([&] { return proxy.store().count("fact-agent") == 4; }));
    const auto memories = proxy.store().list_memories("fact-agent", 20);
    assert(memories.size() == 4);
    for (const auto& body : backend.request_bodies()) {
        const auto request = json::parse(body);
        const bool extraction = request.value("temperature", -1) == 0;
        if (extraction) {
            assert(request["messages"].size() == 1);
            assert(request["messages"][0]["content"].get<std::string>().find("<memory context>") == std::string::npos);
        }
    }
    proxy.stop();
    std::remove(proxy.db_path().c_str());
    std::cout << "test_fact_extraction_end_to_end PASSED\n";
}

void test_fact_extraction_disabled_makes_no_backend_calls(MockBackend& backend) {
    backend.reset_requests();
    TestProxy proxy(FACT_PROXY_PORT, false, deterministic_embedding);
    proxy.start();
    post_chat(proxy.port(), "One ordinary turn.");
    post_chat(proxy.port(), "A second ordinary turn.");
    assert(wait_until([&] { return proxy.store().count("fact-agent") >= 1; }));
    assert(backend.extraction_count() == 0);
    proxy.stop();
    std::remove(proxy.db_path().c_str());
    std::cout << "test_fact_extraction_disabled_makes_no_backend_calls PASSED\n";
}

// --------------- Main ---------------
int main() {
    MockBackend backend;
    backend.start();

    TestProxy proxy;
    proxy.start();

    test_health_endpoint();
    test_admin_stats_empty();
    test_admin_list_memories_empty();
    test_admin_debug_injection();
    test_chat_completions_passthrough();
    test_chat_completions_with_agent_header();
    test_chat_completions_trims_history(backend);
    test_invalid_json_returns_400();
    test_missing_messages_returns_400();
    test_body_too_large_returns_413();
    test_delete_nonexistent_returns_404();
    test_passthrough_other_v1_endpoints();
    test_admin_memories_pagination();
    test_admin_facts_endpoints(proxy);

    proxy.stop();

    test_fact_extraction_end_to_end(backend);
    test_fact_extraction_disabled_makes_no_backend_calls(backend);
    backend.stop();

    std::remove("test_integration.sqlite");
    std::cout << "All integration tests PASSED\n";
    return 0;
}
