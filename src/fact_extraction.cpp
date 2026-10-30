#include "memorylayer/fact_extraction.h"
#include "json.hpp"
#include <algorithm>

namespace memorylayer {
namespace { constexpr std::size_t kMaxPartLength = 400; }

std::vector<ExtractedFact> parse_extracted_facts(const std::string& content) {
    const std::size_t begin = content.find('[');
    const std::size_t end = content.rfind(']');
    if (begin == std::string::npos || end == std::string::npos || begin > end) return {};
    try {
        const auto array = nlohmann::json::parse(content.substr(begin, end - begin + 1));
        if (!array.is_array()) return {};
        std::vector<ExtractedFact> facts;
        for (const auto& item : array) {
            if (!item.is_object() || !item.contains("subject") || !item.contains("attribute") ||
                !item.contains("value") || !item["subject"].is_string() ||
                !item["attribute"].is_string() || !item["value"].is_string()) continue;
            ExtractedFact fact{item["subject"].get<std::string>(), item["attribute"].get<std::string>(), item["value"].get<std::string>()};
            if (fact.subject.empty() || fact.attribute.empty() || fact.value.empty()) continue;
            fact.subject.resize(std::min(fact.subject.size(), kMaxPartLength));
            fact.attribute.resize(std::min(fact.attribute.size(), kMaxPartLength));
            fact.value.resize(std::min(fact.value.size(), kMaxPartLength));
            facts.push_back(std::move(fact));
            if (facts.size() == 10) break;
        }
        return facts;
    } catch (...) { return {}; }
}

std::string fact_extraction_prompt(const std::string& user_text, const std::string& assist_text) {
    return "Extract only durable facts about the user or project from this single exchange. "
           "Return ONLY a JSON array of objects with subject, attribute, and value string fields; "
           "return [] if none. Do not infer uncertain facts.\nUser: " + user_text +
           "\nAssistant: " + assist_text;
}
} // namespace memorylayer
