#pragma once
#include <string>
#include <vector>

namespace memorylayer {
struct ExtractedFact { std::string subject; std::string attribute; std::string value; };
std::vector<ExtractedFact> parse_extracted_facts(const std::string& content);
std::string fact_extraction_prompt(const std::string& user_text, const std::string& assist_text);
} // namespace memorylayer
