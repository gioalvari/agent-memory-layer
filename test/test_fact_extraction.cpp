#include "memorylayer/fact_extraction.h"
#include <cassert>
#include <iostream>

int main() {
    auto fenced = memorylayer::parse_extracted_facts("```json\n[{\"subject\":\"User\",\"attribute\":\"language\",\"value\":\"C++\"}]\n```");
    assert(fenced.size() == 1 && fenced[0].value == "C++");
    auto garbage = memorylayer::parse_extracted_facts("note before [{\"subject\":\"Project\",\"attribute\":\"build\",\"value\":\"CMake\"}] note after");
    assert(garbage.size() == 1 && garbage[0].subject == "Project");
    assert(memorylayer::parse_extracted_facts("[]").empty());
    assert(memorylayer::parse_extracted_facts("[{\"subject\":5}]").empty());
    std::cout << "All fact extraction parser tests PASSED\n";
}
