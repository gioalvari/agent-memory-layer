#include "memorylayer/temporal.h"

#include <cassert>
#include <iostream>
#include <limits>

using memorylayer::TemporalWindow;

namespace {

constexpr double kNow = 1709596800.0;  // 2024-03-05 00:00:00 UTC

void expect_window(const char* query, TemporalWindow expected) {
    const auto actual = memorylayer::parse_temporal_window(query, kNow);
    assert(actual);
    assert(actual->start_day == expected.start_day);
    assert(actual->end_day == expected.end_day);
}

void test_query_parser() {
    const auto mar5 = memorylayer::utc_day(kNow);
    expect_window("today", {mar5, mar5 + 1});
    expect_window("yesterday", {mar5 - 1, mar5});
    expect_window("3 days ago", {mar5 - 3, mar5 - 2});
    expect_window("2 weeks ago", {mar5 - 14, mar5 - 13});
    expect_window("1 month ago", {mar5 - 29, mar5 - 28});
    expect_window("last week", {mar5 - 8, mar5 - 1});
    expect_window("this week", {mar5 - 1, mar5 + 6});
    expect_window("last month", {mar5 - 33, mar5 - 4});
    expect_window("this month", {mar5 - 4, mar5 + 27});
    expect_window("last year", {memorylayer::utc_day(1672531200.0), memorylayer::utc_day(1704067200.0)});
    expect_window("in February 2024", {mar5 - 33, mar5 - 4});
    expect_window("on March 3, 2024", {mar5 - 2, mar5 - 1});
    expect_window("2024-02-29", {mar5 - 5, mar5 - 4});
    expect_window("since 2024-02-29", {mar5 - 5, std::numeric_limits<int64_t>::max()});
    expect_window("before March 1, 2024", {std::numeric_limits<int64_t>::min(), mar5 - 4});
    assert(!memorylayer::parse_temporal_window("tell me about vectors", kNow));
    assert(!memorylayer::parse_temporal_window("2023-02-29", kNow));
    std::cout << "test_query_parser PASSED\n";
}

void test_date_extraction() {
    const auto days = memorylayer::extract_mentioned_days(
        "On March 3, 2024 and 4 March 2024, yesterday was busy. Also 2024-02-29.", kNow);
    // "yesterday" resolves to March 4 and therefore deduplicates with 4 March.
    assert(days.size() == 3);
    assert(days[0] == memorylayer::utc_day(1709164800.0));
    assert(days[1] == memorylayer::utc_day(1709424000.0));
    assert(days[2] == memorylayer::utc_day(1709510400.0));
    std::cout << "test_date_extraction PASSED\n";
}

} // namespace

int main() {
    test_query_parser();
    test_date_extraction();
    std::cout << "All temporal tests PASSED\n";
}
