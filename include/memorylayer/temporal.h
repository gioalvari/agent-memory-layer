#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace memorylayer {

// UTC calendar-day interval, expressed as whole days since 1970-01-01. The
// end is exclusive. All temporal parsing and stored dates use UTC.
struct TemporalWindow {
    int64_t start_day;
    int64_t end_day;
};

int64_t utc_day(double unix_seconds);
std::string utc_date(double unix_seconds);

// Deterministic English-only query parser. It recognizes: today, yesterday,
// N days/weeks/months ago, last/this week/month/year, in <month> [year], on
// <month> <day>[, year], ISO dates, since <date>, and before <date>. `now` is
// supplied by the caller so tests and callers can choose one UTC reference.
std::optional<TemporalWindow> parse_temporal_window(const std::string& query,
                                                     double now_unix);

// Extracts explicit ISO and English dates plus today/yesterday. Relative words
// are resolved against `reference_unix`, normally the memory's creation time.
std::vector<int64_t> extract_mentioned_days(const std::string& text,
                                            double reference_unix);

} // namespace memorylayer
