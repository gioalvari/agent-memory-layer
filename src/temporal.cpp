#include "memorylayer/temporal.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <ctime>
#include <limits>
#include <regex>
#include <sstream>

namespace memorylayer {
namespace {

constexpr std::array<const char*, 12> kMonths = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
};

std::string lowercase(const std::string& value) {
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return lower;
}

// Howard Hinnant's civil-calendar conversion, with 1970-01-01 as day zero.
int64_t days_from_civil(int year, unsigned month, unsigned day) {
    year -= month <= 2;
    const int era = (year >= 0 ? year : year - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(year - era * 400);
    const unsigned doy = (153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int>(doe) - 719468;
}

struct CivilDate { int year; unsigned month; unsigned day; };

CivilDate civil_from_days(int64_t days) {
    days += 719468;
    const int era = (days >= 0 ? days : days - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(days - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const int year = static_cast<int>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    const unsigned day = doy - (153 * mp + 2) / 5 + 1;
    const unsigned month = mp + (mp < 10 ? 3 : -9);
    return {year + (month <= 2), month, day};
}

bool is_leap_year(int year) {
    return year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
}

unsigned days_in_month(int year, unsigned month) {
    static constexpr unsigned kDays[] = {31, 28, 31, 30, 31, 30,
                                          31, 31, 30, 31, 30, 31};
    return month == 2 && is_leap_year(year) ? 29 : kDays[month - 1];
}

std::optional<unsigned> month_number(const std::string& name) {
    const std::string lower = lowercase(name);
    for (unsigned index = 0; index < kMonths.size(); ++index) {
        if (lower == kMonths[index]) return index + 1;
    }
    return std::nullopt;
}

std::optional<int64_t> valid_day(int year, unsigned month, unsigned day) {
    if (month < 1 || month > 12 || day < 1 || day > days_in_month(year, month)) {
        return std::nullopt;
    }
    return days_from_civil(year, month, day);
}

std::optional<int64_t> parse_date(const std::string& text, int default_year) {
    std::smatch match;
    static const std::regex kIso(R"(^\s*(\d{4})-(\d{2})-(\d{2})\s*$)");
    static const std::regex kMonthDay(R"(^\s*([A-Za-z]+)\s+(\d{1,2})(?:,?\s*(\d{4}))?\s*$)");
    static const std::regex kDayMonth(R"(^\s*(\d{1,2})\s+([A-Za-z]+)(?:\s+(\d{4}))?\s*$)");
    if (std::regex_match(text, match, kIso)) {
        return valid_day(std::stoi(match[1]), static_cast<unsigned>(std::stoi(match[2])),
                         static_cast<unsigned>(std::stoi(match[3])));
    }
    if (std::regex_match(text, match, kMonthDay)) {
        const auto month = month_number(match[1]);
        if (!month) return std::nullopt;
        const int year = match[3].matched ? std::stoi(match[3]) : default_year;
        return valid_day(year, *month, static_cast<unsigned>(std::stoi(match[2])));
    }
    if (std::regex_match(text, match, kDayMonth)) {
        const auto month = month_number(match[2]);
        if (!month) return std::nullopt;
        const int year = match[3].matched ? std::stoi(match[3]) : default_year;
        return valid_day(year, *month, static_cast<unsigned>(std::stoi(match[1])));
    }
    return std::nullopt;
}

TemporalWindow one_day(int64_t day) { return {day, day + 1}; }

int64_t monday_of_week(int64_t day) {
    // 1970-01-01 was Thursday; Monday is offset 4 under this representation.
    const int64_t weekday = (day + 3) % 7;
    return day - (weekday < 0 ? weekday + 7 : weekday);
}

int64_t subtract_months(int64_t day, int months) {
    CivilDate date = civil_from_days(day);
    int total = date.year * 12 + static_cast<int>(date.month) - 1 - months;
    int year = total / 12;
    int month = total % 12;
    if (month < 0) { month += 12; --year; }
    const unsigned target_month = static_cast<unsigned>(month + 1);
    return days_from_civil(year, target_month,
                           std::min(date.day, days_in_month(year, target_month)));
}

std::optional<int64_t> first_date_in_text(const std::string& text, int year) {
    std::smatch match;
    static const std::regex kDate(R"((\d{4}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2}(?:,?\s+\d{4})?|\d{1,2}\s+[A-Za-z]+(?:\s+\d{4})?))",
                                  std::regex::icase);
    if (!std::regex_search(text, match, kDate)) return std::nullopt;
    return parse_date(match[1], year);
}

} // namespace

int64_t utc_day(double unix_seconds) {
    return static_cast<int64_t>(std::floor(unix_seconds / 86400.0));
}

std::string utc_date(double unix_seconds) {
    const CivilDate date = civil_from_days(utc_day(unix_seconds));
    std::ostringstream result;
    result.fill('0');
    result.width(4);
    result << date.year << '-';
    result.width(2);
    result << date.month << '-';
    result.width(2);
    result << date.day;
    return result.str();
}

std::optional<TemporalWindow> parse_temporal_window(const std::string& query,
                                                     double now_unix) {
    const std::string text = lowercase(query);
    const int64_t today = utc_day(now_unix);
    const CivilDate now_date = civil_from_days(today);
    std::smatch match;

    if (std::regex_search(text, std::regex(R"(\btoday\b)"))) return one_day(today);
    if (std::regex_search(text, std::regex(R"(\byesterday\b)"))) return one_day(today - 1);
    if (std::regex_search(text, match, std::regex(R"(\b(\d+)\s+(days?|weeks?|months?)\s+ago\b)"))) {
        const int count = std::stoi(match[1]);
        const std::string unit = match[2];
        if (unit[0] == 'd') return one_day(today - count);
        if (unit[0] == 'w') return one_day(today - 7LL * count);
        return one_day(subtract_months(today, count));
    }
    if (std::regex_search(text, std::regex(R"(\blast\s+week\b)"))) {
        const int64_t start = monday_of_week(today) - 7;
        return TemporalWindow{start, start + 7};
    }
    if (std::regex_search(text, std::regex(R"(\bthis\s+week\b)"))) {
        const int64_t start = monday_of_week(today);
        return TemporalWindow{start, start + 7};
    }
    if (std::regex_search(text, std::regex(R"(\blast\s+month\b)"))) {
        const int64_t end = days_from_civil(now_date.year, now_date.month, 1);
        return TemporalWindow{subtract_months(end, 1), end};
    }
    if (std::regex_search(text, std::regex(R"(\bthis\s+month\b)"))) {
        const int64_t start = days_from_civil(now_date.year, now_date.month, 1);
        return TemporalWindow{start, days_from_civil(now_date.year + (now_date.month == 12),
                                                      now_date.month == 12 ? 1 : now_date.month + 1, 1)};
    }
    if (std::regex_search(text, std::regex(R"(\blast\s+year\b)"))) {
        const int64_t end = days_from_civil(now_date.year, 1, 1);
        return TemporalWindow{days_from_civil(now_date.year - 1, 1, 1), end};
    }
    if (std::regex_search(text, std::regex(R"(\bthis\s+year\b)"))) {
        return TemporalWindow{days_from_civil(now_date.year, 1, 1), days_from_civil(now_date.year + 1, 1, 1)};
    }
    if (std::regex_search(text, match, std::regex(R"(\b(since|before)\s+(.+))"))) {
        const auto date = first_date_in_text(match[2], now_date.year);
        if (date) {
            if (match[1] == "since") return TemporalWindow{*date, std::numeric_limits<int64_t>::max()};
            return TemporalWindow{std::numeric_limits<int64_t>::min(), *date};
        }
    }
    if (std::regex_search(text, match,
                          std::regex(R"(\bin\s+([a-z]+)(?:\s+(\d{4}))?\b)"))) {
        const auto month = month_number(match[1]);
        if (month) {
            const int year = match[2].matched ? std::stoi(match[2]) : now_date.year;
            const int64_t start = days_from_civil(year, *month, 1);
            return TemporalWindow{start, days_from_civil(year + (*month == 12), *month == 12 ? 1 : *month + 1, 1)};
        }
    }
    if (std::regex_search(text, match,
                          std::regex(R"(\bon\s+([a-z]+\s+\d{1,2}(?:,?\s+\d{4})?|\d{1,2}\s+[a-z]+(?:\s+\d{4})?)\b)"))) {
        const auto date = parse_date(match[1], now_date.year);
        if (date) return one_day(*date);
    }
    const auto date = first_date_in_text(text, now_date.year);
    return date ? std::optional<TemporalWindow>(one_day(*date)) : std::nullopt;
}

std::vector<int64_t> extract_mentioned_days(const std::string& text,
                                            double reference_unix) {
    const int64_t reference_day = utc_day(reference_unix);
    const int reference_year = civil_from_days(reference_day).year;
    std::vector<int64_t> days;
    const std::string lower = lowercase(text);
    if (std::regex_search(lower, std::regex(R"(\btoday\b)"))) days.push_back(reference_day);
    if (std::regex_search(lower, std::regex(R"(\byesterday\b)"))) days.push_back(reference_day - 1);

    static const std::regex kDates(
        R"((\d{4}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4}))",
        std::regex::icase);
    for (std::sregex_iterator it(text.begin(), text.end(), kDates), end; it != end; ++it) {
        if (const auto day = parse_date((*it)[1], reference_year)) days.push_back(*day);
    }
    std::sort(days.begin(), days.end());
    days.erase(std::unique(days.begin(), days.end()), days.end());
    return days;
}

} // namespace memorylayer
