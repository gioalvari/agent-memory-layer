#pragma once
#include <iostream>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <sstream>

namespace memorylayer {

enum class LogLevel { DEBUG, INFO, WARN, ERROR };

inline const char* level_str(LogLevel lvl) {
    switch (lvl) {
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
    }
    return "?????";
}

inline void log(LogLevel lvl, const std::string& component, const std::string& msg) {
    static std::mutex log_mutex;
    std::lock_guard<std::mutex> lock(log_mutex);
    auto now = std::time(nullptr);
    std::tm tm{};
    localtime_r(&now, &tm);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);
    
    // Keep all application logs on one stream. Mixing stdout and stderr in a
    // redirected process lets their independent buffers interleave records.
    auto& out = std::cerr;
    out << buf << " [" << level_str(lvl) << "] [" << component << "] " << msg << "\n";
}

#define LOG_INFO(comp, msg)  memorylayer::log(memorylayer::LogLevel::INFO, comp, msg)
#define LOG_WARN(comp, msg)  memorylayer::log(memorylayer::LogLevel::WARN, comp, msg)
#define LOG_ERROR(comp, msg) memorylayer::log(memorylayer::LogLevel::ERROR, comp, msg)

} // namespace memorylayer
