#pragma once
#include <iostream>
#include <ctime>
#include <iomanip>
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
    auto now = std::time(nullptr);
    auto* tm = std::localtime(&now);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", tm);
    
    auto& out = (lvl >= LogLevel::WARN) ? std::cerr : std::cout;
    out << buf << " [" << level_str(lvl) << "] [" << component << "] " << msg << "\n";
}

#define LOG_INFO(comp, msg)  memorylayer::log(memorylayer::LogLevel::INFO, comp, msg)
#define LOG_WARN(comp, msg)  memorylayer::log(memorylayer::LogLevel::WARN, comp, msg)
#define LOG_ERROR(comp, msg) memorylayer::log(memorylayer::LogLevel::ERROR, comp, msg)

} // namespace memorylayer
