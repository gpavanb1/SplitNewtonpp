#ifndef HELPER_HPP
#define HELPER_HPP

#include <sstream>
#include <string>
#include <iomanip>

inline std::string to_scientific(double value, int precision = 6)
{
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(precision) << value;
    return stream.str();
}

#endif // HELPER_HPP