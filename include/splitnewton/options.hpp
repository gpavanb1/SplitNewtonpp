#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

namespace splitnewton {

/**
 * @brief Singleton class to handle solver options.
 */
class Options {
public:
    static Options& getInstance() {
        static Options instance;
        return instance;
    }

    /**
     * @brief Initialize options from command line arguments.
     * @param argc Argument count.
     * @param argv Argument vector.
     */
    void initialize(int argc, char** argv) {
        args_.clear();
        for (int i = 0; i < argc; ++i) {
            args_.push_back(std::string(argv[i]));
        }
    }

    /**
     * @brief Check if a flag exists.
     * @param flag Flag to check (e.g., "-use_jacobi").
     * @return true if flag exists.
     */
    bool hasFlag(const std::string& flag) {
        // 1. Check command line args
        if (std::find(args_.begin(), args_.end(), flag) != args_.end()) 
            return true;

        // 2. Also check environment variables (transparently)
        // Convert -use_jacobi -> USE_JACOBI
        std::string env_name = flag;
        if (!env_name.empty() && env_name[0] == '-') env_name = env_name.substr(1);
        std::transform(env_name.begin(), env_name.end(), env_name.begin(), ::toupper);
        
        const char* env_val = std::getenv(env_name.c_str());
        if (env_val != nullptr) {
            std::string val(env_val);
            return (val == "1" || val == "true" || val == "TRUE" || val == "ON");
        }

        return false;
    }

private:
    Options() = default;
    std::vector<std::string> args_;
};

/**
 * @brief Helper function to initialize splitnewton options.
 */
inline void initialize(int argc, char** argv) {
    Options::getInstance().initialize(argc, argv);
}

} // namespace splitnewton

#endif // OPTIONS_HPP
