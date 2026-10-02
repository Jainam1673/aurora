#include "aurora/checkpoint.hpp"
#include "aurora/version.hpp"

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <variant>

namespace aurora {
namespace checkpoint {

namespace {

// Minimal recursive-descent JSON parser and serializer
struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;
using JsonVariant = std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject>;

struct JsonValue {
    JsonVariant val;
    JsonValue() : val(nullptr) {}
    JsonValue(std::nullptr_t) : val(nullptr) {}
    JsonValue(bool b) : val(b) {}
    JsonValue(double d) : val(d) {}
    JsonValue(int i) : val(static_cast<double>(i)) {}
    JsonValue(size_t s) : val(static_cast<double>(s)) {}
    JsonValue(const char* s) : val(std::string(s)) {}
    JsonValue(std::string s) : val(std::move(s)) {}
    JsonValue(JsonArray a) : val(std::move(a)) {}
    JsonValue(JsonObject o) : val(std::move(o)) {}

    [[nodiscard]] bool is_null() const { return std::holds_alternative<std::nullptr_t>(val); }
    [[nodiscard]] bool is_bool() const { return std::holds_alternative<bool>(val); }
    [[nodiscard]] bool is_number() const { return std::holds_alternative<double>(val); }
    [[nodiscard]] bool is_string() const { return std::holds_alternative<std::string>(val); }
    [[nodiscard]] bool is_array() const { return std::holds_alternative<JsonArray>(val); }
    [[nodiscard]] bool is_object() const { return std::holds_alternative<JsonObject>(val); }

    [[nodiscard]] bool as_bool() const { return std::get<bool>(val); }
    [[nodiscard]] double as_number() const { return std::get<double>(val); }
    [[nodiscard]] const std::string& as_string() const { return std::get<std::string>(val); }
    [[nodiscard]] const JsonArray& as_array() const { return std::get<JsonArray>(val); }
    [[nodiscard]] const JsonObject& as_object() const { return std::get<JsonObject>(val); }
};

class JsonParser {
public:
    explicit JsonParser(std::string text) : text_(std::move(text)), pos_(0) {}

    JsonValue parse() {
        skip_whitespace();
        auto result = parse_value();
        skip_whitespace();
        if (pos_ < text_.size()) {
            throw std::runtime_error("Unexpected trailing character in JSON");
        }
        return result;
    }

private:
    std::string text_;
    size_t pos_;

    void skip_whitespace() {
        while (pos_ < text_.size() && (std::isspace(static_cast<unsigned char>(text_[pos_])) != 0)) {
            pos_++;
        }
    }

    char peek() {
        skip_whitespace();
        if (pos_ >= text_.size()) return '\0';
        return text_[pos_];
    }

    char get() {
        skip_whitespace();
        if (pos_ >= text_.size()) return '\0';
        return text_[pos_++];
    }

    JsonValue parse_value() {
        char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        if (c == '-' || (c >= '0' && c <= '9')) return parse_number();
        throw std::runtime_error(std::format("Unexpected character in JSON: '{}' at pos {}", c, pos_));
    }

    JsonObject parse_object() {
        JsonObject obj;
        get(); // consume '{'
        skip_whitespace();
        if (peek() == '}') {
            get();
            return obj;
        }

        while (true) {
            skip_whitespace();
            if (peek() != '"') {
                throw std::runtime_error("Expected string key in object");
            }
            std::string key = parse_string_raw();
            skip_whitespace();
            if (get() != ':') {
                throw std::runtime_error("Expected ':' after object key");
            }
            obj[key] = parse_value();
            skip_whitespace();
            char next = peek();
            if (next == '}') {
                get();
                break;
            }
            if (next == ',') {
                get();
            } else {
                throw std::runtime_error("Expected ',' or '}' in object");
            }
        }
        return obj;
    }

    JsonArray parse_array() {
        JsonArray arr;
        get(); // consume '['
        skip_whitespace();
        if (peek() == ']') {
            get();
            return arr;
        }

        while (true) {
            arr.push_back(parse_value());
            skip_whitespace();
            char next = peek();
            if (next == ']') {
                get();
                break;
            }
            if (next == ',') {
                get();
            } else {
                throw std::runtime_error("Expected ',' or ']' in array");
            }
        }
        return arr;
    }

    std::string parse_string_raw() {
        get(); // consume '"'
        std::string s;
        while (pos_ < text_.size()) {
            char c = text_[pos_++];
            if (c == '"') {
                return s;
            }
            if (c == '\\') {
                if (pos_ >= text_.size()) throw std::runtime_error("Unexpected escape at end of JSON");
                char esc = text_[pos_++];
                switch (esc) {
                    case '"': s.push_back('"'); break;
                    case '\\': s.push_back('\\'); break;
                    case '/': s.push_back('/'); break;
                    case 'b': s.push_back('\b'); break;
                    case 'f': s.push_back('\f'); break;
                    case 'n': s.push_back('\n'); break;
                    case 'r': s.push_back('\r'); break;
                    case 't': s.push_back('\t'); break;
                    default: s.push_back(esc); break;
                }
            } else {
                s.push_back(c);
            }
        }
        throw std::runtime_error("Unterminated string in JSON");
    }

    JsonValue parse_string() {
        return parse_string_raw();
    }

    JsonValue parse_bool() {
        if (text_.substr(pos_, 4) == "true") {
            pos_ += 4;
            return true;
        }
        if (text_.substr(pos_, 5) == "false") {
            pos_ += 5;
            return false;
        }
        throw std::runtime_error("Invalid boolean value in JSON");
    }

    JsonValue parse_null() {
        if (text_.substr(pos_, 4) == "null") {
            pos_ += 4;
            return nullptr;
        }
        throw std::runtime_error("Invalid null value in JSON");
    }

    JsonValue parse_number() {
        size_t start = pos_;
        if (text_[pos_] == '-') pos_++;
        while (pos_ < text_.size() && (std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0)) {
            pos_++;
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            pos_++;
            while (pos_ < text_.size() && (std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0)) {
                pos_++;
            }
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            pos_++;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                pos_++;
            }
            while (pos_ < text_.size() && (std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0)) {
                pos_++;
            }
        }
        std::string num_str(text_.substr(start, pos_ - start));
        return std::stod(num_str);
    }
};

void serialize_json(const JsonValue& v, std::ostream& os, int indent = 0) {
    std::string ind(static_cast<size_t>(indent * 2), ' ');
    if (v.is_null()) {
        os << "null";
    } else if (v.is_bool()) {
        os << (v.as_bool() ? "true" : "false");
    } else if (v.is_number()) {
        os << std::setprecision(17) << v.as_number();
    } else if (v.is_string()) {
        os << '"';
        for (char c : v.as_string()) {
            if (c == '"') os << "\\\"";
            else if (c == '\\') os << "\\\\";
            else if (c == '\n') os << "\\n";
            else if (c == '\t') os << "\\t";
            else os << c;
        }
        os << '"';
    } else if (v.is_array()) {
        const auto& arr = v.as_array();
        if (arr.empty()) {
            os << "[]";
            return;
        }
        // If array consists only of numbers/strings and is small, print compactly
        bool compact = arr.size() <= 8;
        for (const auto& item : arr) {
            if (!item.is_number() && !item.is_string()) {
                compact = false;
                break;
            }
        }
        if (compact) {
            os << "[";
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) os << ", ";
                serialize_json(arr[i], os, 0);
            }
            os << "]";
        } else {
            os << "[\n";
            for (size_t i = 0; i < arr.size(); ++i) {
                os << ind << "  ";
                serialize_json(arr[i], os, indent + 1);
                if (i + 1 < arr.size()) os << ",";
                os << "\n";
            }
            os << ind << "]";
        }
    } else if (v.is_object()) {
        const auto& obj = v.as_object();
        if (obj.empty()) {
            os << "{}";
            return;
        }
        os << "{\n";
        size_t count = 0;
        for (const auto& [k, val] : obj) {
            count++;
            os << ind << "  \"" << k << "\": ";
            serialize_json(val, os, indent + 1);
            if (count < obj.size()) os << ",";
            os << "\n";
        }
        os << ind << "}";
    }
}

} // namespace

void save_checkpoint(const std::string& filepath,
                     const nn::Module& model,
                     const optim::Optimizer* optimizer,
                     const std::map<std::string, std::string>& metadata) {
    std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    JsonObject root;

    // Metadata
    JsonObject meta_obj;
    meta_obj["aurora_version"] = std::string(kVersion);
    auto now = std::chrono::system_clock::now();
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    meta_obj["timestamp"] = static_cast<double>(secs);
    for (const auto& [k, v] : metadata) {
        meta_obj[k] = v;
    }
    root["metadata"] = meta_obj;

    // Model State Dict
    JsonObject model_obj;
    for (const auto& [name, param] : model.named_parameters("", true)) {
        JsonObject entry;
        JsonArray shape_arr;
        for (size_t dim : param->shape()) {
            shape_arr.emplace_back(static_cast<double>(dim));
        }
        entry["shape"] = shape_arr;

        JsonArray data_arr;
        auto data_vec = param->to_vector();
        data_arr.reserve(data_vec.size());
        for (double val : data_vec) {
            data_arr.emplace_back(val);
        }
        entry["data"] = data_arr;

        model_obj[name] = entry;
    }
    root["model_state_dict"] = model_obj;

    // Optimizer State Dict
    if (optimizer != nullptr) {
        JsonObject optim_obj;
        auto s = optimizer->state_dict();
        optim_obj["step_count"] = static_cast<double>(s.step_count);

        JsonObject defaults_obj;
        for (const auto& [k, v] : s.defaults) {
            defaults_obj[k] = v;
        }
        optim_obj["defaults"] = defaults_obj;

        JsonObject state_map_obj;
        for (const auto& [param_idx, p_state] : s.state) {
            JsonObject p_obj;
            for (const auto& [t_name, t_val] : p_state.tensors) {
                JsonObject t_entry;
                JsonArray shape_arr;
                for (size_t dim : t_val->shape()) {
                    shape_arr.emplace_back(static_cast<double>(dim));
                }
                t_entry["shape"] = shape_arr;

                JsonArray data_arr;
                auto d_vec = t_val->to_vector();
                data_arr.reserve(d_vec.size());
                for (double val : d_vec) {
                    data_arr.emplace_back(val);
                }
                t_entry["data"] = data_arr;

                p_obj[t_name] = t_entry;
            }
            state_map_obj[std::to_string(param_idx)] = p_obj;
        }
        optim_obj["state"] = state_map_obj;
        root["optimizer_state_dict"] = optim_obj;
    }

    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        throw std::runtime_error(std::format("Failed to open checkpoint file for writing: {}", filepath));
    }
    serialize_json(root, ofs, 0);
    ofs << "\n";
}

CheckpointData load_checkpoint(const std::string& filepath,
                               nn::Module* model,
                               optim::Optimizer* optimizer) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error(std::format("Failed to open checkpoint file for reading: {}", filepath));
    }
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    JsonParser parser(std::move(content));
    JsonValue root = parser.parse();

    if (!root.is_object()) {
        throw std::runtime_error("Malformed checkpoint: root must be a JSON object");
    }

    const auto& root_obj = root.as_object();
    CheckpointData result;

    // Metadata
    if (root_obj.contains("metadata") && root_obj.at("metadata").is_object()) {
        for (const auto& [k, v] : root_obj.at("metadata").as_object()) {
            if (v.is_string()) {
                result.metadata[k] = v.as_string();
            } else if (v.is_number()) {
                result.metadata[k] = std::to_string(v.as_number());
            }
        }
    }

    // Model State Dict
    if (root_obj.contains("model_state_dict") && root_obj.at("model_state_dict").is_object()) {
        for (const auto& [name, v_entry] : root_obj.at("model_state_dict").as_object()) {
            if (!v_entry.is_object()) continue;
            const auto& entry_obj = v_entry.as_object();
            if (!entry_obj.contains("shape") || !entry_obj.contains("data")) continue;

            std::vector<size_t> shape;
            for (const auto& d : entry_obj.at("shape").as_array()) {
                shape.push_back(static_cast<size_t>(d.as_number()));
            }

            std::vector<double> data;
            for (const auto& val : entry_obj.at("data").as_array()) {
                data.push_back(val.as_number());
            }

            result.model_state_dict[name] = Tensor::create(std::move(shape), std::move(data));
        }

        if (model != nullptr) {
            model->load_state_dict(result.model_state_dict);
        }
    }

    // Optimizer State Dict
    if (root_obj.contains("optimizer_state_dict") && root_obj.at("optimizer_state_dict").is_object()) {
        result.has_optimizer = true;
        const auto& optim_obj = root_obj.at("optimizer_state_dict").as_object();
        if (optim_obj.contains("step_count")) {
            result.optimizer_state_dict.step_count = static_cast<size_t>(optim_obj.at("step_count").as_number());
        }
        if (optim_obj.contains("defaults") && optim_obj.at("defaults").is_object()) {
            for (const auto& [k, v] : optim_obj.at("defaults").as_object()) {
                if (v.is_number()) {
                    result.optimizer_state_dict.defaults[k] = v.as_number();
                }
            }
        }
        if (optim_obj.contains("state") && optim_obj.at("state").is_object()) {
            for (const auto& [idx_str, p_entry] : optim_obj.at("state").as_object()) {
                size_t param_idx = static_cast<size_t>(std::stoul(idx_str));
                if (!p_entry.is_object()) continue;
                optim::ParamState p_state;
                for (const auto& [t_name, t_val] : p_entry.as_object()) {
                    if (!t_val.is_object()) continue;
                    const auto& tobj = t_val.as_object();
                    if (!tobj.contains("shape") || !tobj.contains("data")) continue;

                    std::vector<size_t> shape;
                    for (const auto& d : tobj.at("shape").as_array()) {
                        shape.push_back(static_cast<size_t>(d.as_number()));
                    }
                    std::vector<double> data;
                    for (const auto& d : tobj.at("data").as_array()) {
                        data.push_back(d.as_number());
                    }
                    p_state.tensors[t_name] = Tensor::create(std::move(shape), std::move(data));
                }
                result.optimizer_state_dict.state[param_idx] = std::move(p_state);
            }
        }

        if (optimizer != nullptr) {
            optimizer->load_state_dict(result.optimizer_state_dict);
        }
    }

    return result;
}

} // namespace checkpoint
} // namespace aurora
