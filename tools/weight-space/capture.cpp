// Bounded observation through llama.cpp's documented evaluation callback API.
#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <clocale>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

struct Capture {
    std::filesystem::path directory;
    std::ofstream index;
    int count = 0;
    bool failed = false;
};

static bool observe(ggml_tensor * tensor, bool ask, void * opaque) {
    auto & state = *static_cast<Capture *>(opaque);
    static const std::set<std::string> names = {"l_out-31", "l_out-35", "l_out-39"};
    if (ask) {
        return names.count(tensor->name) > 0;
    }
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        state.failed = true;
        return false;
    }
    const auto bytes = ggml_nbytes(tensor);
    if (bytes > 64 * 1024 * 1024) {
        state.failed = true;
        return false;
    }
    std::vector<char> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
    const std::string filename = std::to_string(state.count++) + "-" + tensor->name + ".f32";
    std::ofstream output(state.directory / filename, std::ios::binary);
    output.write(data.data(), data.size());
    state.failed = state.failed || !output.good();
    state.index << "{\"tensor\":\"" << tensor->name << "\",\"file\":\"" << filename
                << "\",\"shape\":[" << tensor->ne[0] << "," << tensor->ne[1] << ","
                << tensor->ne[2] << "," << tensor->ne[3] << "],\"bytes\":" << bytes << "}\n";
    state.index.flush();
    return !state.failed;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 2;
    }
    const char * directory = std::getenv("WSP_CAPTURE_DIR");
    if (!directory || !std::filesystem::is_directory(directory) ||
        !std::filesystem::is_empty(directory)) {
        std::cerr << "WSP_CAPTURE_DIR must name an existing empty directory\n";
        return 2;
    }
    Capture state;
    state.directory = directory;
    state.index.open(state.directory / "tensors.jsonl");
    if (!state.index.good()) {
        return 2;
    }
    params.cb_eval = observe;
    params.cb_eval_user_data = &state;
    params.warmup = false;
    llama_backend_init();
    llama_numa_init(params.numa);
    int result = 0;
    {
        auto initialized = common_init_from_params(params);
        auto * context = initialized->context();
        if (!context || !initialized->model()) {
            return 2;
        }
        const auto * vocab = llama_model_get_vocab(initialized->model());
        auto tokens = common_tokenize(context, params.prompt, llama_vocab_get_add_bos(vocab), true);
        if (tokens.empty() || tokens.size() > 4096) {
            return 2;
        }
        auto batch = common_batch_get_one(context, tokens);
        // One explicit prefill API call; no generated tokens, loss, or gradient.
        result = llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        llama_synchronize(context);
        llama_perf_context_print(context);
        std::ofstream status(state.directory / "capture-status.json");
        status << "{\"prefill_api_calls\":1,\"prompt_tokens\":" << tokens.size()
               << ",\"generated_tokens\":0,\"captured_tensors\":" << state.count
               << ",\"process_exit\":" << result << ",\"capture_failed\":"
               << (state.failed ? "true" : "false") << "}\n";
        if (state.failed || state.count == 0) {
            result = 1;
        }
    }
    llama_backend_free();
    return result;
}
