// Bounded causal greedy generation with explicit API and graph-visit accounting.
#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "ggml.h"
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
struct Meter {
    int phase = 0; // 0=outside a declared call, 1=prefill, 2=single-token decode
    int graphs[3] = {0,0,0};
    int calls[3] = {0,0,0};
    int attempted_tokens[3] = {0,0,0};
    double seconds[3] = {0,0,0};
    int prompt = 0;
    std::ofstream trace;
};
static bool observe(ggml_tensor * tensor, bool ask, void * opaque) {
    if (std::string(tensor->name) != "attn_norm-0") return false;
    if (!ask) ++static_cast<Meter *>(opaque)->graphs[static_cast<Meter *>(opaque)->phase];
    return true;
}
static int process(llama_context * ctx, llama_batch_ext * batch, int tokens, int phase, Meter & m) {
    m.phase = phase;
    const int before = m.graphs[phase];
    const auto start = Clock::now();
    const int result = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_synchronize(ctx);
    const double seconds = std::chrono::duration<double>(Clock::now()-start).count();
    ++m.calls[phase]; m.attempted_tokens[phase] += tokens; m.seconds[phase] += seconds; m.phase = 0;
    m.trace << "{\"prompt_index\":" << m.prompt << ",\"phase\":\"" << (phase==1?"prefill":"decode")
            << "\",\"input_tokens\":" << tokens << ",\"graph_visits\":" << m.graphs[phase]-before
            << ",\"api_exit\":" << result << ",\"synchronized_host_wall_seconds\":" << seconds << "}\n";
    m.trace.flush();
    return result || m.graphs[phase] <= before || !m.trace.good();
}
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC,"C"); common_init(); common_params params;
    if (!common_params_parse(argc,argv,params,LLAMA_EXAMPLE_COMMON)) return 2;
    if (params.n_predict < 1 || params.n_predict > 1024 ||
        !std::filesystem::is_regular_file(params.model.path) ||
        params.speculative.types != std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE} ||
        (!params.lora_adapters.empty() || params.embedding)) {
        std::cerr << "Requires a local GGUF, causal AR only, no adapters and token cap1..1024\n";return 2;
    }
    const char * output = std::getenv("WSP_METER_DIR");
    if (!output || !std::filesystem::is_directory(output) || !std::filesystem::is_empty(output)) return 2;
    std::filesystem::path directory(output); std::vector<std::string> prompts;
    if (const char * list=std::getenv("WSP_METER_LIST")) {
        std::ifstream files(list);if (!files.good()) return 2;std::string path;
        while (std::getline(files,path)) {
            std::ifstream input(path,std::ios::binary);
            if (!input.good() || prompts.size()>=8) return 2;
            prompts.emplace_back(std::istreambuf_iterator<char>(input),std::istreambuf_iterator<char>());
        }
    } else prompts.push_back(params.prompt);
    if (prompts.empty()) return 2;
    for (const auto & p:prompts) if (p.empty() || p.size()>65536) return 2;
    Meter meter;meter.trace.open(directory/"calls.jsonl");if (!meter.trace.good()) return 2;
    params.warmup=false;params.cb_eval=observe;params.cb_eval_user_data=&meter;
    // This measurement profile deliberately ignores CLI sampler overrides.
    params.sampling=common_params_sampling{};
    std::cerr << "Fixed greedy measurement profile; sampler CLI overrides ignored\n";
    params.sampling.temp=0;params.sampling.top_k=1;params.sampling.top_p=1;params.sampling.min_p=0;
    params.sampling.penalty_repeat=1;params.sampling.penalty_freq=0;params.sampling.penalty_present=0;
    params.sampling.ignore_eos=false;params.sampling.seed=42;
    llama_backend_init();llama_numa_init(params.numa);int result=0,completed=0,total_input=0,sampled=0,emitted=0;
    int effective_batch=0,effective_ubatch=0,effective_ctx=0;
    {
        auto initialized=common_init_from_params(params);if (!initialized) return 2;auto * ctx=initialized->context();auto * model=initialized->model();
        if (!ctx || !model || llama_model_has_encoder(model) || !llama_get_causal_attn(ctx)) return 2;
        effective_batch=llama_n_batch(ctx);effective_ubatch=llama_n_ubatch(ctx);effective_ctx=llama_n_ctx_seq(ctx);
        const auto * vocab=llama_model_get_vocab(model);std::ofstream index(directory/"generations.jsonl");
        for (const auto & prompt:prompts) {
            auto tokens=common_tokenize(ctx,prompt,llama_vocab_get_add_bos(vocab),true);
            if (tokens.empty() || tokens.size()>4096 || tokens.size()+params.n_predict>llama_n_ctx_seq(ctx)) {result=2;break;}
            llama_memory_clear(llama_get_memory(ctx),true);
            auto sampling=params.sampling;
            std::unique_ptr<common_sampler,decltype(&common_sampler_free)> sampler(common_sampler_init(model,sampling),common_sampler_free);
            if (!sampler) {result=2;break;}
            for (auto token:tokens) common_sampler_accept(sampler.get(),token,false);
            common_batch batch(ctx);
            for (int i=0;i<(int)tokens.size();++i) batch.add(tokens[i],i,0,i+1==(int)tokens.size());
            const int pc=meter.calls[1],dc=meter.calls[2],pg=meter.graphs[1],dg=meter.graphs[2];
            total_input+=tokens.size();
            for (int off=0;off<(int)tokens.size();off+=llama_n_batch(ctx)) {
                const int count=std::min((int)llama_n_batch(ctx),(int)tokens.size()-off);
                result=process(ctx,batch.get_sub_batch(off,count),count,1,meter);
                if (result) break;
            }
            if (result) break;
            std::ofstream content(directory/(std::to_string(meter.prompt)+"-content.bin"),std::ios::binary);
            std::ofstream ids(directory/(std::to_string(meter.prompt)+"-tokens.json"));ids<<"[";
            int count=0,nsampled=0;bool eos=false;
            for (int step=0;step<params.n_predict;++step) {
                auto token=common_sampler_sample(sampler.get(),ctx,-1);common_sampler_accept(sampler.get(),token,true);
                if (step) ids<<",";ids<<token;++nsampled;
                if (llama_vocab_is_eog(vocab,token)) {eos=true;break;}
                auto piece=common_token_to_piece(vocab,token,false);content.write(piece.data(),piece.size());++count;
                if (step+1==params.n_predict) break;
                batch.clear();batch.add(token,tokens.size()+step,0,true);
                result=process(ctx,batch.get(),1,2,meter);if (result) break;
            }
            ids<<"]\n";content.flush();ids.flush();
            if (!content.good() || !ids.good()) result=1;
            index<<"{\"prompt_index\":"<<meter.prompt<<",\"prompt_tokens\":"<<tokens.size()
                 <<",\"sampled_including_eog\":"<<nsampled<<",\"emitted_tokens\":"<<count
                 <<",\"eog\":"<<(eos?"true":"false")<<",\"prefill_api_calls\":"<<meter.calls[1]-pc
                 <<",\"decode_api_calls\":"<<meter.calls[2]-dc<<",\"prefill_graph_visits\":"<<meter.graphs[1]-pg
                 <<",\"decode_graph_visits\":"<<meter.graphs[2]-dg<<",\"api_exit\":"<<result<<"}\n";index.flush();
            sampled+=nsampled;emitted+=count;
            if (result || !index.good()) {result=1;break;}
            ++completed;++meter.prompt;
        }
        llama_perf_context_print(ctx);
    }
    if (meter.graphs[0] || completed!=(int)prompts.size()) result=1;
    std::ofstream status(directory/"meter-status.json");
    status<<"{\"completed_prompts\":"<<completed<<",\"prompt_tokens_submitted\":"<<total_input
          <<",\"attempted_prefill_input_tokens\":"<<meter.attempted_tokens[1]<<",\"attempted_decode_input_tokens\":"<<meter.attempted_tokens[2]
          <<",\"token_cap\":"<<params.n_predict<<",\"temperature\":0,\"top_k\":1,\"seed\":42"
          <<",\"n_batch_effective\":"<<effective_batch<<",\"n_ubatch_effective\":"<<effective_ubatch<<",\"n_ctx_effective\":"<<effective_ctx
          <<",\"sampled_including_eog\":"<<sampled<<",\"emitted_tokens\":"<<emitted
          <<",\"prefill_api_calls\":"<<meter.calls[1]<<",\"decode_api_calls\":"<<meter.calls[2]
          <<",\"prefill_graph_visits\":"<<meter.graphs[1]<<",\"decode_graph_visits\":"<<meter.graphs[2]
          <<",\"undeclared_graph_visits\":"<<meter.graphs[0]<<",\"process_exit\":"<<result<<"}\n";
    status.flush();if (!status.good()) result=1;
    llama_backend_free();return result;
}
