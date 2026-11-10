// Reproductions from independent audit 2026-11-10.
// Build: c++ -std=c++17 -O2 -I language/runtime language/tests/runtime_semantic_safety.cpp -o /tmp/nl_safety
// Covers runtime API, including state surviving a C++ exception (unlike CLI).
#include "neural_runtime.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require_test(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
static void near(double actual, double want, const std::string& context) {
  require_test(std::isfinite(actual) && std::abs(actual-want) <= 1e-9,
               context + " actual=" + std::to_string(actual) +
               " expected=" + std::to_string(want));
}

static neural::Model cancellation(bool graph, neural::Model::Mode mode) {
  neural::Model m("Cancellation", 1024);
  if (graph) {
    m.layer_sources("hidden", 64, 1, 11, 0, {"input"});
    m.layer_sources("side", 1, 0, 12, 0, {"input"});
    m.layer_sources("out", 1, 0, 13, 0, {"hidden", "side"});
  } else {
    m.layer("hidden", 2, 1, 11);
    m.layer("out", 1, 0, 12);
  }
  for (auto& layer : m.layers) {
    std::fill(layer.w.begin(), layer.w.end(), 0.);
    std::fill(layer.b.begin(), layer.b.end(), 0.);
  }
  m.layers[0].w[0] = 1.;
  m.layers[0].w[1] = 1.;
  m.layers.back().w[0] = 100000000.;
  m.mode = mode;
  return m;
}
static void check_cancellation(bool graph, neural::Model::Mode mode) {
  auto full = cancellation(graph, neural::Model::FULL);
  auto faster = cancellation(graph, mode);
  std::vector<double> input(1024, 0.);
  input[0] = -900000000.;
  input[1] = 0.00000005;
  full.observe(input);
  faster.observe(input);
  near(full.infer(false)[0], 0., "initial FULL");
  near(faster.infer(false)[0], 0., "initial DELTA/AUTO");
  full.change_input(0, 900000000.);
  faster.change_input(0, 900000000.);
  near(full.infer(false)[0], 5., "post-delta FULL");
  near(faster.infer(false)[0], 5., "post-delta alternative");
  faster.mode = neural::Model::FULL;
  near(faster.infer(false)[0], 5., "FULL after mode switch");
}
static void check_mode_switch_recomputes() {
  auto strict = cancellation(false, neural::Model::DELTA);
  std::vector<double> x(1024, 0.);
  x[0] = .5; x[1] = .25;
  strict.observe(x);
  strict.infer(false);
  strict.change_input(0, .01);
  const double strict_value = strict.infer(false)[0];
  require_test(strict.cache_canonical, "strict delta cache must be canonical");
  const size_t before = strict.counters.full;
  strict.mode = neural::Model::FULL;
  near(strict.infer(false)[0], strict_value, "strict FULL after mode switch");
  require_test(strict.counters.full == before,
               "canonical strict cache should not need a redundant full pass");

  auto experimental = cancellation(false, neural::Model::DELTA);
  experimental.strict_numeric = false;
  experimental.observe(x);
  experimental.infer(false);
  experimental.change_input(0, .01);
  experimental.infer(false);
  require_test(!experimental.cache_canonical,
               "experimental algebraic delta must be marked noncanonical");
  const size_t before_experimental=experimental.counters.full;
  experimental.mode=neural::Model::FULL;
  experimental.infer(false);
  require_test(experimental.counters.full == before_experimental+1,
               "FULL must recompute after experimental noncanonical delta");
}
static void check_mixed_weight_then_input() {
  neural::Model m("Mixed", 1);
  m.layer("out",1,0,11);
  m.layers[0].w[0] = 1.;
  m.layers[0].b[0] = 0.;
  m.mode = neural::Model::DELTA;
  m.observe({600000000.});
  near(m.infer(false)[0], 600000000., "mixed initial");
  m.change("out", 0, 0, 1.);
  // Intermediate (2 * 600M) is invalid, final (2 * 0) is valid.
  m.change_input(0,-600000000.);
  near(m.infer(false)[0], 0., "mixed final");
  near(m.layers[0].w[0], 2., "mixed final weight");
}
static void check_mixed_input_then_weight() {
  neural::Model m("Mixed", 1);
  m.layer("out",1,0,11);
  m.layers[0].w[0] = 2.;
  m.layers[0].b[0] = 0.;
  m.mode = neural::Model::DELTA;
  m.observe({0.});
  near(m.infer(false)[0], 0., "mixed initial");
  m.change_input(0,600000000.);
  // Intermediate (2 * 600M) is invalid, final (1 * 600M) is valid.
  m.change("out",0,0,-1.);
  near(m.infer(false)[0],600000000.,"mixed final");
  near(m.layers[0].w[0],1.,"mixed final weight");
}
static void check_subtle_fp64_cancellation() {
  auto build=[](neural::Model::Mode mode,bool strict){
    neural::Model m("RoundingAmplified",3);
    m.layer("hidden",1,0,19);
    m.layer("out",1,0,20);
    for(auto& layer:m.layers){
      std::fill(layer.w.begin(),layer.w.end(),0.);
      std::fill(layer.b.begin(),layer.b.end(),0.);
    }
    for(int c=0;c<3;c++)m.layers[0].w[c]=1.;
    m.layers[1].w[0]=100000000.;
    m.mode=mode;m.strict_numeric=strict;
    m.observe({-900000000.,0.00000005,900000000.});
    m.infer(false);
    m.change_input(1,0.1);
    return m;
  };
  auto reference=build(neural::Model::FULL,true);
  const double correct=reference.infer(false)[0];
  auto strict=build(neural::Model::DELTA,true);
  const double got=strict.infer(false)[0];
  require_test(got==correct,"canonical partial mismatch at moderate cancellation");
  require_test(strict.counters.input_delta==1,
               "strict incremental path was not actually exercised");
  auto unsafe=build(neural::Model::DELTA,false);
  const double wrong=unsafe.infer(false)[0];
  require_test(std::abs(wrong-correct)>1.0,
               "regression no longer distinguishes historical unsafe algorithm");
  require_test(!unsafe.cache_canonical,"experimental cache provenance not tracked");
  const size_t before=unsafe.counters.full;
  unsafe.strict_numeric=true;
  require_test(unsafe.infer(false)[0]==correct,
               "strict policy toggle exposed prior experimental output");
  require_test(unsafe.counters.full==before+1,
               "strict policy toggle must rerun canonical FULL");
}
static void check_sgd_transaction() {
  neural::Model m("TransactionalSGD",2);
  m.layer("out",1,0,11);
  m.layers[0].w = {0.,0.};
  m.layers[0].b = {0.};
  m.observe({1.,600000000.});
  near(m.infer(false)[0],0.,"sgd initial");
  const auto weights = m.layers[0].w;
  const auto biases = m.layers[0].b;
  const auto version = m.version;
  const auto trained = m.counters.train;
  bool rejected=false;
  try { m.learn({1.},1.); }
  catch(const std::exception&) { rejected=true; }
  require_test(rejected,"SGD invalid update must be rejected");
  require_test(m.layers[0].w == weights && m.layers[0].b == biases,
               "failed SGD mutated weights/biases");
  require_test(m.version == version && m.counters.train == trained,
               "failed SGD modified committed training version/counter");
  near(m.infer(false)[0],0.,"cache after failed SGD");
  m.forward();
  near(m.infer(false)[0],0.,"canonical forward after failed SGD");
}
static void check_dag_sgd_transaction() {
  neural::Model m("TransactionalDAG",2);
  m.layer_sources("left",1,0,11,0,{"input"});
  m.layer_sources("right",1,0,12,0,{"input"});
  m.layer_sources("out",1,0,13,0,{"left","right"});
  for(auto& layer:m.layers){
    std::fill(layer.w.begin(),layer.w.end(),0.);
    std::fill(layer.b.begin(),layer.b.end(),0.);
  }
  m.layers.back().w[0]=1.; // propagate output loss into left branch
  m.observe({1.,600000000.});
  near(m.infer(false)[0],0.,"DAG SGD initial");
  std::vector<std::vector<double>> w_before,b_before;
  for(const auto& layer:m.layers){w_before.push_back(layer.w);b_before.push_back(layer.b);}
  const auto version=m.version, trained=m.counters.train;
  bool rejected=false;
  try{m.learn({1.},1.);}catch(const std::exception&){rejected=true;}
  require_test(rejected,"DAG SGD invalid update must be rejected");
  require_test(m.version==version && m.counters.train==trained,
               "failed DAG SGD committed version/counter");
  for(size_t k=0;k<m.layers.size();k++){
    require_test(m.layers[k].w==w_before[k] && m.layers[k].b==b_before[k],
                 "failed DAG SGD partially wrote layer "+std::to_string(k));
  }
  near(m.infer(false)[0],0.,"DAG cache after failed SGD");
  m.forward();
  near(m.infer(false)[0],0.,"DAG canonical forward after failed SGD");
}
int main() {
  int failures=0;
  auto run = [&](const char* name, const std::function<void()>& test) {
    try { test(); std::cout << "PASS " << name << "\n"; }
    catch(const std::exception& error) {
      ++failures;
      std::cerr << "FAIL " << name << ": " << error.what() << "\n";
    }
  };
  run("cancellation_chain_delta", []{check_cancellation(false,neural::Model::DELTA);});
  run("cancellation_chain_auto", []{check_cancellation(false,neural::Model::AUTO);});
  run("cancellation_dag_delta", []{check_cancellation(true,neural::Model::DELTA);});
  run("cancellation_dag_auto", []{check_cancellation(true,neural::Model::AUTO);});
  run("switch_full_canonicalizes_cache", check_mode_switch_recomputes);
  run("mixed_weight_then_input", check_mixed_weight_then_input);
  run("mixed_input_then_weight", check_mixed_input_then_weight);
  run("subtle_fp64_cancellation", check_subtle_fp64_cancellation);
  run("sgd_transaction_rollback", check_sgd_transaction);
  run("dag_sgd_transaction_rollback", check_dag_sgd_transaction);
  std::cout << "SEMANTIC_SAFETY " << (failures?"FAIL":"PASS") <<
               " cases=10 failed=" << failures << "\n";
  return failures ? 1 : 0;
}
