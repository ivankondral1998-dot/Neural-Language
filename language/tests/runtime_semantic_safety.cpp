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
  auto model = cancellation(false, neural::Model::DELTA);
  std::vector<double> x(1024, 0.);
  x[0] = .5;
  x[1] = .25;
  model.observe(x);
  model.infer(false);
  model.change_input(0, .01);
  model.infer(false);
  const size_t before = model.counters.full;
  model.mode = neural::Model::FULL;
  model.infer(false);
  require_test(model.counters.full == before+1,
               "switch to FULL must canonicalize noncanonical cache");
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
static void check_simple_delta_value() {
  neural::Model fast("InputDelta", 2), full("InputFull", 2);
  fast.layer("out", 1, 0, 11);
  full.layer("out", 1, 0, 11);
  for(auto* model : {&fast, &full}) {
    model->layers[0].w={1.,0.};
    model->layers[0].b={0.};
    model->observe({0.25,0.});
  }
  fast.mode=neural::Model::DELTA;
  full.mode=neural::Model::FULL;
  near(fast.infer(false)[0],0.25,"delta original");
  near(full.infer(false)[0],0.25,"full original");
  fast.change_input(0,0.5);
  full.change_input(0,0.5);
  near(full.infer(false)[0],0.75,"full after input delta");
  near(fast.infer(false)[0],0.75,"delta after input delta");
  require_test(fast.counters.input_delta>0,"simple delta must exercise incremental path");
}
static void check_hidden_rounding_cancellation() {
  neural::Model fast("Roundoff",3), full("Roundoff",3);
  for(auto* model : {&fast,&full}) {
    model->layer("out",1,0,11);
    model->layers[0].w={1.e16,1.,-1.e16};
    model->layers[0].b={0.};
    model->observe({1.,0.,1.});
  }
  fast.mode=neural::Model::DELTA;
  full.mode=neural::Model::FULL;
  near(fast.infer(false)[0],0.,"initial rounding DELTA");
  near(full.infer(false)[0],0.,"initial rounding FULL");
  fast.change_input(1,1.);
  full.change_input(1,1.);
  near(full.infer(false)[0],0.,"canonical cancellation");
  near(fast.infer(false)[0],0.,"incremental cancellation");
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
  run("sgd_transaction_rollback", check_sgd_transaction);
  run("dag_sgd_transaction_rollback", check_dag_sgd_transaction);
  run("simple_input_delta_matches_full", check_simple_delta_value);
  run("hidden_roundoff_cancellation", check_hidden_rounding_cancellation);
  std::cout << "SEMANTIC_SAFETY " << (failures?"FAIL":"PASS") <<
               " cases=11 failed=" << failures << "\n";
  return failures ? 1 : 0;
}
