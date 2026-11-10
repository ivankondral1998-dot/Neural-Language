// Regression reproducer from GPT-6.1 Work audit, 2026-11-10.
// Run this against parent/runtime and patched runtime; parent must fail.
// No external dependencies, no model file; directly sets known FP64 weights.
#include "neural_runtime.hpp"
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool p, const std::string& what) {
  if (!p) throw std::runtime_error(what);
}
static void equal(double actual, double expected, const std::string& what) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-9)
    throw std::runtime_error(what + ": got=" + std::to_string(actual)
                                + " expected=" + std::to_string(expected));
}
static void zero_weights(neural::Model& m) {
  for (auto& layer : m.layers) {
    for(auto& w : layer.w) w = 0.0;
    for(auto& bias : layer.b) bias = 0.0;
  }
}

static neural::Model make_cancellation(bool graph) {
  neural::Model m("Cancellation",1024);
  if(graph) {
    m.layer_sources("hidden",64,1,42,0,{"input"});
    m.layer_sources("side",1,0,43,0,{"input"});
    m.layer_sources("out",1,0,44,0,{"hidden","side"});
  } else {
    m.layer("hidden",2,1,42);
    m.layer("out",1,0,43);
  }
  zero_weights(m);
  m.layers[0].w[0]=1.0; // input[0]
  m.layers[0].w[1]=1.0; // input[1]
  m.layers.back().w[0]=100000000.0;
  return m;
}

static void cancellation(bool graph,neural::Model::Mode mode) {
  auto m=make_cancellation(graph);
  m.mode=mode;
  std::vector<double> x(1024,0.0);
  x[0]=-900000000.0;
  x[1]=0.00000005;
  m.observe(x);
  equal(m.infer(false)[0],0.0,"pre-change prediction");
  m.change_input(0,900000000.0);
  equal(m.infer(false)[0],5.0,"FULL/DELTA cancellation mismatch");
  m.mode=neural::Model::FULL;
  equal(m.infer(false)[0],5.0,"mode switch preserves approximate cache");
}

static void mode_switch_recanonicalizes() {
  neural::Model m("Switch",2);
  m.layer("out",1,0,3);
  zero_weights(m);
  m.layers[0].w[0]=1.0;
  m.mode=neural::Model::DELTA;
  m.observe({0.2,0.0});
  m.infer(false);
  m.change_input(0,0.1);
  equal(m.infer(false)[0],0.3,"delta update");
  check(m.counters.input_delta==1,"expected actual incremental step");
  const auto prior_full=m.counters.full;
  m.mode=neural::Model::FULL;
  equal(m.infer(false)[0],0.3,"FULL after DELTA");
  check(m.counters.full==prior_full+1,"FULL must recanonicalize after DELTA");
}

static void mixed_events_defer_intermediate_error() {
  neural::Model m("MixedEvents",1);
  m.layer("out",1,0,3);
  zero_weights(m);
  m.layers[0].w[0]=1.0;
  m.mode=neural::Model::AUTO;
  m.observe({600000000.0});
  equal(m.infer(false)[0],600000000.0,"mixed precondition");
  m.change("out",0,0,1.0);      // pending new weight=2, intermediate prediction 1.2e9
  m.change_input(0,-600000000.0); // final valid input=0; MUST NOT evaluate intermediate
  equal(m.infer(false)[0],0.0,"mixed events final result");
}

static void failed_learn_is_atomic(bool graph) {
  neural::Model m("TrainingException",2);
  if(graph) {
    m.layer_sources("first",1,0,3,0,{"input"});
    m.layer_sources("out",1,0,4,0,{"first","input"});
  } else {
    m.layer("out",1,0,3);
  }
  zero_weights(m);
  m.observe({1.0,600000000.0});
  equal(m.infer(false)[0],0.0,"learn precondition");
  const auto version=m.version;
  const auto train=m.counters.train;
  const auto full=m.counters.full;
  bool failed=false;
  try {
    m.learn({1.0},1.0);
  } catch (const std::runtime_error&) { failed=true; }
  check(failed,"unsafe SGD gradient must be rejected");
  check(m.version==version,"SGD exception changed version");
  check(m.counters.train==train,"SGD exception changed train count");
  check(m.counters.full==full,"SGD exception changed full count");
  for(const auto& l:m.layers) {
    for(double w:l.w)equal(w,0.0,"SGD exception partially changed weights");
    for(double b:l.b)equal(b,0.0,"SGD exception partially changed bias");
  }
  equal(m.infer(false)[0],0.0,"stale cache after partial SGD");
  m.forward();
  equal(m.infer(false)[0],0.0,"fresh forward changed after SGD failure");
}

int main() {
  try {
    cancellation(false,neural::Model::DELTA);
    std::cout<<"CANCELLATION_CHAIN_DELTA_PASS\n";
    cancellation(false,neural::Model::AUTO);
    std::cout<<"CANCELLATION_CHAIN_AUTO_PASS\n";
    cancellation(true,neural::Model::DELTA);
    std::cout<<"CANCELLATION_DAG_DELTA_PASS\n";
    cancellation(true,neural::Model::AUTO);
    std::cout<<"CANCELLATION_DAG_AUTO_PASS\n";
    mode_switch_recanonicalizes();
    std::cout<<"MODE_SWITCH_RECANONICALIZE_PASS\n";
    mixed_events_defer_intermediate_error();
    std::cout<<"MIXED_EVENTS_PASS\n";
    failed_learn_is_atomic(false);
    std::cout<<"SGD_CHAIN_ATOMIC_PASS\n";
    failed_learn_is_atomic(true);
    std::cout<<"SGD_DAG_ATOMIC_PASS\n";
    std::cout<<"AUDIT_RUNTIME_REGRESSIONS_PASS\n";
    return 0;
  } catch(const std::exception& e) {
    std::cerr<<"AUDIT_RUNTIME_REGRESSIONS_FAIL "<<e.what()<<'\n';
    return 2;
  }
}
