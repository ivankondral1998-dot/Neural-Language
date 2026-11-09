// Old/new runtime regression: dense, masked, quantized and guarded updates.
#include "neural_runtime.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
int main() {
  try {
    std::cout << std::setprecision(17);
    for (int kind=0;kind<4;kind++) {
      neural::Model m("Probe",4);
      const int storage=(kind==2)?1:0;
      m.layer("hidden",16,3,999,storage);
      m.layer("result",2,0,1000,storage);
      if(kind==1) {
        m.gate("hidden",3,2,false);
        m.gate("result",1,7,false);
      }
      for(int i=0;i<100;i++) {
        if(kind==3 && i==50) {
          m.gate("hidden",3,2,false);
          m.gate("hidden",3,2,true);
        }
        if(kind==3 && i==70) {
          m.change("result",1,7,0.002);
        }
        const std::vector<double> x={
          0.1*std::sin(double(i)),0.1*std::cos(double(i)),
          0.001*i,-0.0007*i};
        m.observe(x);
        const auto& y=m.infer(false);
        for(double v:y) {
          if(!std::isfinite(v))throw std::runtime_error("nonfinite");
          std::cout<<v<<' ';
        }
        std::cout<<'\n';
      }
    }
    std::cout << "400_PROBE_STATES_PASS\n";
  } catch (const std::exception& e) {
    std::cerr << "PROBE_ERROR " << e.what() << '\n'; return 2;
  }
}
