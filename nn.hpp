#include "engine.hpp"
#include <random>

class Module {
public:
    virtual std::vector<std::shared_ptr<Value>> parameters() = 0;
    void zero_grad() {
        for (auto& p : parameters()) p->grad = 0.0;
    }
};

class Neuron : public Module {
public:
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
    bool nonlin;

    Neuron(int nin, bool nonlin = true) : nonlin(nonlin) {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> distribution(-1.0, 1.0);
        
        for (int i = 0; i < nin; i++) {
            w.push_back(std::make_shared<Value>(distribution(generator)));
        }
        b = std::make_shared<Value>(distribution(generator));
    }

    std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>> x) {
        std::shared_ptr<Value> act = b;
        for (size_t i = 0; i < w.size(); i++) {
            act = Value::add(act, Value::mul(w[i], x[i]));
        }
        return nonlin ? Value::relu(act) : act;
    }

    std::vector<std::shared_ptr<Value>> parameters() override {
        std::vector<std::shared_ptr<Value>> all_p = w;
        all_p.push_back(b);
        return all_p;
    }
};

class Layer : public Module {
public:
    std::vector<Neuron> neurons;

    Layer(int nin, int nout, bool nonlin = true) {
        for (int i = 0; i < nout; i++) neurons.push_back(Neuron(nin, nonlin));
    }

    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x) {
        std::vector<std::shared_ptr<Value>> outs;
        for (auto& n : neurons) outs.push_back(n(x));
        return outs;
    }

    std::vector<std::shared_ptr<Value>> parameters() override {
        std::vector<std::shared_ptr<Value>> all_p;
        for (auto& n : neurons) {
            auto p = n.parameters();
            all_p.insert(all_p.end(), p.begin(), p.end());
        }
        return all_p;
    }
};