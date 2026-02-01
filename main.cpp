#include "nn.hpp"
#include <iostream>
#include <vector>

class MLP : public Module {
public:
    std::vector<Layer> layers;

    MLP(int nin, std::vector<int> nouts) {
        int sz = nin;
        for (int nout : nouts) {
            layers.push_back(Layer(sz, nout, &nout != &nouts.back()));
            sz = nout;
        }
    }

    std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x) {
        for (auto& layer : layers) {
            x = layer(x);
        }
        return x;
    }

    std::vector<std::shared_ptr<Value>> parameters() override {
        std::vector<std::shared_ptr<Value>> all_p;
        for (auto& layer : layers) {
            auto p = layer.parameters();
            all_p.insert(all_p.end(), p.begin(), p.end());
        }
        return all_p;
    }
};

int main() {
    // Dataset de treinamento (XOR-like problem)
    std::vector<std::vector<double>> xs = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };
    
    std::vector<double> ys = {1.0, -1.0, -1.0, 1.0};

    // Criando MLP: 3 inputs -> 4 hidden -> 4 hidden -> 1 output
    MLP model(3, {4, 4, 1});

    std::cout << "Total de parametros: " << model.parameters().size() << std::endl;
    std::cout << "\n=== Iniciando Treinamento ===\n" << std::endl;

    double learning_rate = 0.01;
    int epochs = 100;

    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        std::vector<std::shared_ptr<Value>> ypred;
        for (auto& x : xs) {
            std::vector<std::shared_ptr<Value>> inputs;
            for (double val : x) {
                inputs.push_back(std::make_shared<Value>(val));
            }
            auto pred = model(inputs);
            ypred.push_back(pred[0]);
        }

        // Calcular loss (Mean Squared Error)
        auto loss = std::make_shared<Value>(0.0);
        for (size_t i = 0; i < ys.size(); i++) {
            auto diff = Value::add(ypred[i], std::make_shared<Value>(-ys[i]));
            auto squared = Value::mul(diff, diff);
            loss = Value::add(loss, squared);
        }

        // Backward pass
        model.zero_grad();
        loss->backward();

        // Update (Gradient Descent)
        for (auto& p : model.parameters()) {
            p->data += -learning_rate * p->grad;
        }

        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << loss->data << std::endl;
        }
    }

    std::cout << "\n=== Treinamento Finalizado ===\n" << std::endl;
    std::cout << "Testando predicoes finais:" << std::endl;

    for (size_t i = 0; i < xs.size(); i++) {
        std::vector<std::shared_ptr<Value>> inputs;
        for (double val : xs[i]) {
            inputs.push_back(std::make_shared<Value>(val));
        }
        auto pred = model(inputs);
        std::cout << "Input [" << xs[i][0] << ", " << xs[i][1] << ", " << xs[i][2] 
                  << "] -> Pred: " << pred[0]->data << " | Target: " << ys[i] << std::endl;
    }

    return 0;
}